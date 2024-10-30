import streamlit as st
from openai import OpenAI
import anthropic
import time
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from bert_score import score
import torch
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import google.generativeai as genai
import requests
import numpy as np
import math
from collections import Counter
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from google.api_core.exceptions import InternalServerError
import ast
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from mistralai import Mistral, UserMessage
import logging
import seaborn as sns
from functools import wraps
from collections import defaultdict
import spacy

# Add this near the top of your file, after the imports
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# Function to download NLTK data
def download_nltk_data():
    nltk.download('punkt', quiet=True)

download_nltk_data()

# Helper functions for BLEU calculation
def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def modified_precision(reference_tokens, candidate_tokens, n):
    reference_count = Counter(ngrams(reference_tokens, n))
    candidate_count = Counter(ngrams(candidate_tokens, n))
    clipped_count = {ngram: min(count, reference_count.get(ngram, 0)) for ngram, count in candidate_count.items()}
    numerator = sum(clipped_count.values())
    denominator = max(1, sum(candidate_count.values()))
    return numerator / denominator if denominator != 0 else 0

def brevity_penalty(reference_tokens, candidate_tokens):
    c = len(candidate_tokens)
    r = len(reference_tokens)
    if c > r:
        return 1
    else:
        return math.exp(1 - r/c)

def calculate_bleu(reference, candidate, weights=[0.7, 0.3]):
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    p_n = [modified_precision(reference_tokens, candidate_tokens, n) for n in range(1, len(weights) + 1)]
    bp = brevity_penalty(reference_tokens, candidate_tokens)
    s = math.exp(sum(w * math.log(p + 1e-10) if p > 0 else float('-inf') for w, p in zip(weights, p_n)))
    return bp * s

# Sidebar for API keys
st.sidebar.header("Additional API Keys (Optional)")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
mistral_api_key = st.sidebar.text_input("Mistral API Key", type="password")


# Sidebar for regeneration options
st.sidebar.header("Regeneration Options")
regeneration_method = st.sidebar.radio(
    "Choose regeneration method:",
    ("Summarize", "Paraphrase")
)

# Add this to the sidebar options, after the regeneration method selection
st.sidebar.header("Detection Options")
include_human_detection = st.sidebar.checkbox("Include Human Authorship Detection", value=False)

# Load models and clients
@st.cache_resource
def load_openai_client():
    logger.info("Loading OpenAI client...")
    start_time = time.time()
    client = OpenAI(api_key=openai_api_key or st.secrets["OPENAI_API_KEY"])
    logger.info(f"OpenAI client loaded in {time.time() - start_time:.2f} seconds")
    return client

@st.cache_resource
def load_anthropic_client():
    logger.info("Loading Anthropic client...")
    start_time = time.time()
    client = anthropic.Anthropic(api_key=anthropic_api_key or st.secrets["ANTHROPIC_API_KEY"])
    logger.info(f"Anthropic client loaded in {time.time() - start_time:.2f} seconds")
    return client

@st.cache_resource
def load_gpt2():
    logger.info("Loading GPT-2 model...")
    start_time = time.time()
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    logger.info(f"GPT-2 model loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer

@st.cache_resource
def load_gemini_client():
    logger.info("Loading Gemini client...")
    start_time = time.time()
    try:
        genai.configure(api_key=gemini_api_key or st.secrets["GEMINI_API_KEY"])
        client = genai.GenerativeModel('gemini-1.5-pro')
        logger.info(f"Gemini client loaded in {time.time() - start_time:.2f} seconds")
        return client
    except Exception as e:
        logger.error(f"Failed to load Gemini client: {e}")
        return None

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

def extract_human_features(text):
    """Extract linguistic features that are typically human-like"""
    nlp = load_spacy_model()
    doc = nlp(text)
    
    features = {
        'sentence_length_variance': np.var([len(sent) for sent in doc.sents]),
        'unique_words_ratio': len(set(token.text.lower() for token in doc)) / len(doc),
        'punctuation_ratio': len([token for token in doc if token.is_punct]) / len(doc),
        'named_entity_ratio': len(doc.ents) / len(doc),
        'stopwords_ratio': len([token for token in doc if token.is_stop]) / len(doc),
        'avg_word_length': np.mean([len(token.text) for token in doc]),
    }
    
    return features

def calculate_human_similarity(text):
    """Calculate similarity to typical human writing patterns"""
    features = extract_human_features(text)
    
    # Adjusted benchmarks based on typical human writing
    human_benchmarks = {
        'sentence_length_variance': 25.0,
        'unique_words_ratio': 0.6,
        'punctuation_ratio': 0.12,
        'named_entity_ratio': 0.05,
        'stopwords_ratio': 0.4,
        'avg_word_length': 4.7,
    }
    
    # Calculate similarity scores with stricter thresholds
    similarities = {}
    for feature, value in features.items():
        benchmark = human_benchmarks[feature]
        # More stringent similarity calculation
        diff = abs(value - benchmark) / benchmark
        similarity = max(0, 1 - (diff * 2))  # Multiply diff by 2 to make it stricter
        similarities[feature] = similarity
    
    # Apply a scaling factor to reduce overall human probability
    human_score = np.mean(list(similarities.values())) * 0.5  # Reduce by 50%
    return human_score

# Add this function after the model loading functions and before the text generation functions
def generate_text(model_name):
    return all_models[model_name]

# Text generation functions for different models
def generate_text_openai(prompt):
    client = load_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1000,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def generate_text_claude(prompt):
    client = load_anthropic_client()
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0.7,   
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text.strip()

def generate_text_gemini(prompt):
    model = load_gemini_client()
    if model is None:
        return "Gemini model client is not available."
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.7
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"Failed to generate text using Gemini: {e}"

def rate_limit(max_per_second):
    min_interval = 1.0 / max_per_second
    def decorator(func):
        last_called = [0.0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@rate_limit(4.5)  # Set to 4.5 to be safe, giving some buffer
def generate_text_mistral(prompt):
    client = Mistral(api_key=mistral_api_key or st.secrets["MISTRAL_API_KEY"])
    model = "mistral-large-latest"

    messages = [
        {
            "role": "user",
            "content": f"{prompt}",
            "max_tokens": 1000,
            "temperature": 0.7,
        },
    ]

    chat_response = client.chat.complete(
        model=model,
        messages=messages,
    )
    
    return chat_response.choices[0].message.content

# Mapping of models to their generation functions
all_models = {
    "OpenAI": generate_text_openai,
    "Claude": generate_text_claude,
    "Gemini": generate_text_gemini,
    "Mistral": generate_text_mistral,
}

available_models = list(all_models.keys())

# Functions for iterative regeneration
def iterative_regeneration(initial_text, model_func, model_name, iterations=5):
    current_text = initial_text
    regenerations = []
    progress_bar = st.progress(0)
    status_area = st.empty()
    for i in range(iterations):
        # Stage I: Generation
        if regeneration_method == "Summarize":
            prompt = f"You are a professional language facilitator. You should summarize the following document using one sentence:\n\n{current_text}"
        else:
            prompt = f"You are a professional language facilitator. You should paraphrase the following sentence and output the final result only:\n\n{current_text}"
        current_text = model_func(prompt)
        regenerations.append(current_text)
        progress = (i + 1) / iterations
        progress_bar.progress(progress)
        status_area.markdown(f"**{model_name} - Iteration {i+1}:** {current_text}")
        time.sleep(0.1)
    progress_bar.empty()
    status_area.empty()
    return regenerations

# Functions for verification
def verification_step(final_output, authentic_model_func, contrasting_model_funcs, model_names):
    # Stage II: Verification
    if regeneration_method == "Summarize":
        prompt = f"You are a professional language facilitator. You should summarize the following document using one sentence:\n\n{final_output}"
    else:
        prompt = f"You are a professional language facilitator. You should paraphrase the following sentence and output the final result only:\n\n{final_output}"
    
    # Re-generation by the authentic model
    y_a = authentic_model_func(prompt)
    # Re-generation by contrasting models
    y_cs = [model_func(prompt) for model_func in contrasting_model_funcs]
    return y_a, y_cs

# Scoring functions
def calculate_metrics(reference, candidates):
    metrics = []
    for candidate in candidates:
        bertscore = calculate_bertscore(reference, candidate)
        cosine = calculate_cosine_similarity(reference, candidate)
        rouge = calculate_rouge(reference, candidate)
        bleu = calculate_bleu(reference, candidate, weights=[0.7, 0.3])  # Use bigram weights with more emphasis on unigrams
        meteor = calculate_meteor(reference, candidate)
        perplexity = calculate_perplexity(candidate)
        metrics.append({
            'BERTScore': bertscore,
            'Cosine Similarity': cosine,
            'ROUGE-L': rouge,
            'BLEU': bleu,
            'METEOR': meteor,
            'Perplexity': perplexity
        })
    return metrics

def calculate_bertscore(reference, candidate):
    P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
    return F1.item()

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]

def calculate_perplexity(text):
    model, tokenizer = load_gpt2()
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
    return torch.exp(outputs.loss).item()

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rougeL'].fmeasure

def calculate_meteor(reference, candidate):
    reference_tokens = word_tokenize(reference)
    candidate_tokens = word_tokenize(candidate)
    return meteor_score([reference_tokens], candidate_tokens)

# Add this function before verify_authorship
def verify_authorship_core(text, authentic_model, authentic_name, all_models, iterations=5):
    """Core verification logic without human verification"""
    authorship_iterations = []
    verification_iterations = []
    
    # Create progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate total steps (iterations + verification steps)
    total_steps = iterations + len(all_models)
    current_step = 0
    
    # Stage I: Iterative Regeneration
    status_text.text("Stage I: Performing iterative regeneration...")
    current_text = text
    for i in range(iterations):
        if regeneration_method == "Summarize":
            prompt = f"Please summarize the following text into one sentence:\n\n{current_text}"
        else:
            prompt = f"Please paraphrase the following text, maintaining its original meaning:\n\n{current_text}"
        current_text = authentic_model(prompt)
        authorship_iterations.append({
            'iteration': i + 1,
            'text': current_text
        })
        
        # Update progress
        current_step += 1
        progress_bar.progress(current_step / total_steps)
    
    final_output = current_text
    
    # Stage II: Verification
    status_text.text("Stage II: Performing verification steps...")
    contrasting_models = {name: func for name, func in all_models.items() if name != authentic_name}
    if regeneration_method == "Summarize":
        prompt = f"Please summarize the following text into one sentence:\n\n{final_output}"
    else:
        prompt = f"Please paraphrase the following text, maintaining its original meaning:\n\n{final_output}"
    
    # Authentic model verification
    y_a = authentic_model(prompt)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # Contrasting models verification
    y_cs = []
    for name, model in contrasting_models.items():
        status_text.text(f"Verifying against {name}...")
        y_cs.append(model(prompt))
        current_step += 1
        progress_bar.progress(current_step / total_steps)
    
    verification_iterations.append({
        'authentic_output': y_a,
        'contrasting_outputs': dict(zip(contrasting_models.keys(), y_cs))
    })
    
    # Calculate metrics
    status_text.text("Calculating metrics...")
    authentic_metrics = calculate_metrics(final_output, [y_a])[0]
    contrasting_metrics = calculate_metrics(final_output, y_cs)
    
    # Calculate authorship probability
    probabilities, weighted_scores, weights = calculate_authorship_probability(authentic_metrics, contrasting_metrics)
    model_names = [authentic_name] + list(contrasting_models.keys())
    authorship_result = model_names[np.argmax(probabilities)]
    
    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return (
        prompt,
        authorship_iterations,
        probabilities,
        authorship_result,
        model_names,
        authentic_metrics,
        contrasting_metrics,
        verification_iterations,
        weighted_scores,
        weights
    )

def verify_authorship(text, authentic_model, authentic_name, all_models, iterations=5):
    """Wrapper function that can optionally include human verification"""
    # Get core verification results
    results = verify_authorship_core(
        text, authentic_model, authentic_name, all_models, iterations
    )
    
    # Unpack results
    (prompt, authorship_iterations, probabilities, authorship_result, 
     model_names, authentic_metrics, contrasting_metrics, 
     verification_iterations, weighted_scores, weights) = results
    
    if include_human_detection:
        # Add human similarity score
        human_score = calculate_human_similarity(text)
        
        # Add human weighted scores
        human_weighted_scores = np.zeros((1, weighted_scores.shape[1]))
        weighted_scores = np.vstack([weighted_scores, human_weighted_scores])
        
        # Adjust probabilities to include human probability with threshold
        all_probabilities = np.append(probabilities, human_score)
        all_model_names = model_names + ['Human']
        
        # Only consider human authorship if the score is significantly high
        HUMAN_THRESHOLD = 0.8
        if human_score < HUMAN_THRESHOLD:
            # If below threshold, redistribute human probability to AI models
            human_prob = all_probabilities[-1]
            all_probabilities = all_probabilities[:-1]
            # Redistribute human probability proportionally to AI models
            all_probabilities += (human_prob * (all_probabilities / np.sum(all_probabilities)))
            all_probabilities = np.append(all_probabilities, 0.0)  # Add back human with 0 probability
        
        # Normalize probabilities
        all_probabilities = all_probabilities / np.sum(all_probabilities)
        
        # Determine final authorship with stricter human threshold
        max_prob_idx = np.argmax(all_probabilities)
        final_authorship = "Human" if (max_prob_idx == len(all_model_names) - 1 and human_score >= HUMAN_THRESHOLD) else all_model_names[max_prob_idx]
    else:
        all_probabilities = probabilities
        all_model_names = model_names
        final_authorship = model_names[np.argmax(probabilities)]
    
    return (
        prompt, 
        authorship_iterations, 
        all_probabilities, 
        final_authorship, 
        all_model_names, 
        authentic_metrics, 
        contrasting_metrics, 
        verification_iterations, 
        weighted_scores,
        weights
    )

def calculate_authorship_probability(authentic_metrics, contrasting_metrics):
    weights = np.array([0.3, 0.3, 0.15, 0.15, 0.1, 0.05])  # Adjusted weights for ROUGE and BLEU
    all_metrics = [authentic_metrics] + contrasting_metrics
    scores = []
    for metric in all_metrics:
        score_values = list(metric.values())
        # Invert perplexity
        score_values[5] = 1 / (1 + score_values[5])
        scores.append(score_values)
    normalized_scores = normalize_scores(np.array(scores))
    weighted_scores = normalized_scores * weights
    final_scores = weighted_scores.sum(axis=1)
    probabilities = np.exp(final_scores) / np.sum(np.exp(final_scores))
    return probabilities, weighted_scores, weights

def normalize_scores(scores):
    normalized = np.array(scores, dtype=float)
    for i in range(normalized.shape[1]):
        min_val = np.min(normalized[:, i])
        max_val = np.max(normalized[:, i])
        if max_val > min_val:
            normalized[:, i] = (normalized[:, i] - min_val) / (max_val - min_val)
        else:
            normalized[:, i] = 1
    return normalized

def determine_authorship(probabilities, model_names):
    max_index = np.argmax(probabilities)
    if max_index == 0:
        return "Authentic"
    else:
        return f"Contrasting ({model_names[max_index]})"

def serialize_metrics(metrics):
    serialized = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)):
            serialized[key] = value
        elif isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        else:
            serialized[key] = str(value)
    return serialized

# Streamlit UI components
st.title("Text Input Options")

input_option = st.radio(
    "Choose input method:",
    ("Enter text manually", "Generate text using a model")
)

if input_option == "Generate text using a model":
    generation_model = st.selectbox(
        "Select model for text generation",
        available_models
    )
    example_prompts = [
        "Explain the concept of quantum entanglement in simple terms.",
        "Write a short story about a time traveler who accidentally changes history.",
        "Describe the process of photosynthesis in plants.",
        "Create a recipe for a unique fusion dish combining Italian and Japanese cuisines."
    ]
    selected_prompt = st.selectbox(
        "Select an example prompt or write your own:",
        ["Write your own prompt..."] + example_prompts
    )
    if selected_prompt == "Write your own prompt...":
        prompt = st.text_area("Enter your prompt for text generation")
    else:
        prompt = st.text_area("Enter your prompt for text generation", value=selected_prompt)
    if st.button("Generate Text"):
        try:
            generated_text = generate_text(generation_model)(prompt)
            st.write("Generated Text:")
            st.write(generated_text)
            st.session_state.generated_text = generated_text
        except Exception as e:
            st.error(f"Error generating text: {str(e)}")

if 'input_text' not in st.session_state:
    st.session_state.input_text = "Artificial Intelligence (AI) has significantly influenced various sectors, including healthcare, education, and finance. It helps in diagnostics, personalised learning, and algorithmic trading, driving efficiency and innovation. However, ethical concerns like data privacy and bias remain challenging issues."

st.title("Self-Watermarking Experiment")

model_choice = st.selectbox(
    "Select the model to verify against:",
    list(all_models.keys())
)

authentic_model = all_models[model_choice]

if input_option == "Enter text manually":
    st.session_state.input_text = st.text_area("Enter the text to verify:", st.session_state.input_text)
else:
    st.session_state.input_text = st.text_area("Enter the text to verify:",
                                               value=st.session_state.get('generated_text', st.session_state.input_text),
                                               help="You can edit the generated text or enter new text here.")

if st.button("Run Verification", key="run_verification_button"):
    with st.spinner("Running verification..."):
        iterations = 5
        (prompt, authorship_iterations, probabilities, authorship_result, 
         model_names, authentic_metrics, contrasting_metrics, 
         verification_iterations, weighted_scores, weights) = verify_authorship(
            st.session_state.input_text, authentic_model, model_choice, all_models, iterations)

        # Display results in the results container
        st.markdown("### Verification Results")
        if include_human_detection and authorship_result == "Human":
            st.success(f"**Authorship Result:** {authorship_result}")
            st.info("The text appears to be written by a human")
        elif authorship_result == model_choice:
            st.success(f"**Authorship Result:** Authentic (Original Model: {model_choice})")
            st.info(f"The predicted original model that generated the text is **{model_choice}**")
        else:
            st.error(f"**Authorship Result:** {authorship_result}")
            st.info(f"The predicted original model that generated the text is **{authorship_result}**")

        # Model Probabilities
        st.markdown("### Model Probabilities")
        prob_df = pd.DataFrame({
            'Model': model_names,
            'Probability': probabilities
        })
        prob_styler = prob_df.style.format({'Probability': '{:.2%}'}).hide(axis='index')
        st.write(prob_styler.to_html(), unsafe_allow_html=True)

        # Regeneration Iterations
        st.markdown("### Regeneration Iterations")
        for iteration in authorship_iterations:
            st.markdown(f"**Iteration {iteration['iteration']}:**")
            st.markdown(f'<div class="wrapped-text">{iteration["text"]}</div>', unsafe_allow_html=True)

        # Verification Iterations
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### Verification Iterations")
        st.markdown("**Authentic Output:**")
        st.markdown(f'<div class="wrapped-text">{verification_iterations[0]["authentic_output"]}</div>', unsafe_allow_html=True)
        st.markdown("**Contrasting Outputs:**")
        for model_name, output in verification_iterations[0]['contrasting_outputs'].items():
            st.markdown(f"**{model_name}:**")
            st.markdown(f'<div class="wrapped-text">{output}</div>', unsafe_allow_html=True)
       
        # Raw Metric Scores
        st.markdown("### Raw Metric Scores")
        
        # Define metric names first
        metric_names = list(authentic_metrics.keys())
        
        # Create DataFrame for raw scores - handle both cases
        if include_human_detection:
            raw_scores_df = pd.DataFrame(
                [authentic_metrics] + contrasting_metrics,
                columns=metric_names,
                index=model_names[:-1]  # Exclude 'Human' from index
            )
        else:
            raw_scores_df = pd.DataFrame(
                [authentic_metrics] + contrasting_metrics,
                columns=metric_names,
                index=model_names  # Use all model names when human detection is disabled
            )
        
        # Display the DataFrame with formatting
        st.dataframe(
            raw_scores_df.style
            .format("{:.4f}")
            .background_gradient(cmap="YlGnBu")
        )

        # Metric Contributions
        st.markdown("### Metric Contributions to Final Probability")
        
        # Create DataFrame with proper index - handle both cases
        if include_human_detection:
            contribution_df = pd.DataFrame(
                weighted_scores,
                columns=metric_names,
                index=model_names  # Include all models including Human
            )
        else:
            contribution_df = pd.DataFrame(
                weighted_scores,
                columns=metric_names,
                index=model_names  # Use all AI model names
            )
        
        # Display the DataFrame
        st.dataframe(
            contribution_df.style
            .format("{:.4f}")
            .background_gradient(cmap="YlGnBu")
        )

        # Stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        contribution_df.plot(kind='bar', stacked=True, ax=ax)
        plt.title("Metric Contributions to Final Probability")
        plt.xlabel("Models")
        plt.ylabel("Weighted Score")
        plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Only show Human Writing Analysis if human detection is enabled
        if include_human_detection:
            st.markdown("### Human Writing Analysis")
            human_features = extract_human_features(st.session_state.input_text)
            
            # Create radar chart for human features
            features_df = pd.DataFrame({
                'Feature': list(human_features.keys()),
                'Score': list(human_features.values())
            })
            
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='polar')
            
            angles = np.linspace(0, 2*np.pi, len(features_df), endpoint=False)
            values = features_df['Score'].values
            
            ax.plot(angles, values)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles)
            ax.set_xticklabels(features_df['Feature'], rotation=45)
            
            plt.title("Human Writing Features Analysis")
            st.pyplot(fig)
            
            # Show human probability
            human_prob = probabilities[-1]
            st.markdown(f"**Probability of Human Authorship:** {human_prob:.2%}")


# Add a logging statement at the beginning of your main script
logger.info("Starting Streamlit app...")

# Add this at the end of your script
logger.info("Streamlit app initialization complete")

# Add this near the top of your file, after the imports
st.markdown("""
<style>
.wrapped-text {
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: monospace;
    background-color: #f0f0f0;
    font-size: 12px;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 20px;  /* Add space below each text block */
}
.section-gap {
    margin-top: 40px;  /* Add more space between major sections */
}
</style>
""", unsafe_allow_html=True)











