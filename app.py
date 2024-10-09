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

def calculate_bleu(reference, candidate, weights=[0.25, 0.25, 0.25, 0.25]):
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

# Load models and clients
@st.cache_resource
def load_openai_client():
    return OpenAI(api_key=openai_api_key or st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_anthropic_client():
    return anthropic.Anthropic(api_key=anthropic_api_key or st.secrets["ANTHROPIC_API_KEY"])

@st.cache_resource
def load_gpt2():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return model, tokenizer

@st.cache_resource
def load_gemini_client():
    try:
        genai.configure(api_key=gemini_api_key or st.secrets["GEMINI_API_KEY"])
        return genai.GenerativeModel('gemini-1.5-pro')
    except Exception as e:
        st.error(f"Failed to load Gemini client: {e}")
        return None

# Text generation functions for different models
def generate_text_openai(prompt):
    client = load_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=250,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def generate_text_claude(prompt):
    client = load_anthropic_client()
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=250,
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
                max_output_tokens=250,
                temperature=0.7
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"Failed to generate text using Gemini: {e}"

# Mapping of models to their generation functions
all_models = {
    "OpenAI": generate_text_openai,
    "Claude": generate_text_claude,
    "Gemini": generate_text_gemini,
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
        prompt = f"Please summarise the following text into one sentence:\n\n{current_text}"
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
    # Re-generation by the authentic model
    prompt_authentic = f"Please summarise the following text into one sentence:\n\n{final_output}"
    y_a = authentic_model_func(prompt_authentic)
    # Re-generation by contrasting models
    y_cs = []
    for model_func in contrasting_model_funcs:
        prompt_contrasting = f"Please summarise the following text into one sentence:\n\n{final_output}"
        y_c = model_func(prompt_contrasting)
        y_cs.append(y_c)
    return y_a, y_cs

# Scoring functions
def calculate_metrics(reference, candidates):
    metrics = []
    for candidate in candidates:
        bertscore = calculate_bertscore(reference, candidate)
        cosine = calculate_cosine_similarity(reference, candidate)
        perplexity = calculate_perplexity(candidate)
        rouge = calculate_rouge(reference, candidate)
        meteor = calculate_meteor(reference, candidate)
        bleu = calculate_bleu(reference, candidate)
        metrics.append({
            'BERTScore': bertscore,
            'Cosine Similarity': cosine,
            'Perplexity': perplexity,
            'ROUGE-L': rouge,
            'METEOR': meteor,
            'BLEU': bleu
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

# Verification function
def verify_authorship(text, authentic_model, authentic_name, all_models, iterations):
    iteration_container = st.empty()
    results_container = st.empty()

    with iteration_container.container():
        st.markdown(f"### Iterations for {authentic_name}")
        authentic_regens = iterative_regeneration(text, authentic_model, authentic_name, iterations=iterations)

    final_output = authentic_regens[-1]
    contrasting_models = {name: func for name, func in all_models.items() if name != authentic_name}
    y_a, y_cs = verification_step(final_output, authentic_model, list(contrasting_models.values()), list(contrasting_models.keys()))

    # Calculating metrics
    authentic_metrics = calculate_metrics(final_output, [y_a])[0]
    contrasting_metrics = calculate_metrics(final_output, y_cs)
    results = {authentic_name: authentic_metrics}
    for name, metrics in zip(contrasting_models.keys(), contrasting_metrics):
        results[name] = metrics

    # Calculate authorship probability
    probabilities = calculate_authorship_probability(authentic_metrics, contrasting_metrics)
    model_names = [authentic_name] + list(contrasting_models.keys())
    authorship_result = determine_authorship(probabilities, model_names)

    regenerations = {authentic_name: authentic_regens}

    # Inside the verify_authorship function, after calculating the metrics:

    st.markdown("### Model Outputs")
    for model_name, model_func in all_models.items():
        output = model_func(f"Please summarise the following text into one sentence:\n\n{final_output}")
        st.subheader(f"{model_name} Output:")
        st.write(output)

    st.markdown("### Detailed Metrics")
    st.json(results)  # This will display the raw metrics dictionary

    # Keep the existing metrics display code
    metrics_df = pd.DataFrame(results).T
    metrics_styler = metrics_df.style.format({
        'BERTScore': '{:.4f}',
        'Cosine Similarity': '{:.4f}',
        'Perplexity': '{:.4f}',
        'ROUGE-L': '{:.4f}',
        'METEOR': '{:.4f}',
        'BLEU': '{:.4f}'
    })
    st.write(metrics_styler.to_html(), unsafe_allow_html=True)

    return authentic_regens, results, probabilities, authorship_result, model_names, results_container, iteration_container, regenerations

def calculate_authorship_probability(authentic_metrics, contrasting_metrics):
    # Adjust weights for metrics
    weights = np.array([0.3, 0.25, 0.15, 0.15, 0.15, 0.1])  # Added BLEU
    all_metrics = [authentic_metrics] + contrasting_metrics
    scores = []
    for metric in all_metrics:
        score_values = list(metric.values())
        # Invert perplexity
        score_values[2] = 1 / (1 + score_values[2])
        scores.append(score_values)
    normalized_scores = normalize_scores(np.array(scores))
    weighted_scores = normalized_scores * weights
    final_scores = weighted_scores.sum(axis=1)
    probabilities = np.exp(final_scores) / np.sum(np.exp(final_scores))
    return probabilities

def normalize_scores(scores):
    normalized = np.array(scores, dtype=float)
    # Min-max normalization
    for i in range(normalized.shape[1]):
        min_val = np.min(normalized[:, i])
        max_val = np.max(normalized[:, i])
        if max_val > min_val:
            normalized[:, i] = (normalized[:, i] - min_val) / (max_val - min_val)
        else:
            normalized[:, i] = 1
    return normalized

def determine_authorship(probabilities, model_names, threshold=0.4):
    max_prob = np.max(probabilities)
    max_index = np.argmax(probabilities)
    if max_prob >= threshold and max_index == 0:
        return "Authentic"
    elif max_prob >= threshold:
        return f"Contrasting ({model_names[max_index]})"
    else:
        return "Inconclusive"

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
        authentic_regens, results, probabilities, authorship_result, model_names, results_container, iteration_container, regenerations = verify_authorship(
            st.session_state.input_text, authentic_model, model_choice, all_models, iterations)

        # Clear the iteration container
        iteration_container.empty()

        # Display results in the results container
        with results_container.container():
            st.markdown("### Verification Results")
            if authorship_result == "Authentic":
                st.markdown(f"**Authorship Result:** {authorship_result} (Original Model: {model_choice})")
                st.markdown(f"The predicted original model that generated the text is {model_choice}")
            else:
                predicted_model = model_names[np.argmax(probabilities)]
                st.markdown(f"**Authorship Result:** {authorship_result}")
                st.markdown(f"The predicted original model that generated the text is {predicted_model}")

            st.markdown("### Model Probabilities")
            prob_df = pd.DataFrame({'Model': model_names, 'Probability': probabilities})
            prob_styler = prob_df.style.format({'Probability': '{:.2%}'}).hide(axis='index')
            st.write(prob_styler.to_html(), unsafe_allow_html=True)

            st.markdown("### Regeneration Iterations")
            for model_name, regens in regenerations.items():
                with st.expander(f"Iterations for {model_name}"):
                    if isinstance(regens, list):
                        for i, iteration in enumerate(regens, 1):
                            st.markdown(f"**Iteration {i}:**")
                            st.text(iteration)
                    else:
                        st.text(regens)