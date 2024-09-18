import streamlit as st
from openai import OpenAI
import anthropic
import time

from bert_score import score
import torch
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import google.generativeai as genai
import requests
import ollama
import numpy as np
import math
from collections import Counter
import os
import pandas as pd

# Silence warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK data
import subprocess
import sys

st.set_page_config(page_title="Text Verification", layout="wide")

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import nltk
except ImportError:
    install('nltk')
    import nltk

nltk.download('punkt', quiet=True)

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def count_ngrams(tokens, n):
    return Counter(ngrams(tokens, n))

def clip_count(candidate_count, reference_count):
    return {ngram: min(count, reference_count[ngram]) for ngram, count in candidate_count.items()}

def modified_precision(reference_tokens, candidate_tokens, n):
    reference_count = count_ngrams(reference_tokens, n)
    candidate_count = count_ngrams(candidate_tokens, n)
    clipped_count = clip_count(candidate_count, reference_count)
    
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
    
    s = math.exp(sum(w * math.log(p) if p > 0 else float('-inf') for w, p in zip(weights, p_n)))
    
    return bp * s

st.sidebar.header("Additional API Keys (Optional)")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")

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
    genai.configure(api_key=gemini_api_key or st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel('gemini-pro')

def is_ollama_available():
    # Check if we're running in a Streamlit cloud environment
    if os.environ.get('STREAMLIT_SHARING') or os.environ.get('STREAMLIT_CLOUD'):
        return False
    
    try:
        import ollama
        return True
    except ImportError:
        return False

def generate_text_ollama_simple(prompt):
    if is_ollama_available():
        import ollama
        try:
            response = ollama.chat(model='llama3', messages=[
                {'role': 'user', 'content': prompt},
            ], options={"num_predict": 250})
            return response['message']['content']
        except Exception as e:
            return f"Error using Ollama: {str(e)}"
    else:
        return "Ollama (LLaMA) is not available in this environment."

def generate_text_ollama(prompt):
    if is_ollama_available():
        import ollama
        try:
            response = ollama.chat(model='llama3', messages=[
                {'role': 'user', 'content': f"You are a professional language facilitator. You should paraphrase the following sentence and output the final result only: {prompt}"},
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error using Ollama: {str(e)}"
    else:
        return "Ollama (LLaMA) is not available in this environment."

def generate_text_openai_simple(prompt):
    client = load_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250
    )
    return response.choices[0].message.content.strip()

def generate_text_claude_simple(prompt):
    client = load_anthropic_client()
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=250,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

def generate_text_gemini_simple(prompt):
    model = load_gemini_client()
    response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
        max_output_tokens=250
    ))
    return response.text.strip()

def generate_text(model, prompt):
    if model == "OpenAI":
        return generate_text_openai_simple(prompt)
    elif model == "Claude":
        return generate_text_claude_simple(prompt)
    elif model == "Gemini":
        return generate_text_gemini_simple(prompt)
    elif model == "Ollama (LLaMA)":
        return generate_text_ollama_simple(prompt)

def generate_text_openai(prompt):
    client = load_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=250,
        messages=[
            {"role": "user", "content": f"You are a professional language facilitator. You should paraphrase the following sentence and output the final result only: {prompt} Remember to only output the final result"}
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
            {"role": "user", "content": f"You are a professional language facilitator. You should paraphrase the following sentence and output the final result only: {prompt} Remember to only output the final result"}
        ]
    )
    return response.content[0].text.strip()

def generate_text_gemini(prompt):
    model = load_gemini_client()
    #response = model.generate_content(f"You are a professional language facilitator. You should paraphrase the following sentence and output the final result only: {prompt} Remember to only output the final result")
    response = model.generate_content(
    f"You are a professional language facilitator. You should paraphrase the following sentence and output the final result only: {prompt} Remember to only output the final result",
    generation_config=genai.types.GenerationConfig(
        temperature=0.7,
        max_output_tokens=250,
    ),
    )
    return response.text.strip()


# Update the all_models dictionary
all_models = {
    "OpenAI": generate_text_openai,
    "Claude": generate_text_claude,
    "Gemini": generate_text_gemini,
}
if is_ollama_available():
    all_models["Ollama (LLaMA)"] = generate_text_ollama

# Update the model selection dropdown
available_models = ["OpenAI", "Claude", "Gemini"]
if is_ollama_available():
    available_models.append("Ollama (LLaMA)")

def iterative_regeneration(initial_text, model_func, model_name, iterations=1):
    current_text = initial_text
    progress_bar = st.progress(0)
    status_area = st.empty()
    
    for i in range(iterations):
        current_text = model_func(current_text)
        progress = (i + 1) / iterations
        progress_bar.progress(progress)
        status_area.markdown(f"**{model_name} - Iteration {i+1}:** {current_text}")
        time.sleep(0.1)  # To avoid hitting rate limits
    
    time.sleep(0.5)  # Give a moment to see the final state
    progress_bar.empty()
    status_area.empty()
    return current_text

def calculate_bertscore(reference, candidate):
    P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
    return F1.item()  # Return the F1 score

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]

def calculate_perplexity(text):
    model, tokenizer = load_gpt2()
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
    return torch.exp(outputs.loss).item()

import numpy as np

def normalize_scores(scores):
    normalized = np.array(scores, dtype=float)
    # Invert perplexity so that lower is better
    normalized[-1] = 1 / (1 + normalized[-1])
    
    # Min-max normalization for each metric
    for i in range(len(normalized)):
        min_val = np.min(normalized[:, i])
        max_val = np.max(normalized[:, i])
        if max_val > min_val:
            normalized[:, i] = (normalized[:, i] - min_val) / (max_val - min_val)
        else:
            normalized[:, i] = 1  # If all values are the same, set to 1
    
    return normalized

def calculate_authorship_probability(authentic_scores, contrasting_scores):
    # Adjust weights for BERTScore, Cosine Similarity, Inverse Perplexity
    weights = np.array([0.4, 0.4, 0.2])
    
    all_scores = np.array([authentic_scores] + contrasting_scores)
    
    # Normalize scores for each metric
    normalized_scores = normalize_scores(all_scores)
    
    # Apply weights
    weighted_scores = normalized_scores * weights
    
    # Sum weighted scores for each model
    final_scores = weighted_scores.sum(axis=1)
    
    # Convert to probabilities using softmax with a higher temperature
    temperature = 0.5  # Adjust this value to control the "sharpness" of the distribution
    exp_scores = np.exp(final_scores / temperature)
    probabilities = exp_scores / np.sum(exp_scores)
    
    return probabilities

def determine_authorship(probabilities, model_names, threshold=0.4):
    max_prob = np.max(probabilities)
    max_index = np.argmax(probabilities)
    if max_prob >= threshold and max_index == 0:
        return "Authentic"
    elif max_prob >= threshold:
        return f"Contrasting ({model_names[max_index]})"
    else:
        return "Inconclusive"

def verify_authorship(text, authentic_model, authentic_name, all_models, iterations):
    iteration_container = st.empty()
    results_container = st.empty()
    
    with iteration_container.container():
        st.markdown(f"### Iterations for {authentic_name}")
        authentic_regen = iterative_regeneration(text, authentic_model, authentic_name, iterations=iterations)
    
    results = {}
    contrasting_scores = []
    
    authentic_bertscore = calculate_bertscore(text, authentic_regen)
    authentic_cosine = calculate_cosine_similarity(text, authentic_regen)
    authentic_perplexity = calculate_perplexity(authentic_regen)
    
    authentic_scores = [authentic_bertscore, authentic_cosine, authentic_perplexity]
    
    results[authentic_name] = {
        'BERTScore': authentic_bertscore,
        'Cosine Similarity': authentic_cosine,
        'Perplexity': authentic_perplexity
    }
    
    model_names = [authentic_name]
    for model_name, model_func in all_models.items():
        if model_name != authentic_name:
            with iteration_container.container():
                st.markdown(f"### Iterations for {model_name}")
                contrasting_regen = iterative_regeneration(text, model_func, model_name, iterations=1)
            if "Error using Ollama" in contrasting_regen or "Ollama (LLaMA) is not available" in contrasting_regen:
                st.warning(f"Skipping {model_name} due to unavailability.")
                continue
            bertscore = calculate_bertscore(text, contrasting_regen)
            cosine = calculate_cosine_similarity(text, contrasting_regen)
            perplexity = calculate_perplexity(contrasting_regen)
            results[model_name] = {
                'BERTScore': bertscore,
                'Cosine Similarity': cosine,
                'Perplexity': perplexity
            }
            contrasting_scores.append([bertscore, cosine, perplexity])
            model_names.append(model_name)
    
    probabilities = calculate_authorship_probability(authentic_scores, contrasting_scores)
    authorship_result = determine_authorship(probabilities, model_names)
    
    return authentic_regen, results, probabilities, authorship_result, model_names, results_container, iteration_container



# Custom CSS to make tables consistent and improve appearance
st.markdown("""
<style>
    .stTable, .dataframe {
        width: 100%;
        max-width: 100%;
    }
    .stTable td, .stTable th, .dataframe td, .dataframe th {
        text-align: left;
        padding: 8px;
    }
    .stTable tr:nth-child(even), .dataframe tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    .stTable th, .dataframe th {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

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
    
    # Add example prompts
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
            generated_text = generate_text(generation_model, prompt)
            st.write("Generated Text:")
            st.write(generated_text)
            st.session_state.generated_text = generated_text
        except Exception as e:
            st.error(f"Error generating text: {str(e)}")

if 'input_text' not in st.session_state:
    st.session_state.input_text = "The quick brown fox jumps over the lazy dog."

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

if st.button("Run Verification"):
    with st.spinner("Running verification..."):
        iterations = 5
        authentic_regen, results, probabilities, authorship_result, model_names, results_container, iteration_container = verify_authorship(st.session_state.input_text, authentic_model, model_choice, all_models, iterations)
        
        # Clear the iteration container
        iteration_container.empty()
        
        # Store all generations
        all_generations = {model_choice: authentic_regen}
        for model_name, model_func in all_models.items():
            if model_name != model_choice:
                all_generations[model_name] = iterative_regeneration(st.session_state.input_text, model_func, model_name, iterations=1)
        
        # Display results in the results container
        with results_container.container():
            st.markdown("### Final Iteration for Authentic Model")
            st.markdown(authentic_regen)
            
            st.markdown("### Verification Results")
            if authorship_result == "Authentic":
                st.markdown(f"**Authorship Result:** {authorship_result} ({model_choice})")
                st.markdown(f"The predicted original model that generated the text is {model_choice}")
            else:
                predicted_model = model_names[np.argmax(probabilities)]
                st.markdown(f"**Authorship Result:** ({predicted_model})")
                st.markdown(f"The predicted original model that generated the text is {predicted_model}")
            

            
            st.markdown("### Model Probabilities")
            prob_df = pd.DataFrame({'Model': model_names, 'Probability': probabilities})
            prob_styler = prob_df.style.format({'Probability': '{:.2%}'}).hide(axis='index')
            st.write(prob_styler.to_html(), unsafe_allow_html=True)
            
            st.markdown("### Detailed Metrics")
            metrics_df = pd.DataFrame(results).T
            metrics_styler = metrics_df.style.format({
                'BERTScore': '{:.4f}',
                'Cosine Similarity': '{:.4f}',
                'Perplexity': '{:.4f}'
            })
            st.write(metrics_styler.to_html(), unsafe_allow_html=True)
            
            # Add dropdown to view all generations
            st.markdown("### View All Generations")
            selected_model = st.selectbox("Select a model to view its generation:", list(all_generations.keys()))
            st.markdown(f"**Generation by {selected_model}:**")
            st.markdown(all_generations[selected_model])
