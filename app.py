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

# Silence warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK data
import subprocess
import sys

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
            ])
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
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_text_claude_simple(prompt):
    client = load_anthropic_client()
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=100,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

def generate_text_gemini_simple(prompt):
    model = load_gemini_client()
    response = model.generate_content(prompt)
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
        messages=[
            {"role": "user", "content": f"You are a professional language facilitator. You should paraphrase the following sentence and output the final result only: {prompt} Remember to only output the final result"}
        ]
    )
    return response.choices[0].message.content.strip()

def generate_text_claude(prompt):
    client = load_anthropic_client()
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=100,
        temperature=0.7,
        messages=[
            {"role": "user", "content": f"You are a professional language facilitator. You should paraphrase the following sentence and output the final result only: {prompt} Remember to only output the final result"}
        ]
    )
    return response.content[0].text.strip()

def generate_text_gemini(prompt):
    model = load_gemini_client()
    response = model.generate_content(f"You are a professional language facilitator. You should paraphrase the following sentence and output the final result only: {prompt} Remember to only output the final result")
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
    for i in range(iterations):
        current_text = model_func(current_text)
        st.write(f"{model_name} - Iteration {i+1}: {current_text}")
        time.sleep(0.1)  # To avoid hitting rate limits
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

def normalize_scores(scores, power=2):
    normalized = np.array(scores, dtype=float)
    # Invert perplexity so that lower is better
    normalized[-1] = 1 / (1 + normalized[-1])
    
    # Apply power normalization
    normalized = normalized ** power
    
    return normalized

def calculate_authorship_probability(authentic_scores, contrasting_scores):
    # Adjust weights for BERTScore, Cosine Similarity, Inverse Perplexity
    weights = np.array([0.4, 0.4, 0.2])
    
    all_scores = np.array([authentic_scores] + contrasting_scores)
    
    # Normalize scores for each metric
    normalized_scores = np.apply_along_axis(normalize_scores, 0, all_scores)
    
    # Apply weights
    weighted_scores = normalized_scores * weights
    
    # Sum weighted scores for each model
    final_scores = weighted_scores.sum(axis=1)
    
    # Apply softmax with temperature to get probabilities
    temperature = 0.1  # Adjust this value to control the "sharpness" of the distribution
    exp_scores = np.exp((final_scores - np.max(final_scores)) / temperature)  # Subtract max for numerical stability
    probabilities = exp_scores / np.sum(exp_scores)
    
    # Handle any remaining NaN values
    probabilities = np.nan_to_num(probabilities, nan=0.0)
    
    # If all probabilities are zero, distribute evenly
    if np.sum(probabilities) == 0:
        probabilities = np.ones_like(probabilities) / len(probabilities)
    
    return probabilities

def determine_authorship(probabilities, model_names, threshold=0.35):
    max_prob = np.max(probabilities)
    max_index = np.argmax(probabilities)
    if max_prob >= threshold and max_index == 0:
        return "Authentic"
    elif max_prob >= threshold:
        return f"Contrasting ({model_names[max_index]})"
    else:
        return "Inconclusive"

def verify_authorship(text, authentic_model, authentic_name, all_models, iterations):
    authentic_regen = iterative_regeneration(text, authentic_model, authentic_name, iterations=5)
    results = {}
    contrasting_scores = []
    
    authentic_bertscore = calculate_bertscore(text, authentic_regen)
    authentic_cosine = calculate_cosine_similarity(text, authentic_regen)
    authentic_perplexity = calculate_perplexity(authentic_regen)
    
    authentic_scores = [authentic_bertscore, authentic_cosine, authentic_perplexity]
    
    results[authentic_name] = {
        'bertscore': authentic_bertscore,
        'cosine': authentic_cosine,
        'perplexity': authentic_perplexity
    }
    
    model_names = [authentic_name]
    for model_name, model_func in all_models.items():
        if model_name != authentic_name:
            contrasting_regen = iterative_regeneration(text, model_func, model_name, iterations=1)  # Always use 1 iteration for contrasting models
            if "Error using Ollama" in contrasting_regen or "Ollama (LLaMA) is not available" in contrasting_regen:
                st.warning(f"Skipping {model_name} due to unavailability.")
                continue
            bertscore = calculate_bertscore(text, contrasting_regen)
            cosine = calculate_cosine_similarity(text, contrasting_regen)
            perplexity = calculate_perplexity(contrasting_regen)
            results[model_name] = {
                'bertscore': bertscore,
                'cosine': cosine,
                'perplexity': perplexity
            }
            contrasting_scores.append([bertscore, cosine, perplexity])
            model_names.append(model_name)
    
    probabilities = calculate_authorship_probability(authentic_scores, contrasting_scores)
    authorship_result = determine_authorship(probabilities, model_names)
    
    return authentic_regen, results, probabilities, authorship_result, model_names

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
    prompt = st.text_area("Enter your prompt for text generation")
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

iterations_choice = st.radio(
    "Choose iteration mode for contrasting models:",
    ("One-shot", "5 iterations")
)

if st.button("Run Verification"):
    with st.spinner("Running verification..."):
        iterations = 5 if iterations_choice == "5 iterations" else 1
        authentic_regen, results, probabilities, authorship_result, model_names = verify_authorship(st.session_state.input_text, authentic_model, model_choice, all_models, iterations)
        
        st.markdown("## Authorship Probabilities")
        for i, model_name in enumerate(model_names):
            if i < len(probabilities):
                st.markdown(f"**{model_name}**: {probabilities[i]*100:.2f}%")
            else:
                st.markdown(f"**{model_name}**: Probability not calculated")
        
        st.markdown(f"## Final Result: **{authorship_result}**")
        
        st.markdown("## Detailed Metrics")
        for model_name, scores in results.items():
            st.markdown(f"### {model_name} ({'5 iterations' if model_name == model_choice else iterations_choice})")
            st.markdown(f"- **BERTScore**: {scores['bertscore']}")
            st.markdown(f"- **Cosine Similarity**: {scores['cosine']:.4f}")
            st.markdown(f"- **Perplexity**: {scores['perplexity']:.4f} (lower is better)")