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
import ollama
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
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

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
    try:
        genai.configure(api_key=gemini_api_key or st.secrets["GEMINI_API_KEY"])
        return genai.GenerativeModel('gemini-1.5-pro')
    except Exception as e:
        st.error(f"Failed to load Gemini client: {e}")
        return None

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
                {'role': 'user', 'content': f"You are a professional language facilitator. You should paraphrase the following and output the final result only: {prompt}"},
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
        max_output_tokens=250,
        temperature=0.7,
    ))
    return response.text.strip()

def generate_text(model, prompt):
    if model == "OpenAI":
        return generate_text_openai_simple(prompt)
    elif model == "Claude":
        return generate_text_claude_simple(prompt)
    elif model == "Gemini":
        gemini_response = generate_text_gemini(prompt)
        if gemini_response.startswith("Failed"):
            return "Gemini model is currently unavailable."
        return gemini_response
    elif model == "Ollama (LLaMA)":
        return generate_text_ollama_simple(prompt)

def generate_text_openai(prompt):
    client = load_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=250,
        temperature=0.7,
        messages=[
            {"role": "user", "content": f"You are a professional language facilitator. You should paraphrase the following and output the final result only: {prompt} Remember to only output the final result"}
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
            {"role": "user", "content": f"You are a professional language facilitator. You should paraphrase the following and output the final result only: {prompt} Remember to only output the final result"}
        ]
    )
    return response.content[0].text.strip()

def generate_text_gemini(prompt, retries=3, delay=2):
    model = load_gemini_client()
    if model is None:
        return "Gemini model client is not available."
    
    for attempt in range(retries):
        try:
            response = model.generate_content(
                f"You are a professional language facilitator. You should paraphrase the following and output the final result only: {prompt} Remember to only output the final result",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=250,
                ),
            )
            return response.text.strip()
        except InternalServerError as e:
            st.error(f"Internal server error with Gemini API: {e}. Retrying ({attempt + 1}/{retries})...")
            time.sleep(delay)
        except Exception as e:
            st.error(f"An unexpected error occurred with Gemini API: {e}")
            break
    return "Failed to generate text using Gemini after multiple attempts."

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
    regenerations = []
    progress_bar = st.progress(0)
    status_area = st.empty()
    
    for i in range(iterations):
        current_text = model_func(current_text)
        regenerations.append(current_text)
        progress = (i + 1) / iterations
        progress_bar.progress(progress)
        status_area.markdown(f"**{model_name} - Iteration {i+1}:** {current_text}")
        time.sleep(0.1)  # To avoid hitting rate limits
    
    time.sleep(0.5)  # Give a moment to see the final state
    progress_bar.empty()
    status_area.empty()
    return regenerations

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

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rougeL'].fmeasure

def calculate_meteor(reference, candidate):
    reference_tokens = word_tokenize(reference)
    candidate_tokens = word_tokenize(candidate)
    return meteor_score([reference_tokens], candidate_tokens)

import numpy as np

def normalize_scores(scores):
    normalized = np.array(scores, dtype=float)
    # Invert perplexity so that lower is better
    normalized[:, 2] = 1 / (1 + normalized[:, 2])
    
    # Min-max normalization for each metric
    for i in range(normalized.shape[1]):
        min_val = np.min(normalized[:, i])
        max_val = np.max(normalized[:, i])
        if max_val > min_val:
            normalized[:, i] = (normalized[:, i] - min_val) / (max_val - min_val)
        else:
            normalized[:, i] = 1  # If all values are the same, set to 1
    
    return normalized

def calculate_authorship_probability(authentic_scores, contrasting_scores):
    # Adjust weights for BERTScore, Cosine Similarity, Inverse Perplexity, ROUGE-L, METEOR
    weights = np.array([0.3, 0.25, 0.15, 0.15, 0.15])
    
    all_scores = np.array([authentic_scores] + contrasting_scores)
    
    # Invert perplexity so that lower is better
    all_scores[:, 2] = 1 / (1 + all_scores[:, 2])
    
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
        authentic_regens = iterative_regeneration(text, authentic_model, authentic_name, iterations=iterations)
    
    results = {}
    contrasting_scores = []
    regenerations = {authentic_name: authentic_regens}
    
    # Use the last regeneration for calculations
    authentic_regen = authentic_regens[-1]
    
    authentic_bertscore = calculate_bertscore(text, authentic_regen)
    authentic_cosine = calculate_cosine_similarity(text, authentic_regen)
    authentic_perplexity = calculate_perplexity(authentic_regen)
    authentic_rouge = calculate_rouge(text, authentic_regen)
    authentic_meteor = calculate_meteor(text, authentic_regen)
    
    authentic_scores = [authentic_bertscore, authentic_cosine, authentic_perplexity, authentic_rouge, authentic_meteor]
    
    results[authentic_name] = {
        'BERTScore': authentic_bertscore,
        'Cosine Similarity': authentic_cosine,
        'Perplexity': authentic_perplexity,
        'Inverse Perplexity': 1 / (1 + authentic_perplexity),
        'ROUGE-L': authentic_rouge,
        'METEOR': authentic_meteor
    }
    
    model_names = [authentic_name]
    for model_name, model_func in all_models.items():
        if model_name != authentic_name:
            with iteration_container.container():
                st.markdown(f"### Iterations for {model_name}")
                contrasting_regen = iterative_regeneration(text, model_func, model_name, iterations=1)
            regenerations[model_name] = contrasting_regen
            if "Error using Ollama" in contrasting_regen[0] or "Ollama (LLaMA) is not available" in contrasting_regen[0]:
                st.warning(f"Skipping {model_name} due to unavailability.")
                continue
            bertscore = calculate_bertscore(text, contrasting_regen[0])
            cosine = calculate_cosine_similarity(text, contrasting_regen[0])
            perplexity = calculate_perplexity(contrasting_regen[0])
            rouge = calculate_rouge(text, contrasting_regen[0])
            meteor = calculate_meteor(text, contrasting_regen[0])
            results[model_name] = {
                'BERTScore': bertscore,
                'Cosine Similarity': cosine,
                'Perplexity': perplexity,
                'Inverse Perplexity': 1 / (1 + perplexity),
                'ROUGE-L': rouge,
                'METEOR': meteor
            }
            contrasting_scores.append([bertscore, cosine, perplexity, rouge, meteor])
            model_names.append(model_name)
    
    probabilities = calculate_authorship_probability(authentic_scores, contrasting_scores)
    authorship_result = determine_authorship(probabilities, model_names)
    
    return authentic_regens, results, probabilities, authorship_result, model_names, results_container, iteration_container, regenerations

# New function to test different metric combinations
def test_metric_combinations(test_data, all_models):
    metric_combinations = [
        ([1, 0, 0], 'BERTScore only'),
        ([0, 1, 0], 'Cosine Similarity only'),
        ([0, 0, 1], 'Inverse Perplexity only'),
        ([0.5, 0.5, 0], 'BERTScore + Cosine'),
        ([0.5, 0, 0.5], 'BERTScore + Inverse Perplexity'),
        ([0, 0.5, 0.5], 'Cosine + Inverse Perplexity'),
        ([0.33, 0.33, 0.34], 'Equal weights'),
        ([0.4, 0.4, 0.2], 'Current weights'),
        ([0.6, 0.3, 0.1], 'High BERTScore weight'),
        ([0.3, 0.6, 0.1], 'High Cosine weight'),
        ([0.3, 0.3, 0.4], 'High Inverse Perplexity weight'),
    ]

    results = []

    for weights, description in metric_combinations:
        correct_predictions = 0
        total_predictions = 0

        for sample in test_data:
            text = sample["text"]
            true_author = sample["true_author"]

            authentic_scores = []
            contrasting_scores = []

            for model_name, model_func in all_models.items():
                regenerated_text = model_func(text)
                bertscore = calculate_bertscore(text, regenerated_text)
                cosine = calculate_cosine_similarity(text, regenerated_text)
                perplexity = calculate_perplexity(regenerated_text)
                scores = [bertscore, cosine, 1 / (1 + perplexity)]  # Inverse perplexity

                if model_name == true_author:
                    authentic_scores = scores
                else:
                    contrasting_scores.append(scores)

            probabilities = calculate_authorship_probability(authentic_scores, contrasting_scores)
            predicted_author = list(all_models.keys())[np.argmax(probabilities)]

            if predicted_author == true_author:
                correct_predictions += 1
            total_predictions += 1

        accuracy = correct_predictions / total_predictions
        results.append({
            'weights': weights,
            'description': description,
            'accuracy': accuracy
        })

    return results


def generate_test_data(models, num_samples_per_model=10):
    test_data = []
    prompts = [
        "Explain a complex scientific concept in simple terms.",
        "Write a short story about an unexpected adventure.",
        "Describe the process of making a traditional dish from your culture.",
        "Discuss the potential impacts of a new technology on society.",
        "Create a poem about the changing seasons."
    ]
    for model_name, model_func in models.items():
        for i in range(num_samples_per_model):
            prompt = prompts[i % len(prompts)]
            text = model_func(prompt)
            test_data.append({"text": text, "true_author": model_name})
    return test_data

# Create tabs
tab1, tab2 = st.tabs(["Verification", "Automated Testing"])

with tab1:
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

    # If there's a "Run Verification" button in this tab, give it a unique key
    if st.button("Run Verification", key="run_verification_button"):
        with st.spinner("Running verification..."):
            iterations = 5
            authentic_regens, results, probabilities, authorship_result, model_names, results_container, iteration_container, regenerations = verify_authorship(st.session_state.input_text, authentic_model, model_choice, all_models, iterations)
            
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
                    st.markdown(f"**Authorship Result:** {authorship_result} ")
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
                    'Perplexity': '{:.4f}',
                    'Inverse Perplexity': '{:.4f}',
                    'ROUGE-L': '{:.4f}',
                    'METEOR': '{:.4f}'
                })
                st.write(metrics_styler.to_html(), unsafe_allow_html=True)
                
                st.markdown("### Regeneration Iterations")
                for model_name, regens in regenerations.items():
                    with st.expander(f"Iterations for {model_name}"):
                        if isinstance(regens, list):
                            for i, iteration in enumerate(regens, 1):
                                st.markdown(f"**Iteration {i}:**")
                                st.text(iteration)
                        else:
                            st.text(regens)


def analyze_results(results):
    df = pd.DataFrame(results)
    
    # Get unique authors from both true and predicted
    unique_authors = sorted(set(df["true_author"]) | set(df["predicted_author"]))
    
    # Overall accuracy
    accuracy = (df["true_author"] == df["predicted_author"]).mean()
    
    # Confusion matrix
    true_labels = df["true_author"]
    predicted_labels = df["predicted_author"]
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_authors)
    
    # Classification report
    report = classification_report(true_labels, predicted_labels, labels=unique_authors, output_dict=True)
    
    return accuracy, cm, report

with tab2:
    st.title("Automated Testing")
    # List of 25 example prompts
    example_prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a short story about a time traveler.",
        "Describe the process of photosynthesis.",
        "Discuss the impact of social media on modern society.",
        "Explain how a computer processor works.",
        "Write a poem about the changing seasons.",
        "Describe the water cycle and its importance.",
        "Discuss the pros and cons of renewable energy sources.",
        "Explain the concept of artificial intelligence.",
        "Write a brief history of the Internet.",
        "Describe the process of making chocolate from cocoa beans.",
        "Discuss the importance of biodiversity in ecosystems.",
        "Explain how vaccines work to prevent diseases.",
        "Write about the cultural significance of tea ceremonies.",
        "Describe the process of plate tectonics and its effects.",
        "Discuss the impact of space exploration on technology.",
        "Explain the basics of quantum computing.",
        "Write about the evolution of human language.",
        "Describe the process of how a bill becomes a law.",
        "Discuss the psychological effects of color in marketing.",
        "Explain the concept of blockchain technology.",
        "Write about the history and cultural impact of jazz music.",
        "Describe the process of wine making from grape to bottle.",
        "Discuss the ethical considerations of genetic engineering.",
        "Explain how neural networks in machine learning work."
    ]
    # Allow user to select 1 to 5 prompts
    num_prompts = st.slider("Select the number of prompts for testing:", min_value=1, max_value=5, value=3)
    
    selected_prompts = st.multiselect(
        f"Select {num_prompts} prompt{'s' if num_prompts > 1 else ''} for testing:",
        example_prompts,
        default=example_prompts[:num_prompts],
        max_selections=num_prompts
    )

    # Select the authentic model
    authentic_model = st.selectbox(
        "Select the model to be considered as authentic:",
        list(all_models.keys())
    )

    # Add a slider for the number of repetitions
    num_repetitions = st.slider("Select the number of repetitions for each prompt:", min_value=1, max_value=5, value=3)

    def run_automated_tests(prompts, all_models, authentic_model, num_repetitions):
        results = []
        total_tests = len(prompts) * num_repetitions
        progress_bar = st.progress(0)
        status_text = st.empty()
        test_counter = 0
        start_time = time.time()

        for prompt in prompts:
            for iteration in range(1, num_repetitions + 1):
                try:
                    # Generate text using the selected authentic model
                    if authentic_model == "OpenAI":
                        original_text = generate_text_openai_simple(prompt)
                    elif authentic_model == "Claude":
                        original_text = generate_text_claude_simple(prompt)
                    elif authentic_model == "Gemini":
                        original_text = generate_text_gemini_simple(prompt)
                    elif authentic_model == "Ollama (LLaMA)":
                        original_text = generate_text_ollama_simple(prompt)
                    else:
                        original_text = all_models[authentic_model](prompt)

                    # Verify authorship
                    with st.empty():
                        regenerations, metrics, probabilities, authorship_result, model_names, _, _ = verify_authorship(
                            original_text, all_models[authentic_model], authentic_model, all_models, iterations=5
                        )
                    
                    predicted_author = model_names[np.argmax(probabilities)]
                    result = {
                        "prompt": prompt,
                        "iteration": iteration,
                        "original_text": original_text,
                        "true_author": authentic_model,
                        "predicted_author": predicted_author,
                        "authorship_result": authorship_result,
                        "probabilities": dict(zip(model_names, probabilities)),
                        "metrics": metrics,
                        "regenerations": regenerations
                    }
                    results.append(result)
                    print(f"Test result: {result}")  # Print each result to the terminal
                except Exception as e:
                    error_msg = f"Error in test: {e}"
                    st.warning(error_msg)
                    print(error_msg)  # Print error to the terminal
                    results.append({
                        "prompt": prompt,
                        "iteration": iteration,
                        "original_text": "Error generating text",
                        "true_author": authentic_model,
                        "predicted_author": "Error",
                        "authorship_result": f"Error: {e}",
                        "probabilities": {},
                        "metrics": {},
                        "regenerations": {}
                    })
                
                # Update progress and info
                test_counter += 1
                progress_percentage = test_counter / total_tests
                progress_bar.progress(progress_percentage)
                
                elapsed_time = time.time() - start_time
                estimated_total_time = (elapsed_time / test_counter) * total_tests
                estimated_remaining_time = estimated_total_time - elapsed_time
                
                status_text.text(f"Progress: {progress_percentage:.1%} ({test_counter}/{total_tests} tests completed)\n"
                             f"Estimated time remaining: {estimated_remaining_time:.1f} seconds")

        progress_bar.empty()
        status_text.empty()
        return results

    # Give this button a unique key
    if st.button("Run Automated Tests", key="run_automated_tests_button"):
        if len(selected_prompts) != num_prompts:
            st.error(f"Please select exactly {num_prompts} prompt{'s' if num_prompts > 1 else ''} for testing.")
        else:
            # Check if Ollama (LLaMA) is available
            ollama_available = True
            try:
                all_models["Ollama (LLaMA)"]("Test prompt")
            except Exception:
                ollama_available = False
                st.warning("Skipping Ollama (LLaMA) due to unavailability")

            # Filter out Ollama if it's not available
            test_models = {k: v for k, v in all_models.items() if k != "Ollama (LLaMA)" or ollama_available}
            
            test_results = run_automated_tests(selected_prompts, test_models, authentic_model, num_repetitions)
            
            if test_results:
                accuracy, cm, report = analyze_results(test_results)
                
                st.markdown("### Automated Test Results")
                st.markdown(f"**Overall Accuracy:** {accuracy:.2%}")
                
                st.markdown("### Confusion Matrix")
                unique_authors = sorted(set(result['true_author'] for result in test_results) | 
                                        set(result['predicted_author'] for result in test_results))
                cm_df = pd.DataFrame(cm, index=unique_authors, columns=unique_authors)
                st.write(cm_df)
                
                st.markdown("### Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.write(report_df)
                
                #st.markdown("### Debug: Result Structure")
                #st.json(test_results[0])
                
                st.markdown("### Detailed Results")
                for result in test_results:
                    st.markdown(f"**Prompt:** {result['prompt']}")
                    st.markdown(f"**Iteration:** {result['iteration']}")
                    st.markdown(f"**Original Text:**")
                    st.text(result['original_text'])
                    st.markdown(f"**True Author:** {result['true_author']}")
                    st.markdown(f"**Predicted Author:** {result['predicted_author']}")
                    st.markdown(f"**Authorship Result:** {result['authorship_result']}")
                    st.markdown("**Probabilities:**")
                    st.write(pd.DataFrame([result['probabilities']]))
                    st.markdown("**Metrics:**")
                    st.write(pd.DataFrame(result['metrics']).T)
                    
                    st.markdown("**Regenerations:**")
                    if 'regenerations' in result:
                        if isinstance(result['regenerations'], dict):
                            for model, regens in result['regenerations'].items():
                                st.markdown(f"*{model}:*")
                                if isinstance(regens, list):
                                    for i, regen in enumerate(regens, 1):
                                        st.markdown(f"Regeneration {i}:")
                                        st.text(regen)
                                else:
                                    st.text(str(regens))
                        else:
                            st.text(str(result['regenerations']))
                    else:
                        st.text("No regenerations data available")
                    
                    st.markdown("---")
                
                # Create a downloadable CSV file
                results_df = pd.DataFrame(test_results)

                # Flatten the nested dictionaries (probabilities and metrics)
                for col in ['probabilities', 'metrics']:
                    if col in results_df.columns:
                        flattened = pd.json_normalize(results_df[col])
                        flattened.columns = [f"{col}_{subcol}" for subcol in flattened.columns]
                        results_df = pd.concat([results_df.drop(columns=[col]), flattened], axis=1)

                # Handle regenerations
                if 'regenerations' in results_df.columns:
                    def parse_regenerations(regen_str):
                        try:
                            return ast.literal_eval(regen_str)
                        except:
                            return {}

                    results_df['regenerations'] = results_df['regenerations'].apply(parse_regenerations)

                    for i in range(5):  # 5 regenerations for authentic model
                        col_name = f'authentic_regeneration_{i+1}'
                        results_df[col_name] = results_df['regenerations'].apply(
                            lambda x: x.get(authentic_model, [])[i] if isinstance(x.get(authentic_model), list) and i < len(x.get(authentic_model)) else ''
                        )
                    
                    for model in all_models.keys():
                        if model != authentic_model:
                            col_name = f'contrasting_regeneration_{model}'
                            results_df[col_name] = results_df['regenerations'].apply(
                                lambda x: x.get(model, [''])[0] if isinstance(x.get(model), list) and len(x.get(model)) > 0 else ''
                            )

                    #results_df = results_df.drop(columns=['regenerations'])

                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="automated_test_results.csv",
                    mime="text/csv",
                )
            else:
                error_msg = "No test results available. Please check the logs for errors."
                st.error(error_msg)
                print(error_msg)  # Print error to the terminal

    if st.button("Run Metric Combination Tests"):
        test_data = generate_test_data(all_models, num_samples_per_model=20)
        metric_results = test_metric_combinations(test_data, all_models)

        st.markdown("### Metric Combination Test Results")
        results_df = pd.DataFrame(metric_results)
        results_df = results_df.sort_values('accuracy', ascending=False)
        st.write(results_df)

        st.markdown("### Best Performing Metric Combination")
        best_combination = results_df.iloc[0]
        st.write(f"Description: {best_combination['description']}")
        st.write(f"Weights: {best_combination['weights']}")
        st.write(f"Accuracy: {best_combination['accuracy']:.2%}")

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(results_df['description'], results_df['accuracy'])
        ax.set_xlabel('Metric Combination')
        ax.set_ylabel('Accuracy')
        ax.set_title('Performance of Different Metric Combinations')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)



def run_automated_tests(test_data, all_models):
    results = []
    for sample in test_data:
        text = sample["text"]
        true_author = sample["true_author"]
        for model_name, model_func in all_models.items():
            try:
                _, _, probabilities, authorship_result, model_names, _, _ = verify_authorship(
                    text, model_func, model_name, all_models, iterations=5
                )
                predicted_author = model_names[np.argmax(probabilities)]
                result = {
                    "true_author": true_author,
                    "tested_model": model_name,
                    "predicted_author": predicted_author,
                    "authorship_result": authorship_result
                }
                results.append(result)
                print(f"Test result: {result}")  # Print each result to the terminal
            except Exception as e:
                error_msg = f"Error testing model {model_name}: {e}"
                st.warning(error_msg)
                print(error_msg)  # Print error to the terminal
                results.append({
                    "true_author": true_author,
                    "tested_model": model_name,
                    "predicted_author": "Error",
                    "authorship_result": f"Error: {e}"
                })
    return results
