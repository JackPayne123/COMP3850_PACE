import streamlit as st
from openai import OpenAI
import anthropic
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from bert_score import score
import torch
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Silence warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

@st.cache_resource
def load_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_anthropic_client():
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

@st.cache_resource
def load_gpt2():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return model, tokenizer

def generate_text_openai(prompt):
    client = load_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"You are a professional language facilitator. You should paraphrase the following sentence and output the final result only: {prompt}"}
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
            {"role": "user", "content": f"You are a professional language facilitator. You should paraphrase the following sentence and output the final result only: {prompt}"}
        ]
    )
    return response.content[0].text.strip()

def iterative_regeneration(initial_text, model_func, iterations=5):
    current_text = initial_text
    for i in range(iterations):
        current_text = model_func(current_text)
        st.write(f"iteration {i+1}: {current_text}")
        time.sleep(0.1)  # To avoid hitting rate limits
    return current_text

def calculate_bleu(reference, candidate):
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    smoothie = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)

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

def verify_authorship(text, authentic_model, contrasting_model):
    authentic_regen = iterative_regeneration(text, authentic_model)
    contrasting_regen = contrasting_model(text)
    st.write(f"Constrast regen: {contrasting_regen}")
    
    authentic_bleu = calculate_bleu(text, authentic_regen)
    contrasting_bleu = calculate_bleu(text, contrasting_regen)
    
    authentic_bertscore = calculate_bertscore(text, authentic_regen)
    contrasting_bertscore = calculate_bertscore(text, contrasting_regen)
    
    authentic_cosine = calculate_cosine_similarity(text, authentic_regen)
    contrasting_cosine = calculate_cosine_similarity(text, contrasting_regen)
    
    authentic_perplexity = calculate_perplexity(authentic_regen)
    contrasting_perplexity = calculate_perplexity(contrasting_regen)
    
    return authentic_bleu, contrasting_bleu, authentic_bertscore, contrasting_bertscore, authentic_cosine, contrasting_cosine, authentic_perplexity, contrasting_perplexity

def determine_authorship(authentic_scores, contrasting_scores):
    authentic_avg = sum(authentic_scores) / len(authentic_scores)
    contrasting_avg = sum(contrasting_scores) / len(contrasting_scores)
    threshold = 0.55  # Adjust as needed
    if authentic_avg > contrasting_avg and authentic_avg > threshold:
        return "authentic"
    elif contrasting_avg > authentic_avg and contrasting_avg > threshold:
        return "contrasting"
    else:
        return "inconclusive"

st.title("Self-Watermarking Experiment")

model_choice = st.radio(
    "Select the model to verify against:",
    ("OpenAI", "Claude")
)

if model_choice == "Claude":
    authentic_model = generate_text_claude
    contrasting_model = generate_text_openai
    authentic_name = "Claude"
    contrasting_name = "OpenAI"
else:
    authentic_model = generate_text_openai
    contrasting_model = generate_text_claude
    authentic_name = "OpenAI"
    contrasting_name = "Claude"

input_text = st.text_area("Enter the text to verify:", "The quick brown fox jumps over the lazy dog.")

if st.button("Run Verification"):
    with st.spinner("Running verification..."):
        authentic_bleu, contrasting_bleu, authentic_bertscore, contrasting_bertscore, authentic_cosine, contrasting_cosine, authentic_perplexity, contrasting_perplexity = verify_authorship(input_text, authentic_model, contrasting_model)
        
        st.write(f"Authentic BLEU score ({authentic_name}, iterative): {authentic_bleu}")
        st.write(f"Contrasting BLEU score ({contrasting_name}, one-step): {contrasting_bleu}")
        st.write(f"Authentic BERTScore ({authentic_name}, iterative): {authentic_bertscore}")
        st.write(f"Contrasting BERTScore ({contrasting_name}, one-step): {contrasting_bertscore}")
        
        bleu_authenticity = authentic_bleu / (authentic_bleu + contrasting_bleu)
        bertscore_authenticity = authentic_bertscore / (authentic_bertscore + contrasting_bertscore)
        cosine_authenticity = authentic_cosine / (authentic_cosine + contrasting_cosine)
        
        st.write("\nAuthenticity Scores:")
        st.write(f"BLEU-based Authenticity Score: {bleu_authenticity}")
        st.write(f"BERTScore-based Authenticity Score: {bertscore_authenticity}")
        st.write(f"Cosine-based Authenticity Score: {cosine_authenticity}")
        
        authentic_scores = [bleu_authenticity, bertscore_authenticity, cosine_authenticity]
        contrasting_scores = [1 - bleu_authenticity, 1 - bertscore_authenticity, 1 - cosine_authenticity]
        authorship = determine_authorship(authentic_scores, contrasting_scores)
        
        st.write("\nDetailed Metrics:")
        st.write(f"Authentic Cosine Similarity: {authentic_cosine:.4f}")
        st.write(f"Contrasting Cosine Similarity: {contrasting_cosine:.4f}")
        st.write(f"Authentic Perplexity: {authentic_perplexity:.4f}")
        st.write(f"Contrasting Perplexity: {contrasting_perplexity:.4f}")

        if authorship == "authentic":
            st.subheader(f"\nThe text is more likely to be from {authentic_name}.")
        elif authorship == "contrasting":
            st.subheader(f"\nThe text is more likely to be from {contrasting_name}.")
        else:
            st.subheader(f"\nThe authorship is inconclusive.")