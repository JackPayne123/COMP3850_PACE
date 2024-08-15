import streamlit as st
from openai import OpenAI
import anthropic
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from bert_score import score
import torch
import warnings

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

def verify_authorship(text, authentic_model, contrasting_model):
    authentic_regen = iterative_regeneration(text, authentic_model)
    contrasting_regen = contrasting_model(text)
    st.write(f"Constrast regen: {contrasting_regen}")
    
    authentic_bleu = calculate_bleu(text, authentic_regen)
    contrasting_bleu = calculate_bleu(text, contrasting_regen)
    
    authentic_bertscore = calculate_bertscore(text, authentic_regen)
    contrasting_bertscore = calculate_bertscore(text, contrasting_regen)
    
    return authentic_bleu, contrasting_bleu, authentic_bertscore, contrasting_bertscore

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
        authentic_bleu, contrasting_bleu, authentic_bertscore, contrasting_bertscore = verify_authorship(input_text, authentic_model, contrasting_model)
        
        st.write(f"Authentic BLEU score ({authentic_name}, iterative): {authentic_bleu}")
        st.write(f"Contrasting BLEU score ({contrasting_name}, one-step): {contrasting_bleu}")
        st.write(f"Authentic BERTScore ({authentic_name}, iterative): {authentic_bertscore}")
        st.write(f"Contrasting BERTScore ({contrasting_name}, one-step): {contrasting_bertscore}")
        
        bleu_authenticity = authentic_bleu / (authentic_bleu + contrasting_bleu)
        bertscore_authenticity = authentic_bertscore / (authentic_bertscore + contrasting_bertscore)
        
        st.write("\nAuthenticity Scores:")
        st.write(f"BLEU-based Authenticity Score: {bleu_authenticity}")
        st.write(f"BERTScore-based Authenticity Score: {bertscore_authenticity}")
        
        if bleu_authenticity > 0.5 and bertscore_authenticity > 0.5:
            st.subheader(f"\nThe text is more likely to be from {authentic_name}.")
        else:
            st.subheader(f"\nThe text is more likely to be from {contrasting_name}.")