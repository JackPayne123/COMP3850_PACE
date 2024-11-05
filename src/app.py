import streamlit as st
import logging
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps
from mistralai import Mistral

# Import our modular components
from models.model_loader import (
    load_openai_client, load_anthropic_client, 
    load_gemini_client, load_gpt2
)
from utils.metrics import calculate_metrics
from utils.text_analysis import calculate_human_similarity, extract_human_features
from utils.verification import verify_authorship
from config.settings import (
    DEFAULT_TEXT, METRIC_WEIGHTS, 
    EXAMPLE_PROMPTS, HUMAN_THRESHOLD
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence warnings
import warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

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

# Text generation functions
def generate_text_openai(prompt):
    client = load_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_text_claude(prompt):
    client = load_anthropic_client()
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

def generate_text_gemini(prompt):
    model = load_gemini_client()
    if model is None:
        return "Gemini model client is not available."
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Failed to generate text using Gemini: {e}"

@rate_limit(4.5)
def generate_text_mistral(prompt):
    client = Mistral(api_key=st.session_state.get('mistral_api_key') or st.secrets["MISTRAL_API_KEY"])
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# Model mapping
ALL_MODELS = {
    "OpenAI": generate_text_openai,
    "Claude": generate_text_claude,
    "Gemini": generate_text_gemini,
    "Mistral": generate_text_mistral,
}

def setup_sidebar():
    """Setup the sidebar with API keys and options"""
    st.sidebar.header("API Keys (Optional)")
    st.session_state['openai_api_key'] = st.sidebar.text_input("OpenAI API Key", type="password")
    st.session_state['anthropic_api_key'] = st.sidebar.text_input("Anthropic API Key", type="password")
    st.session_state['gemini_api_key'] = st.sidebar.text_input("Gemini API Key", type="password")
    st.session_state['mistral_api_key'] = st.sidebar.text_input("Mistral API Key", type="password")

    st.sidebar.header("Regeneration Options")
    st.session_state['regeneration_method'] = st.sidebar.radio(
        "Choose regeneration method:",
        ("Summarize", "Paraphrase")
    )
    
    st.sidebar.header("Detection Options")
    st.session_state['include_human_detection'] = st.sidebar.checkbox(
        "Include Human Authorship Detection", 
        value=False
    )

def setup_input_section():
    st.title("Text Input Options")
    input_option = st.radio(
        "Choose input method:",
        ("Enter text manually", "Generate text using a model")
    )

    if input_option == "Generate text using a model":
        generation_model = st.selectbox(
            "Select model for text generation",
            list(ALL_MODELS.keys())
        )
        
        selected_prompt = st.selectbox(
            "Select an example prompt or write your own:",
            ["Write your own prompt..."] + EXAMPLE_PROMPTS
        )
        
        prompt = st.text_area(
            "Enter your prompt for text generation",
            value=selected_prompt if selected_prompt != "Write your own prompt..." else ""
        )
        
        if st.button("Generate Text"):
            try:
                generated_text = ALL_MODELS[generation_model](prompt)
                st.write("Generated Text:")
                st.write(generated_text)
                st.session_state.generated_text = generated_text
            except Exception as e:
                st.error(f"Error generating text: {str(e)}")

    return input_option

def main():
    # Initialize session state
    if 'input_text' not in st.session_state:
        st.session_state.input_text = DEFAULT_TEXT

    # Setup sidebar and input sections
    setup_sidebar()
    input_option = setup_input_section()

    st.title("Self-Watermarking Experiment")

    # Model selection
    model_choice = st.selectbox(
        "Select the model to verify against:",
        list(ALL_MODELS.keys())
    )

    # Text input
    if input_option == "Enter text manually":
        st.session_state.input_text = st.text_area(
            "Enter the text to verify:", 
            st.session_state.input_text
        )
    else:
        st.session_state.input_text = st.text_area(
            "Enter the text to verify:",
            value=st.session_state.get('generated_text', st.session_state.input_text),
            help="You can edit the generated text or enter new text here."
        )

    if st.button("Run Verification"):
        run_verification(model_choice)

def run_verification(model_choice):
    """Run the verification process for the input text"""
    authentic_model = ALL_MODELS[model_choice]
    verify_authorship(
        st.session_state.input_text,
        authentic_model,
        model_choice,
        ALL_MODELS
    )

if __name__ == "__main__":
    main()

# Add custom CSS
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
    margin-bottom: 20px;
}
.section-gap {
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True) 