import streamlit as st
from openai import OpenAI
import anthropic
import google.generativeai as genai
from mistralai import Mistral
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
import time
import spacy

logger = logging.getLogger(__name__)

@st.cache_resource
def load_openai_client():
    logger.info("Loading OpenAI client...")
    start_time = time.time()
    client = OpenAI(api_key=st.session_state.get('openai_api_key') or st.secrets["OPENAI_API_KEY"])
    logger.info(f"OpenAI client loaded in {time.time() - start_time:.2f} seconds")
    return client

@st.cache_resource
def load_anthropic_client():
    logger.info("Loading Anthropic client...")
    start_time = time.time()
    client = anthropic.Anthropic(api_key=st.session_state.get('anthropic_api_key') or st.secrets["ANTHROPIC_API_KEY"])
    logger.info(f"Anthropic client loaded in {time.time() - start_time:.2f} seconds")
    return client

@st.cache_resource
def load_gemini_client():
    logger.info("Loading Gemini client...")
    start_time = time.time()
    try:
        genai.configure(api_key=st.session_state.get('gemini_api_key') or st.secrets["GEMINI_API_KEY"])
        client = genai.GenerativeModel('gemini-1.5-pro')
        logger.info(f"Gemini client loaded in {time.time() - start_time:.2f} seconds")
        return client
    except Exception as e:
        logger.error(f"Failed to load Gemini client: {e}")
        return None

@st.cache_resource
def load_gpt2():
    logger.info("Loading GPT-2 model...")
    start_time = time.time()
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    logger.info(f"GPT-2 model loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm") 