import streamlit as st

# Default text for the application
DEFAULT_TEXT = """Artificial Intelligence (AI) has significantly influenced various sectors, including healthcare, education, and finance. It helps in diagnostics, personalised learning, and algorithmic trading, driving efficiency and innovation. However, ethical concerns like data privacy and bias remain challenging issues."""

# Metric weights for authorship verification
METRIC_WEIGHTS = {
    'BERTScore': 0.3,
    'Cosine Similarity': 0.3,
    'ROUGE-L': 0.15,
    'BLEU': 0.15,
    'METEOR': 0.1,
    'Perplexity': 0.05
}

# Example prompts for text generation
EXAMPLE_PROMPTS = [
    "Explain the concept of quantum entanglement in simple terms.",
    "Write a short story about a time traveler who accidentally changes history.",
    "Describe the process of photosynthesis in plants.",
    "Create a recipe for a unique fusion dish combining Italian and Japanese cuisines."
]

# Human detection settings
HUMAN_THRESHOLD = 0.2 