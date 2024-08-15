from openai import OpenAI
import anthropic
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from bert_score import score
import torch
import warnings
import random

# Silence warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

def generate_text_openai(prompt):
    client = OpenAI(api_key='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')
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            #{"role": "system", "content": "You are a professional language facilitator."},
            {"role": "user", "content": f"You are a professional language facilitator. You should paraphrase the following sentence and output the final result only: {prompt}"}
        ]
    )
    return response.choices[0].message.content.strip()

def generate_text_claude(prompt):
    client = anthropic.Anthropic(api_key='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')
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
    # Iterative regeneration for the authentic model
    authentic_regen = iterative_regeneration(text, authentic_model)
    
    # One-step regeneration for the contrasting model
    contrasting_regen = contrasting_model(text)
    
    authentic_bleu = calculate_bleu(text, authentic_regen)
    contrasting_bleu = calculate_bleu(text, contrasting_regen)
    
    authentic_bertscore = calculate_bertscore(text, authentic_regen)
    contrasting_bertscore = calculate_bertscore(text, contrasting_regen)
    
    return authentic_bleu, contrasting_bleu, authentic_bertscore, contrasting_bertscore

# List of prompts
prompts = [
    "Working in groups of 2 or 4 (even numbers may work better later on), suggest a heuristic to evaluate the utility of a given board configuration.",
    "Therefore, to improve the ethical coping ability of Chinese NLP techniques and their application effectiveness in the field of ethics, the Chinese ethics knowledge base and KEPTMs for ethics domain are researched",
    "All that glitters is not gold.",
    "A journey of a thousand miles begins with a single step.",
    "Actions speak louder than words.",
    "Where there's smoke, there's fire.",
    "The early bird catches the worm.",
    "Don't count your chickens before they hatch.",
    "A picture is worth a thousand words.",
    "When in Rome, do as the Romans do."
]

# Set the authentic and contrasting models
authentic_model = generate_text_openai
contrasting_model = generate_text_claude
authentic_name = "OpenAI"
contrasting_name = "Claude"

# Run the experiment
results = []
for prompt in prompts:
    original_text = authentic_model(prompt)
    print(f"Original text: {original_text}")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        authentic_bleu, contrasting_bleu, authentic_bertscore, contrasting_bertscore = verify_authorship(original_text, authentic_model, contrasting_model)
    
    results.append({
        'authentic_bleu': authentic_bleu,
        'contrasting_bleu': contrasting_bleu,
        'authentic_bertscore': authentic_bertscore,
        'contrasting_bertscore': contrasting_bertscore
    })
    
    print(f"Authentic BLEU score ({authentic_name}, iterative): {authentic_bleu}")
    print(f"Contrasting BLEU score ({contrasting_name}, one-step): {contrasting_bleu}")
    print(f"Authentic BERTScore ({authentic_name}, iterative): {authentic_bertscore}")
    print(f"Contrasting BERTScore ({contrasting_name}, one-step): {contrasting_bertscore}")
    print("---")

# Calculate averages
avg_authentic_bleu = sum(r['authentic_bleu'] for r in results) / len(results)
avg_contrasting_bleu = sum(r['contrasting_bleu'] for r in results) / len(results)
avg_authentic_bertscore = sum(r['authentic_bertscore'] for r in results) / len(results)
avg_contrasting_bertscore = sum(r['contrasting_bertscore'] for r in results) / len(results)

print("\nAverage Scores:")
print(f"Average Authentic BLEU score ({authentic_name}, iterative): {avg_authentic_bleu}")
print(f"Average Contrasting BLEU score ({contrasting_name}, one-step): {avg_contrasting_bleu}")
print(f"Average Authentic BERTScore ({authentic_name}, iterative): {avg_authentic_bertscore}")
print(f"Average Contrasting BERTScore ({contrasting_name}, one-step): {avg_contrasting_bertscore}")

# Calculate average authenticity scores
avg_bleu_authenticity = avg_authentic_bleu / (avg_authentic_bleu + avg_contrasting_bleu)
avg_bertscore_authenticity = avg_authentic_bertscore / (avg_authentic_bertscore + avg_contrasting_bertscore)

print("\nAverage Authenticity Scores:")
print(f"BLEU-based Authenticity Score: {avg_bleu_authenticity}")
print(f"BERTScore-based Authenticity Score: {avg_bertscore_authenticity}")