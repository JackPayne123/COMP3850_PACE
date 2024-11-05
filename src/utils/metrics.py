import numpy as np
from bert_score import score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from models.model_loader import load_gpt2
from collections import Counter
import math

def calculate_bertscore(reference, candidate):
    """Calculate BERTScore between reference and candidate texts"""
    P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
    return F1.item()

def calculate_cosine_similarity(reference, candidate):
    """Calculate cosine similarity between reference and candidate texts"""
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([reference, candidate])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        return 0.0

def calculate_rouge(reference, candidate):
    """Calculate ROUGE-L score between reference and candidate texts"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rougeL'].fmeasure

def ngrams(tokens, n):
    """Generate n-grams from tokens"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def modified_precision(reference_tokens, candidate_tokens, n):
    """Calculate modified precision for BLEU score"""
    reference_count = Counter(ngrams(reference_tokens, n))
    candidate_count = Counter(ngrams(candidate_tokens, n))
    
    clipped_count = {
        ngram: min(count, reference_count.get(ngram, 0)) 
        for ngram, count in candidate_count.items()
    }
    
    numerator = sum(clipped_count.values())
    denominator = max(1, sum(candidate_count.values()))
    
    return numerator / denominator if denominator != 0 else 0

def brevity_penalty(reference_tokens, candidate_tokens):
    """Calculate brevity penalty for BLEU score"""
    c = len(candidate_tokens)
    r = len(reference_tokens)
    
    if c == 0:
        return 0
    elif c < r:
        return math.exp(1 - r/c)
    else:
        return 1

def calculate_bleu(reference, candidate, weights=(0.7, 0.3)):
    """Calculate BLEU score between reference and candidate texts"""
    try:
        reference_tokens = word_tokenize(reference.lower())
        candidate_tokens = word_tokenize(candidate.lower())
        
        if len(candidate_tokens) == 0:
            return 0.0
            
        # Calculate n-gram precisions
        precisions = [
            modified_precision(reference_tokens, candidate_tokens, n)
            for n in range(1, len(weights) + 1)
        ]
        
        # Calculate logarithmic mean of precisions
        log_precision_sum = 0
        for weight, precision in zip(weights, precisions):
            if precision > 0:
                log_precision_sum += weight * math.log(precision)
            else:
                return 0.0
        
        bp = brevity_penalty(reference_tokens, candidate_tokens)
        bleu_score = bp * math.exp(log_precision_sum)
        
        return max(0.0, min(1.0, bleu_score))  # Ensure score is between 0 and 1
        
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0

def calculate_meteor(reference, candidate):
    """Calculate METEOR score between reference and candidate texts"""
    reference_tokens = word_tokenize(reference)
    candidate_tokens = word_tokenize(candidate)
    try:
        return meteor_score([reference_tokens], candidate_tokens)
    except:
        return 0.0

def calculate_perplexity(text):
    """Calculate perplexity score for the text"""
    model, tokenizer = load_gpt2()
    try:
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
        return torch.exp(outputs.loss).item()
    except:
        return float('inf')

def calculate_metrics(reference, candidates):
    """Calculate all metrics between reference and candidate texts"""
    metrics = []
    for candidate in candidates:
        bertscore = calculate_bertscore(reference, candidate)
        cosine = calculate_cosine_similarity(reference, candidate)
        rouge = calculate_rouge(reference, candidate)
        bleu = calculate_bleu(reference, candidate)
        meteor = calculate_meteor(reference, candidate)
        perplexity = calculate_perplexity(candidate)
        
        metrics.append({
            'BERTScore': bertscore,
            'Cosine Similarity': cosine,
            'ROUGE-L': rouge,
            'BLEU': bleu,
            'METEOR': meteor,
            'Perplexity': perplexity
        })
    return metrics

# Add the rest of the metric calculation functions here... 