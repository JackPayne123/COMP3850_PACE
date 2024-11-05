import spacy
import numpy as np
from models.model_loader import load_spacy_model

def extract_human_features(text):
    """Extract linguistic features that are typically human-like"""
    nlp = load_spacy_model()
    doc = nlp(text)
    
    features = {
        'sentence_length_variance': np.var([len(sent) for sent in doc.sents]),
        'unique_words_ratio': len(set(token.text.lower() for token in doc)) / len(doc),
        'punctuation_ratio': len([token for token in doc if token.is_punct]) / len(doc),
        'named_entity_ratio': len(doc.ents) / len(doc),
        'stopwords_ratio': len([token for token in doc if token.is_stop]) / len(doc),
        'avg_word_length': np.mean([len(token.text) for token in doc]),
    }
    
    return features

def calculate_human_similarity(text):
    """Calculate similarity to typical human writing patterns"""
    features = extract_human_features(text)
    
    human_benchmarks = {
        'sentence_length_variance': 25.0,
        'unique_words_ratio': 0.6,
        'punctuation_ratio': 0.12,
        'named_entity_ratio': 0.05,
        'stopwords_ratio': 0.4,
        'avg_word_length': 4.7,
    }
    
    similarities = {}
    for feature, value in features.items():
        benchmark = human_benchmarks[feature]
        diff = abs(value - benchmark) / benchmark
        similarity = max(0, 1 - diff)
        similarities[feature] = similarity
    
    human_score = np.mean(list(similarities.values()))
    return human_score 