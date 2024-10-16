import app
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import json
import concurrent.futures
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize

# Use the metric calculations from app.py
from app import (
    calculate_bertscore, calculate_cosine_similarity, calculate_perplexity,
    calculate_rouge, calculate_meteor, calculate_bleu, serialize_metrics,
    verify_authorship, all_models
)

def calculate_metrics(reference, candidates):
    metrics = []
    for candidate in candidates:
        bertscore = calculate_bertscore(reference, candidate)
        cosine = calculate_cosine_similarity(reference, candidate)
        rouge = calculate_rouge(reference, candidate)
        bleu = calculate_bleu(reference, candidate, weights=[0.7, 0.3])  # Use bigram weights with more emphasis on unigrams
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

def load_gpt2():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return model, tokenizer

def load_summarization_datasets():
    cnn_daily_mail = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")

    return cnn_daily_mail

def load_paraphrase_datasets():
    quora = load_dataset("quora", split="train", trust_remote_code=True)
    paranmt = load_dataset("paranmt", split="train", trust_remote_code=True)
    return quora, paranmt

def generate_test_cases(num_cases, models):
    cnn_daily_mail = load_summarization_datasets()
    #quora, paranmt = load_paraphrase_datasets()
    
    test_cases = []
    for _ in tqdm(range(num_cases), desc="Generating test cases"):
        true_model = random.choice(models)
        task = "summarize"
        

        sample = random.choice(cnn_daily_mail)
        prompt = f"Summarize the following text:\n\n{sample['article']}"
        reference = sample['highlights']

        
        text = app.all_models[true_model](prompt)
        test_cases.append({
            'prompt': prompt,
            'text': text,
            'true_model': true_model,
            'task': task,
            'reference': reference
        })
    return test_cases

def save_test_cases(test_cases, filename):
    with open(filename, 'w') as f:
        json.dump(test_cases, f, indent=2)
    print(f"Test cases saved to {filename}")

def load_test_cases(filename):
    with open(filename, 'r') as f:
        test_cases = json.load(f)
    return test_cases

def serialize_metrics(metrics):
    serialized = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)):
            serialized[key] = value
        elif isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        else:
            serialized[key] = str(value)
    return serialized

def evaluate_summary(generated, reference):
    metrics = calculate_metrics(reference, [generated])[0]
    return metrics

def evaluate_paraphrase(generated, reference):
    return evaluate_summary(generated, reference)

def run_verification_test(test_cases):
    results = []
    
    for i, case in enumerate(tqdm(test_cases, desc="Running verification tests")):
        try:
            text = case['text']
            true_model = case['true_model']
            prompt, iterations, probabilities, authorship_result, model_names, authentic_metrics, contrasting_metrics, verification_iterations, weighted_scores, weights = verify_authorship(
                text, all_models[true_model], true_model, all_models, iterations=5
            )
            
            prob_list = probabilities if isinstance(probabilities, list) else probabilities.tolist()
            predicted_model = model_names[np.argmax(prob_list)]
            
            # Evaluate the generated text against the reference
            if case['task'] == 'summarize':
                task_score = evaluate_summary(text, case['reference'])
            else:
                task_score = evaluate_paraphrase(text, case['reference'])
            
            serialized_authentic_metrics = serialize_metrics(authentic_metrics)
            serialized_contrasting_metrics = [serialize_metrics(m) for m in contrasting_metrics]
            serialized_iterations = [serialize_metrics(iter_data) for iter_data in iterations]
            serialized_verification_iterations = [serialize_metrics(iter_data) for iter_data in verification_iterations]
            
            result = {
                'prompt': case['prompt'],
                'text': text,
                'true_model': true_model,
                'predicted_model': predicted_model,
                'authorship_result': authorship_result,
                'probabilities': dict(zip(model_names, prob_list)),
                'authorship_iterations': json.dumps(serialized_iterations),
                'verification_iterations': json.dumps(serialized_verification_iterations),
                'authentic_metrics': json.dumps(serialized_authentic_metrics),
                'contrasting_metrics': json.dumps(serialized_contrasting_metrics),
                'weighted_scores': weighted_scores.tolist(),
                'weights': weights.tolist(),
                'task': case['task'],
                'task_score': task_score
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing case {i}: {str(e)}")
        
        # Save results every 10 completed tests
        if (i + 1) % 10 == 0:
            save_interim_results(results, f'interim_results_{i+1}.json')
            print(f"Saved interim results for {i+1} tests")

    print(f"Total results: {len(results)}")
    print(f"Sample result keys: {list(results[0].keys()) if results else 'No results'}")
    return results

def save_interim_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Interim results saved to {filename}")

def analyze_results(results, regeneration_method):
    df = pd.DataFrame(results)
    
    print(f"Available columns: {df.columns.tolist()}")
    print(f"Number of rows: {len(df)}")
    
    if 'true_model' in df.columns and 'predicted_model' in df.columns:
        # Overall accuracy
        accuracy = (df['true_model'] == df['predicted_model']).mean()
        print(f"Overall Accuracy ({regeneration_method}): {accuracy:.2%}")
        
        # Confusion Matrix
        true_labels = df['true_model']
        predicted_labels = df['predicted_model']
        cm = confusion_matrix(true_labels, predicted_labels, labels=app.available_models)
        
        # Classification Report
        cr = classification_report(true_labels, predicted_labels, labels=app.available_models)
        print("\nClassification Report:")
        print(cr)
        
        # Visualize Confusion Matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(app.available_models))
        plt.xticks(tick_marks, app.available_models, rotation=45)
        plt.yticks(tick_marks, app.available_models)
        plt.ylabel('True Model')
        plt.xlabel('Predicted Model')
        
        # Add text annotations to the confusion matrix
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{regeneration_method}.png')
        print(f"Confusion matrix saved as 'confusion_matrix_{regeneration_method}.png'")
    else:
        print("Warning: 'true_model' or 'predicted_model' columns not found in the results.")
    
    # Task-specific performance
    if 'task' in df.columns and 'task_score' in df.columns:
        # Assuming task_score is a dictionary, let's extract ROUGE-L scores
        df['rouge_l'] = df['task_score'].apply(lambda x: x.get('ROUGE-L', {}).get('f', 0) if isinstance(x, dict) else 0)
        
        summarization_score = df[df['task'] == 'summarize']['rouge_l'].mean()
        paraphrase_score = df[df['task'] == 'paraphrase']['rouge_l'].mean()
        print(f"Average Summarization ROUGE-L Score: {summarization_score:.4f}")
        print(f"Average Paraphrase ROUGE-L Score: {paraphrase_score:.4f}")
    else:
        print("Warning: 'task' or 'task_score' columns not found in the results.")
    
    return df

if __name__ == "__main__":
    num_test_cases = 100  # Adjust this number based on your needs and API rate limits
    test_cases_file = 'test_cases_with_datasets.json'
    
    # Generate and save test cases
    test_cases = generate_test_cases(num_test_cases, app.available_models)
    save_test_cases(test_cases, test_cases_file)
    
    # Load test cases and run verification
    loaded_test_cases = load_test_cases(test_cases_file)
    
    print("\nRunning verification test")
    results = run_verification_test(loaded_test_cases)
    
    print(f"Number of results: {len(results)}")
    if results:
        print(f"Keys in first result: {list(results[0].keys())}")
    else:
        print("No results were returned")
    
    results_df = analyze_results(results, "Summarize")
    
    # Save detailed results to CSV
    results_df.to_csv('verification_test_results.csv', index=False)
    print("Detailed results saved to 'verification_test_results.csv'")

    # Save even more detailed results as JSON
    with open('verification_test_results_detailed.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Highly detailed results saved to 'verification_test_results_detailed.json'")
