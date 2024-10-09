import warnings
import logging
import transformers
from streamlit.runtime.scriptrunner import add_script_run_ctx
import streamlit as st

# Suppress specific warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")
warnings.filterwarnings("ignore", message="Some weights of")

# Set logging level for transformers
transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.ERROR)

print("Script is starting...")

import os
print("Imported os")
import time
print("Imported time")
import ast
print("Imported ast")
import numpy as np
print("Imported numpy")
import pandas as pd
print("Imported pandas")
import matplotlib.pyplot as plt
print("Imported matplotlib")

print("About to import nltk...")
try:
    import nltk
    print("Imported nltk successfully")
except Exception as e:
    print(f"Error importing nltk: {e}")
    import traceback
    traceback.print_exc()

print("Continuing with other imports...")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    print("Imported NLTK modules successfully")
except Exception as e:
    print(f"Error importing NLTK modules: {e}")
    import traceback
    traceback.print_exc()

from rouge_score import rouge_scorer
print("Imported rouge_scorer")
from sklearn.metrics import confusion_matrix, classification_report
print("Imported sklearn metrics")
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
print("Imported concurrent and itertools")

# Move NLTK downloads to a separate function
def download_nltk_data():
    print("Downloading NLTK data...")
    try:
        nltk.download('wordnet', quiet=False)
        nltk.download('omw-1.4', quiet=False)
        nltk.download('punkt', quiet=False)  # Add this line
        print("NLTK data downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        import traceback
        traceback.print_exc()

# Import functions and models from the main app
print("Importing from app...")
from app import (
    all_models,
    generate_text_openai_simple,
    generate_text_claude_simple,
    generate_text_gemini_simple,
    generate_text_ollama_simple,
    calculate_bertscore,
    calculate_cosine_similarity,
    calculate_perplexity,
    calculate_authorship_probability,
    determine_authorship,
    normalize_scores,
    is_ollama_available,
)
print("Imports from app completed")

# Extend metrics to include ROUGE and METEOR scores
def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rougeL'].fmeasure

from nltk.tokenize import word_tokenize

def calculate_meteor(reference, candidate):
    reference_tokens = word_tokenize(reference)
    candidate_tokens = word_tokenize(candidate)
    return meteor_score([reference_tokens], candidate_tokens)

# Update the function to include new metrics
def calculate_authorship_probability_extended(authentic_scores, contrasting_scores, weights):
    all_scores = np.array([authentic_scores] + contrasting_scores)

    # Invert perplexity so that higher is better
    all_scores[:, 2] = 1 / (1 + all_scores[:, 2])

    # Normalize scores
    normalized_scores = normalize_scores(all_scores)

    # Apply weights
    weighted_scores = normalized_scores * weights

    # Sum weighted scores
    final_scores = weighted_scores.sum(axis=1)

    # Convert to probabilities
    probabilities = np.exp(final_scores) / np.sum(np.exp(final_scores))

    return probabilities

def verify_authorship_extended(text, authentic_model, authentic_name, all_models, iterations, weights):
    authentic_regens = []
    contrasting_scores = []
    results = {}
    model_names = [authentic_name]

    # Generate regenerations for the authentic model
    current_text = text
    for _ in range(iterations):
        current_text = authentic_model(current_text)
        authentic_regens.append(current_text)

    # Calculate metrics for the authentic model
    authentic_bertscore = calculate_bertscore(text, current_text)
    authentic_cosine = calculate_cosine_similarity(text, current_text)
    authentic_perplexity = calculate_perplexity(current_text)
    authentic_rouge = calculate_rouge(text, current_text)
    authentic_meteor = calculate_meteor(text, current_text)

    authentic_scores = [authentic_bertscore, authentic_cosine, authentic_perplexity, authentic_rouge, authentic_meteor]
    results[authentic_name] = {
        'BERTScore': authentic_bertscore,
        'Cosine Similarity': authentic_cosine,
        'Perplexity': authentic_perplexity,
        'ROUGE-L': authentic_rouge,
        'METEOR': authentic_meteor,
    }

    # Generate and evaluate contrasting models
    for model_name, model_func in all_models.items():
        if model_name != authentic_name:
            try:
                # Generate a single regeneration
                regenerated_text = model_func(text)
                bertscore = calculate_bertscore(text, regenerated_text)
                cosine = calculate_cosine_similarity(text, regenerated_text)
                perplexity = calculate_perplexity(regenerated_text)
                rouge = calculate_rouge(text, regenerated_text)
                meteor = calculate_meteor(text, regenerated_text)

                scores = [bertscore, cosine, perplexity, rouge, meteor]
                contrasting_scores.append(scores)
                results[model_name] = {
                    'BERTScore': bertscore,
                    'Cosine Similarity': cosine,
                    'Perplexity': perplexity,
                    'ROUGE-L': rouge,
                    'METEOR': meteor,
                }
                model_names.append(model_name)
            except Exception as e:
                print(f"Error processing model {model_name}: {e}")

    probabilities = calculate_authorship_probability_extended(authentic_scores, contrasting_scores, weights)
    authorship_result = determine_authorship(probabilities, model_names)

    return current_text, results, probabilities, authorship_result, model_names

def run_statistical_tests(test_results):
    from scipy.stats import ttest_rel

    # Collect metrics for authentic and non-authentic texts
    bert_scores_authentic = []
    bert_scores_contrasting = []
    rouge_scores_authentic = []
    rouge_scores_contrasting = []
    meteor_scores_authentic = []
    meteor_scores_contrasting = []

    for result in test_results:
        true_author = result['true_author']
        predicted_author = result['predicted_author']
        metrics = result['metrics']

        if predicted_author == true_author:
            bert_scores_authentic.append(metrics[true_author]['BERTScore'])
            rouge_scores_authentic.append(metrics[true_author]['ROUGE-L'])
            meteor_scores_authentic.append(metrics[true_author]['METEOR'])
        else:
            for model_name, model_metrics in metrics.items():
                if model_name != true_author:
                    bert_scores_contrasting.append(model_metrics['BERTScore'])
                    rouge_scores_contrasting.append(model_metrics['ROUGE-L'])
                    meteor_scores_contrasting.append(model_metrics['METEOR'])

    # Perform paired t-tests
    bert_t_stat, bert_p_value = ttest_rel(bert_scores_authentic, bert_scores_contrasting)
    rouge_t_stat, rouge_p_value = ttest_rel(rouge_scores_authentic, rouge_scores_contrasting)
    meteor_t_stat, meteor_p_value = ttest_rel(meteor_scores_authentic, meteor_scores_contrasting)

    statistical_results = {
        'BERTScore': {'t_stat': bert_t_stat, 'p_value': bert_p_value},
        'ROUGE-L': {'t_stat': rouge_t_stat, 'p_value': rouge_p_value},
        'METEOR': {'t_stat': meteor_t_stat, 'p_value': meteor_p_value},
    }

    return statistical_results

def run_extended_tests(prompts, models_to_test, authentic_model_name, num_iterations=5, weights=None):
    if weights is None:
        # Default weights: [BERTScore, Cosine Similarity, Inverse Perplexity, ROUGE-L, METEOR]
        weights = np.array([0.3, 0.25, 0.15, 0.15, 0.15])

    test_results = []
    total_tests = len(prompts)
    start_time = time.time()

    def process_prompt(prompt):
        result = {}
        try:
            text = models_to_test[authentic_model_name](prompt)
            regenerated_text, metrics, probabilities, authorship_result, model_names = verify_authorship_extended(
                text,
                models_to_test[authentic_model_name],
                authentic_model_name,
                models_to_test,
                iterations=num_iterations,
                weights=weights,
            )

            predicted_author = model_names[np.argmax(probabilities)]
            result = {
                'prompt': prompt,
                'original_text': text,
                'true_author': authentic_model_name,
                'predicted_author': predicted_author,
                'authorship_result': authorship_result,
                'probabilities': dict(zip(model_names, probabilities)),
                'metrics': metrics,
                'regenerated_text': regenerated_text,
            }
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")
        return result

    # Use multithreading to process prompts concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Add Streamlit context to threads
        for t in executor._threads:
            add_script_run_ctx(t)

        future_to_prompt = {executor.submit(process_prompt, prompt): prompt for prompt in prompts}
        for future in as_completed(future_to_prompt):
            prompt = future_to_prompt[future]
            try:
                result = future.result()
                if result:
                    test_results.append(result)
            except Exception as e:
                print(f"Error in future for prompt '{prompt}': {e}")

    elapsed_time = time.time() - start_time
    print(f"Completed {total_tests} tests in {elapsed_time:.2f} seconds.")

    return test_results

def test_metric_combinations(test_data, all_models, metric_combinations):
    results = []
    total_combinations = len(metric_combinations)

    print(f"Starting metric combination tests. Total combinations: {total_combinations}")

    for index, (weights, description) in enumerate(metric_combinations, 1):
        print(f"\nTesting combination {index}/{total_combinations}: {description}")
        correct_predictions = 0
        total_predictions = 0

        for sample_index, sample in enumerate(test_data, 1):
            if sample_index % 5 == 0 or sample_index == 1:  # Print progress every 5 samples and for the first sample
                print(f"  Processing sample {sample_index}/{len(test_data)}")

            text = sample["text"]
            true_author = sample["true_author"]

            authentic_scores = []
            contrasting_scores = []

            for model_name, model_func in all_models.items():
                regenerated_text = model_func(text)
                bertscore = calculate_bertscore(text, regenerated_text)
                cosine = calculate_cosine_similarity(text, regenerated_text)
                perplexity = calculate_perplexity(regenerated_text)
                rouge = calculate_rouge(text, regenerated_text)
                meteor = calculate_meteor(text, regenerated_text)
                scores = [bertscore, cosine, 1 / (1 + perplexity), rouge, meteor]

                if model_name == true_author:
                    authentic_scores = scores
                else:
                    contrasting_scores.append(scores)

            probabilities = calculate_authorship_probability_extended(authentic_scores, contrasting_scores, weights)
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
        print(f"Completed combination {index}/{total_combinations}. Accuracy: {accuracy:.2%}")

    print("\nAll metric combination tests completed.")
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
    
    total_models = len(models)
    total_samples = total_models * num_samples_per_model
    
    print(f"Generating test data: {total_samples} total samples from {total_models} models")
    
    for model_index, (model_name, model_func) in enumerate(models.items(), 1):
        print(f"\nGenerating samples for model {model_index}/{total_models}: {model_name}")
        for i in range(num_samples_per_model):
            if (i + 1) % 5 == 0 or i == 0:  # Print progress every 5 samples and for the first sample
                print(f"  Generating sample {i+1}/{num_samples_per_model}")
            prompt = prompts[i % len(prompts)]
            text = model_func(prompt)
            test_data.append({"text": text, "true_author": model_name})
    
    print(f"\nTest data generation complete. Total samples generated: {len(test_data)}")
    return test_data

def main():
    print("Main function started")
    download_nltk_data()

    print("Starting main function (small test version)...")

    # Reduced example prompts for testing
    example_prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a short story about a time traveler.",
    ]
    print(f"Loaded {len(example_prompts)} example prompts.")

    # Select models to test (reduced set)
    models_to_test = {
        "OpenAI": generate_text_openai_simple,
        "Claude": generate_text_claude_simple,
    }
    if is_ollama_available():
        models_to_test["Ollama (LLaMA)"] = generate_text_ollama_simple
    print(f"Selected {len(models_to_test)} models for testing: {', '.join(models_to_test.keys())}")

    # Generate test data (reduced amount)
    print("\nStarting test data generation...")
    test_data = generate_test_data(models_to_test, num_samples_per_model=2)
    print(f"Test data generation completed. Total samples: {len(test_data)}")

    # Temporarily disable logging for specific operations
    logging.disable(logging.WARNING)
    
    # Run metric combination tests (reduced combinations)
    print("\nRunning metric combination tests...")
    metric_combinations = [
        ([1, 0, 0, 0, 0], 'BERTScore only'),
        ([0.2, 0.2, 0.2, 0.2, 0.2], 'Equal weights (all metrics)'),
        ([0.3, 0.25, 0.15, 0.15, 0.15], 'Current weights'),
    ]
    metric_results = test_metric_combinations(test_data, models_to_test, metric_combinations)
    
    # Re-enable logging
    logging.disable(logging.NOTSET)

    # Sort results by accuracy
    metric_results.sort(key=lambda x: x['accuracy'], reverse=True)

    # Display results
    print("\nMetric Combinations Results:")
    for i, result in enumerate(metric_results, 1):
        print(f"{i}. {result['description']}")
        print(f"   Weights: {result['weights']}")
        print(f"   Accuracy: {result['accuracy']:.2%}")

    # Visualize results
    print("\nGenerating performance visualization...")
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(metric_results)), [r['accuracy'] for r in metric_results])
    plt.xlabel('Metric Combination')
    plt.ylabel('Accuracy')
    plt.title('Performance of Different Metric Combinations')
    plt.xticks(range(len(metric_results)), [r['description'] for r in metric_results], rotation=90)
    plt.tight_layout()
    plt.savefig('metric_combinations_performance.png')
    print("Performance visualization saved as 'metric_combinations_performance.png'")

    # Use the best performing weights for the extended tests
    best_weights = metric_results[0]['weights']
    print(f"\nUsing best performing weights: {best_weights}")

    # Run extended tests with the best weights (reduced iterations)
    print("\nRunning extended tests with best weights...")
    test_results = run_extended_tests(
        prompts=example_prompts,
        models_to_test=models_to_test,
        authentic_model_name="OpenAI",
        num_iterations=2,
        weights=best_weights
    )

    # Analyze results
    print("\nAnalyzing extended test results...")
    accuracy, cm, report = analyze_extended_results(test_results)

    # Display results
    print("\nOverall Accuracy: {:.2%}".format(accuracy))
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Perform statistical tests
    print("\nPerforming statistical significance tests...")
    statistical_results = run_statistical_tests(test_results)
    print("\nStatistical Significance Tests:")
    for metric, stats in statistical_results.items():
        print(f"{metric}: t-statistic = {stats['t_stat']:.4f}, p-value = {stats['p_value']:.4f}")

    print("\nMain function (small test version) completed.")

def analyze_extended_results(test_results):
    df = pd.DataFrame(test_results)
    unique_authors = sorted(set(df["true_author"]) | set(df["predicted_author"]))

    accuracy = (df["true_author"] == df["predicted_author"]).mean()

    true_labels = df["true_author"]
    predicted_labels = df["predicted_author"]
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_authors)
    cm_df = pd.DataFrame(cm, index=unique_authors, columns=unique_authors)

    report = classification_report(true_labels, predicted_labels, labels=unique_authors)

    return accuracy, cm_df, report

if __name__ == "__main__":
    print("Script started. Calling main function (small test version)...")
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    print("Script execution completed.")