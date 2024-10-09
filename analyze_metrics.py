import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import defaultdict

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def parse_probabilities(prob_string):
    return eval(prob_string)

def parse_metrics(metrics_string):
    return eval(metrics_string)

def analyze_metrics(df, exclude_model=None):
    metrics = ['BERTScore', 'Cosine Similarity', 'ROUGE-L', 'BLEU', 'METEOR', 'Perplexity']
    metric_accuracies = {}

    for metric in metrics:
        correct_predictions = 0
        total_predictions = 0

        for _, row in df.iterrows():
            true_model = row['true_model']
            
            if exclude_model and true_model == exclude_model:
                continue

            authentic_metrics = parse_metrics(row['authentic_metrics'])
            contrasting_metrics = parse_metrics(row['contrasting_metrics'])

            authentic_value = authentic_metrics[metric]
            contrasting_values = [m[metric] for m in contrasting_metrics]

            if metric != 'Perplexity':
                predicted_model = true_model if authentic_value > max(contrasting_values) else 'Other'
            else:
                predicted_model = true_model if authentic_value < min(contrasting_values) else 'Other'

            if predicted_model == true_model:
                correct_predictions += 1
            total_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        metric_accuracies[metric] = accuracy

    return metric_accuracies

def analyze_model_exclusions(df):
    models = df['true_model'].unique()
    exclusion_results = {}

    for model in models:
        df_excluded = df[df['true_model'] != model]
        overall_accuracy = accuracy_score(df_excluded['true_model'], df_excluded['predicted_model'])
        exclusion_results[model] = overall_accuracy

    return exclusion_results

def print_exclusion_insights(overall_accuracy, exclusion_results):
    print("\nInsights on model exclusions:")
    for model, excluded_accuracy in exclusion_results.items():
        diff = excluded_accuracy - overall_accuracy
        print(f"If {model} were removed as a true author:")
        print(f"  Overall Accuracy: {excluded_accuracy:.2f} (Change: {diff:+.2f})")

def plot_metric_accuracies(metric_accuracies):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(metric_accuracies.keys()), y=list(metric_accuracies.values()))
    plt.title('Metric Accuracies in Predicting Correct Author')
    plt.xlabel('Metrics')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('metric_accuracies.png')
    plt.close()

def main():
    df = load_data('verification_test_results.csv')
    metric_accuracies = analyze_metrics(df)
    
    print("Metric Accuracies:")
    for metric, accuracy in metric_accuracies.items():
        print(f"{metric}: {accuracy:.2f}")

    plot_metric_accuracies(metric_accuracies)

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(df['true_model'], df['predicted_model'])
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}")

    # Calculate precision, recall, and F1-score for each model
    precision, recall, f1, _ = precision_recall_fscore_support(df['true_model'], df['predicted_model'], average=None)
    models = df['true_model'].unique()

    print("\nPer-model Metrics:")
    for model, p, r, f in zip(models, precision, recall, f1):
        print(f"{model}:")
        print(f"  Precision: {p:.2f}")
        print(f"  Recall: {r:.2f}")
        print(f"  F1-score: {f:.2f}")

    # Analyze and print insights on model exclusions
    exclusion_results = analyze_model_exclusions(df)
    print_exclusion_insights(overall_accuracy, exclusion_results)

if __name__ == "__main__":
    main()