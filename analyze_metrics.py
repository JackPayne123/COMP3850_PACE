import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from ast import literal_eval
from collections import defaultdict

def load_data(file_path):
    df = pd.read_csv(file_path)
    # Convert string representations of lists and dicts to actual objects
    for col in ['probabilities', 'authorship_iterations', 'verification_iterations', 'authentic_metrics', 'contrasting_metrics', 'weighted_scores', 'weights']:
        df[col] = df[col].apply(literal_eval)
    return df

def analyze_metrics(df):
    # Debugging: Print the first few rows of weighted_scores and weights
    print("\nDebugging Weighted Scores and Weights:")
    for i, (weighted_scores, weights) in enumerate(zip(df['weighted_scores'].head(), df['weights'].head())):
        print(f"\nSample {i+1}:")
        print(f"Weighted Scores: {weighted_scores}")
        print(f"Weights: {weights}")
        
        # Check for zero weights
        zero_weights = [i for i, w in enumerate(weights) if w == 0]
        if zero_weights:
            print(f"Zero weights found at indices: {zero_weights}")
        
        # Calculate total weighted score
        total_weighted_score = sum(max(scores) for scores in weighted_scores)
        print(f"Total Weighted Score: {total_weighted_score}")

    # Debugging: Print probabilities for a few samples
    print("\nDebugging Probabilities:")
    for i, probs in enumerate(df['probabilities'].head()):
        print(f"\nSample {i+1} Probabilities:")
        for model, prob in probs.items():
            print(f"{model}: {prob:.4f}")
        predicted_model = max(probs, key=probs.get)
        print(f"Predicted Model: {predicted_model}")
        print(f"True Model: {df['true_model'].iloc[i]}")
        print(f"Correct Prediction: {predicted_model == df['true_model'].iloc[i]}")

    # Calculate correct predictions
    df['correct_prediction'] = df.apply(lambda row: row['true_model'] == max(row['probabilities'], key=row['probabilities'].get), axis=1)

    # Debugging: Print correct prediction calculation for a few samples
    print("\nDebugging Correct Prediction Calculation:")
    for i, row in df.head().iterrows():
        print(f"\nSample {i+1}:")
        print(f"True Model: {row['true_model']}")
        print(f"Predicted Model: {max(row['probabilities'], key=row['probabilities'].get)}")
        print(f"Correct Prediction: {row['correct_prediction']}")

    # 1. Metric Correlation Analysis
    metric_columns = ['BERTScore', 'Cosine Similarity', 'ROUGE-L', 'BLEU', 'METEOR', 'Perplexity']
    metric_data = pd.DataFrame([row['authentic_metrics'] for _, row in df.iterrows()])
    correlation_matrix = metric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Metric Correlation Heatmap')
    plt.savefig('metric_correlation.png')
    plt.close()

    # 2. Model-specific Performance
    df['correct_prediction'] = df['true_model'] == df['predicted_model']
    model_performance = df.groupby('true_model')['correct_prediction'].mean()
    print("Model-specific Performance:")
    print(model_performance)

    # 3. Prompt Complexity Analysis
    df['prompt_length'] = df['prompt'].str.len()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='prompt_length', y='correct_prediction', data=df)
    plt.title('Prompt Length vs Prediction Accuracy')
    plt.savefig('prompt_complexity.png')
    plt.close()

    # 4. Confidence Threshold Analysis
    df['max_probability'] = df['probabilities'].apply(lambda x: max(x.values()))
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='max_probability', y='correct_prediction', data=df)
    plt.title('Prediction Confidence vs Accuracy')
    plt.savefig('confidence_analysis.png')
    plt.close()

    # 5. Weighted Score Analysis
    df['total_weighted_score'] = df['weighted_scores'].apply(lambda x: sum(max(scores) for scores in x))
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='total_weighted_score', y='correct_prediction', data=df)
    plt.title('Total Weighted Score vs Accuracy')
    plt.savefig('weighted_score_analysis.png')
    plt.close()

    # 6. Error Analysis
    misclassified = df[df['true_model'] != df['predicted_model']]
    print("\nMisclassified cases:")
    for _, row in misclassified.iterrows():
        print(f"True: {row['true_model']}, Predicted: {row['predicted_model']}, Prompt: {row['prompt'][:50]}...")

    # 7. Iteration Analysis
    df['num_authorship_iterations'] = df['authorship_iterations'].apply(len)
    df['num_verification_iterations'] = df['verification_iterations'].apply(len)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='num_authorship_iterations', y='correct_prediction', data=df)
    plt.title('Number of Authorship Iterations vs Accuracy')
    plt.savefig('authorship_iterations_analysis.png')
    plt.close()

    # 8. Perplexity Range Analysis
    df['perplexity'] = df['authentic_metrics'].apply(lambda x: x['Perplexity'])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='perplexity', y='correct_prediction', data=df)
    plt.title('Perplexity vs Accuracy')
    plt.savefig('perplexity_analysis.png')
    plt.close()

    # Overall accuracy
    overall_accuracy = accuracy_score(df['true_model'], df['predicted_model'])
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}")

    # Precision, recall, and F1-score for each model
    precision, recall, f1, _ = precision_recall_fscore_support(df['true_model'], df['predicted_model'], average=None)
    models = df['true_model'].unique()

    print("\nPer-model Metrics:")
    for model, p, r, f in zip(models, precision, recall, f1):
        print(f"{model}:")
        print(f"  Precision: {p:.2f}")
        print(f"  Recall: {r:.2f}")
        print(f"  F1-score: {f:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(df['true_model'], df['predicted_model'], labels=models)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=models, yticklabels=models)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Dataset balance check
    print("\nDataset Balance:")
    print(df['true_model'].value_counts())

    # Correct prediction counts
    print("\nCorrect Prediction Counts:")
    print(df.groupby('true_model')['correct_prediction'].sum())

    # Confusion matrix raw data
    print("\nConfusion Matrix Raw Data:")
    for true_model in df['true_model'].unique():
        for pred_model in df['predicted_model'].unique():
            count = ((df['true_model'] == true_model) & (df['predicted_model'] == pred_model)).sum()
            print(f"True: {true_model}, Predicted: {pred_model}, Count: {count}")

    # Analyze misclassifications
    print("\nMisclassification Analysis:")
    misclassified = df[df['true_model'] != df['predicted_model']].head(10)  # Analyze first 10 misclassifications
    for _, row in misclassified.iterrows():
        print(f"\nTrue Model: {row['true_model']}, Predicted: {row['predicted_model']}")
        print(f"Probabilities: {row['probabilities']}")
        print(f"Weighted Scores: {row['weighted_scores']}")

    # Calculate probability of model being predicted when not the true author
    print("\nProbability of model being predicted when not the true author:")
    model_predictions = defaultdict(lambda: defaultdict(int))
    total_non_author_samples = defaultdict(int)

    for _, row in df.iterrows():
        true_model = row['true_model']
        predicted_model = row['predicted_model']
        for model in row['probabilities'].keys():
            if model != true_model:
                total_non_author_samples[model] += 1
                if predicted_model == model:
                    model_predictions[model]['correct'] += 1

    for model in total_non_author_samples.keys():
        correct_predictions = model_predictions[model]['correct']
        total_samples = total_non_author_samples[model]
        probability = correct_predictions / total_samples if total_samples > 0 else 0
        print(f"{model}: {probability:.2%} ({correct_predictions}/{total_samples})")

    # Calculate how often the true author is the second highest predicted model
    print("\nAnalysis of second highest predictions:")
    second_highest_correct = 0
    total_incorrect = 0
    model_second_highest = defaultdict(int)
    model_total_incorrect = defaultdict(int)

    for _, row in df.iterrows():
        true_model = row['true_model']
        probabilities = row['probabilities']
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_probs[0][0] != true_model:
            total_incorrect += 1
            model_total_incorrect[true_model] += 1
            if sorted_probs[1][0] == true_model:
                second_highest_correct += 1
                model_second_highest[true_model] += 1

    print(f"Overall, when incorrect, the true author is the second highest prediction "
          f"{second_highest_correct/total_incorrect:.2%} of the time "
          f"({second_highest_correct}/{total_incorrect})")

    print("\nBreakdown by model:")
    for model in df['true_model'].unique():
        if model_total_incorrect[model] > 0:
            percentage = model_second_highest[model] / model_total_incorrect[model]
            print(f"{model}: {percentage:.2%} "
                  f"({model_second_highest[model]}/{model_total_incorrect[model]})")
        else:
            print(f"{model}: N/A (No incorrect predictions)")

    # Calculate correct predictions per model
    print("\nCorrect predictions per model:")
    model_correct = defaultdict(int)
    model_total = defaultdict(int)

    for _, row in df.iterrows():
        true_model = row['true_model']
        predicted_model = row['predicted_model']
        model_total[true_model] += 1
        if true_model == predicted_model:
            model_correct[true_model] += 1

    for model in df['true_model'].unique():
        correct = model_correct[model]
        total = model_total[model]
        accuracy = correct / total if total > 0 else 0
        print(f"{model}: {accuracy:.2%} ({correct}/{total})")

def main():
    df = load_data('interim_results_500.csv')
    df['predicted_model'] = df['probabilities'].apply(lambda x: max(x, key=x.get))
    analyze_metrics(df)

if __name__ == "__main__":
    main()
