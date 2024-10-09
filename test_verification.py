import app
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import json

def generate_test_cases(num_cases, models):
    prompts = [
        # General prompts
        "Explain a complex scientific concept in simple terms.",
        "Write a short story about an unexpected adventure.",
        "Describe the process of making your favorite dish.",
        "Argue for or against a controversial topic.",
        "Summarize the plot of a famous movie or book.",
        
        # Topic-specific prompts
        "Explain the concept of gene editing and its potential implications for medicine.",
        "Describe the causes and effects of hyperinflation in an economy.",
        "Analyze the impact of the Industrial Revolution on social structures in 19th century Europe.",
        "Explain the principles of quantum computing and its potential applications.",
        "Discuss the role of neurotransmitters in mood regulation and mental health.",
        
        # Complex, multi-part prompts
        "Compare and contrast renewable and non-renewable energy sources. Then, propose a strategy for transitioning to clean energy in a developing country.",
        "Explain the concept of artificial intelligence, its current applications, and potential ethical concerns. Conclude with your opinion on how AI might shape society in the next decade.",
        "Describe the process of photosynthesis in plants. Then, explain how this process contributes to the global carbon cycle and its importance in climate regulation.",
        
        # Prompts for different writing styles
        "Write a technical report on the latest advancements in autonomous vehicle technology.",
        "Compose a casual, conversational blog post about the benefits of mindfulness meditation.",
        "Create a poetic description of a sunset over the ocean, focusing on sensory details.",
        "Draft a formal business proposal for a startup idea in the field of sustainable fashion.",
        
        # Prompts requiring unique knowledge representation
        "Explain the concept of 'opportunity cost' in economics and provide real-world examples.",
        "Describe the major differences between common law and civil law legal systems.",
        "Discuss the role of epigenetics in gene expression and its implications for heredity.",
        "Explain the principles of cognitive behavioral therapy and its applications in treating anxiety disorders.",
        
        # Prompts for nuanced answers
        "Analyze the pros and cons of social media's impact on modern communication and relationships.",
        "Discuss the ethical implications of using CRISPR technology for human genetic modification.",
        "Evaluate the effectiveness of various climate change mitigation strategies, considering both environmental and economic factors.",
        "Compare the philosophical concepts of free will and determinism, and discuss their implications for personal responsibility."
    ]
    
    test_cases = []
    for _ in tqdm(range(num_cases), desc="Generating test cases"):
        true_model = random.choice(models)
        prompt = random.choice(prompts)
        text = app.all_models[true_model](prompt)
        test_cases.append({
            'prompt': prompt,
            'text': text,
            'true_model': true_model
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

def run_verification_test(test_cases):
    results = []
    for case in tqdm(test_cases, desc="Running verification tests"):
        text = case['text']
        true_model = case['true_model']
        prompt, iterations, probabilities, authorship_result, model_names, authentic_metrics, contrasting_metrics, verification_iterations, weighted_scores, weights = app.verify_authorship(
            text, app.all_models[true_model], true_model, app.all_models, iterations=5
        )
        
        # Check if probabilities is already a list, if not, convert it
        prob_list = probabilities if isinstance(probabilities, list) else probabilities.tolist()
        
        predicted_model = model_names[np.argmax(prob_list)]
        
        # Serialize the metrics and iterations
        serialized_authentic_metrics = serialize_metrics(authentic_metrics)
        serialized_contrasting_metrics = [serialize_metrics(m) for m in contrasting_metrics]
        serialized_iterations = [serialize_metrics(iter_data) for iter_data in iterations]
        serialized_verification_iterations = [serialize_metrics(iter_data) for iter_data in verification_iterations]
        
        results.append({
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
            'weights': weights.tolist()
        })
    return results

def analyze_results(results):
    df = pd.DataFrame(results)
    
    # Overall accuracy
    accuracy = (df['true_model'] == df['predicted_model']).mean()
    print(f"Overall Accuracy: {accuracy:.2%}")
    
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
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    return df

if __name__ == "__main__":
    num_test_cases = 500  # Adjust this number based on your needs and API rate limits
    test_cases_file = 'test_cases.json'
    
    # Generate and save test cases
    test_cases = generate_test_cases(num_test_cases, app.available_models)
    save_test_cases(test_cases, test_cases_file)
    
    # Load test cases and run verification
    loaded_test_cases = load_test_cases(test_cases_file)
    results = run_verification_test(loaded_test_cases)
    results_df = analyze_results(results)
    
    # Save detailed results to CSV
    results_df.to_csv('verification_test_results.csv', index=False)
    print("Detailed results saved to 'verification_test_results.csv'")

    # Optional: Save even more detailed results as JSON
    with open('verification_test_results_detailed.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Highly detailed results saved to 'verification_test_results_detailed.json'")