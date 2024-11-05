import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .metrics import calculate_metrics
from .text_analysis import calculate_human_similarity, extract_human_features
from config.settings import HUMAN_THRESHOLD, METRIC_WEIGHTS

def verify_authorship(text, authentic_model, authentic_name, all_models, iterations=5):
    """Main verification logic for text authorship"""
    with st.spinner("Running verification..."):
        results = verify_authorship_core(
            text, authentic_model, authentic_name, all_models, iterations
        )
        display_verification_results(results)
        return results

def verify_authorship_core(text, authentic_model, authentic_name, all_models, iterations=5):
    """Core verification logic without display elements"""
    authorship_iterations = []
    verification_iterations = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = iterations + len(all_models)
    current_step = 0
    
    # Stage I: Iterative Regeneration
    status_text.text("Stage I: Performing iterative regeneration...")
    current_text = text
    
    for i in range(iterations):
        if st.session_state['regeneration_method'] == "Summarize":
            prompt = f"Please summarize the following text into one sentence:\n\n{current_text}"
        else:
            prompt = f"Please paraphrase the following text, maintaining its original meaning:\n\n{current_text}"
            
        current_text = authentic_model(prompt)
        authorship_iterations.append({
            'iteration': i + 1,
            'text': current_text
        })
        
        current_step += 1
        progress_bar.progress(current_step / total_steps)
    
    final_output = current_text
    
    # Stage II: Verification
    status_text.text("Stage II: Performing verification steps...")
    contrasting_models = {name: func for name, func in all_models.items() if name != authentic_name}
    
    if st.session_state['regeneration_method'] == "Summarize":
        prompt = f"Please summarize the following text into one sentence:\n\n{final_output}"
    else:
        prompt = f"Please paraphrase the following text, maintaining its original meaning:\n\n{final_output}"
    
    # Authentic model verification
    y_a = authentic_model(prompt)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # Contrasting models verification
    y_cs = []
    for name, model in contrasting_models.items():
        status_text.text(f"Verifying against {name}...")
        y_cs.append(model(prompt))
        current_step += 1
        progress_bar.progress(current_step / total_steps)
    
    verification_iterations.append({
        'authentic_output': y_a,
        'contrasting_outputs': dict(zip(contrasting_models.keys(), y_cs))
    })
    
    # Calculate metrics
    status_text.text("Calculating metrics...")
    authentic_metrics = calculate_metrics(final_output, [y_a])[0]
    contrasting_metrics = calculate_metrics(final_output, y_cs)
    
    # Calculate authorship probability
    probabilities, weighted_scores, weights = calculate_authorship_probability(
        authentic_metrics, contrasting_metrics
    )
    
    model_names = [authentic_name] + list(contrasting_models.keys())
    
    # Add human detection if enabled
    if st.session_state['include_human_detection']:
        human_score = calculate_human_similarity(text)
        all_probabilities = np.append(probabilities, human_score)
        all_model_names = model_names + ['Human']
        
        if human_score < HUMAN_THRESHOLD:
            human_prob = all_probabilities[-1]
            redistribution_factor = 0.5
            remaining_human_prob = human_prob * (1 - redistribution_factor)
            redistribution_amount = human_prob * redistribution_factor
            
            all_probabilities = all_probabilities[:-1]
            all_probabilities += (redistribution_amount * (all_probabilities / np.sum(all_probabilities)))
            all_probabilities = np.append(all_probabilities, remaining_human_prob)
        
        all_probabilities = all_probabilities / np.sum(all_probabilities)
        authorship_result = "Human" if (np.argmax(all_probabilities) == len(all_model_names) - 1 and 
                                      human_score >= HUMAN_THRESHOLD) else all_model_names[np.argmax(all_probabilities)]
    else:
        all_probabilities = probabilities
        all_model_names = model_names
        authorship_result = model_names[np.argmax(probabilities)]
    
    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return {
        'prompt': prompt,
        'authorship_iterations': authorship_iterations,
        'probabilities': all_probabilities,
        'authorship_result': authorship_result,
        'model_names': all_model_names,
        'authentic_metrics': authentic_metrics,
        'contrasting_metrics': contrasting_metrics,
        'verification_iterations': verification_iterations,
        'weighted_scores': weighted_scores,
        'weights': weights
    }

def display_verification_results(results):
    """Display verification results with visualizations"""
    # Display authorship result
    if results['authorship_result'] == "Human":
        st.success(f"**Authorship Result:** {results['authorship_result']}")
        st.info("The text appears to be written by a human")
    elif results['authorship_result'] == results['model_names'][0]:
        st.success(f"**Authorship Result:** Authentic (Original Model: {results['model_names'][0]})")
        st.info(f"The predicted original model that generated the text is **{results['model_names'][0]}**")
    else:
        st.error(f"**Authorship Result:** {results['authorship_result']}")
        st.info(f"The predicted original model that generated the text is **{results['authorship_result']}**")

    # Model Probabilities
    st.markdown("### Model Probabilities")
    prob_df = pd.DataFrame({
        'Model': results['model_names'],
        'Probability': results['probabilities']
    })
    prob_styler = prob_df.style.format({'Probability': '{:.2%}'}).hide(axis='index')
    st.write(prob_styler.to_html(), unsafe_allow_html=True)

    # Regeneration Iterations
    st.markdown("### Regeneration Iterations")
    for iteration in results['authorship_iterations']:
        st.markdown(f"**Iteration {iteration['iteration']}:**")
        st.markdown(f'<div class="wrapped-text">{iteration["text"]}</div>', unsafe_allow_html=True)

    # Verification Iterations
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### Verification Iterations")
    st.markdown("**Authentic Output:**")
    st.markdown(f'<div class="wrapped-text">{results["verification_iterations"][0]["authentic_output"]}</div>', 
                unsafe_allow_html=True)
    
    st.markdown("**Contrasting Outputs:**")
    for model_name, output in results["verification_iterations"][0]['contrasting_outputs'].items():
        st.markdown(f"**{model_name}:**")
        st.markdown(f'<div class="wrapped-text">{output}</div>', unsafe_allow_html=True)

    # Display metrics and visualizations
    display_metrics_and_visualizations(results)

def display_metrics_and_visualizations(results):
    """Display detailed metrics and create visualizations"""
    st.markdown("### Raw Metric Scores")
    
    metric_names = list(results['authentic_metrics'].keys())
    raw_scores_df = pd.DataFrame(
        [results['authentic_metrics']] + results['contrasting_metrics'],
        columns=metric_names,
        index=results['model_names'][:-1] if 'Human' in results['model_names'] else results['model_names']
    )
    
    st.dataframe(
        raw_scores_df.style
        .format("{:.4f}")
        .background_gradient(cmap="YlGnBu")
    )

    # Metric Contributions
    st.markdown("### Metric Contributions to Final Probability")
    contribution_df = pd.DataFrame(
        results['weighted_scores'],
        columns=metric_names,
        index=results['model_names']
    )
    
    st.dataframe(
        contribution_df.style
        .format("{:.4f}")
        .background_gradient(cmap="YlGnBu")
    )

    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    contribution_df.plot(kind='bar', stacked=True, ax=ax)
    plt.title("Metric Contributions to Final Probability")
    plt.xlabel("Models")
    plt.ylabel("Weighted Score")
    plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def calculate_authorship_probability(authentic_metrics, contrasting_metrics):
    """Calculate probability of authorship based on metrics"""
    weights = np.array(list(METRIC_WEIGHTS.values()))
    all_metrics = [authentic_metrics] + contrasting_metrics
    scores = []
    
    for metric in all_metrics:
        score_values = list(metric.values())
        score_values[5] = 1 / (1 + score_values[5])  # Invert perplexity
        scores.append(score_values)
        
    normalized_scores = normalize_scores(np.array(scores))
    weighted_scores = normalized_scores * weights
    final_scores = weighted_scores.sum(axis=1)
    probabilities = np.exp(final_scores) / np.sum(np.exp(final_scores))
    
    return probabilities, weighted_scores, weights

def normalize_scores(scores):
    """Normalize metric scores"""
    normalized = np.array(scores, dtype=float)
    for i in range(normalized.shape[1]):
        min_val = np.min(normalized[:, i])
        max_val = np.max(normalized[:, i])
        if max_val > min_val:
            normalized[:, i] = (normalized[:, i] - min_val) / (max_val - min_val)
        else:
            normalized[:, i] = 1
    return normalized