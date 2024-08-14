from text_generation_client.py import TextGenerationClient
from watermarking import generate_and_watermark
from verification import verify_outputs

def main():
    # Initialize clients
    openai_client = TextGenerationClient(api_key="your-openai-api-key", model_name="openai")
    # Initialize other clients (Gemini, Claude) similarly

    # Define prompt
    prompt = "Write a short story about a brave knight."

    # Generate and watermark texts
    original_outputs_openai, watermarked_outputs_openai = generate_and_watermark(openai_client, prompt)
    # Generate and watermark texts for other clients similarly

    # Verify outputs
    distances_openai = verify_outputs(original_outputs_openai, watermarked_outputs_openai)
    # Verify outputs for other clients similarly

    # Output results
    print("OpenAI Distances:", distances_openai)
    # Print results for other clients similarly

if __name__ == "__main__":
    main()
