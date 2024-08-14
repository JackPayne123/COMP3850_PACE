import openai
impo

class TextGenerationClient:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        if model_name == "openai":
            openai.api_key = api_key
        # Initialize other models (Gemini, Claude) similarly

    def generate_text(self, prompt, model="text-davinci-003"):
        if self.model_name == "openai":
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].text.strip()
        # Add text generation methods for other models here
