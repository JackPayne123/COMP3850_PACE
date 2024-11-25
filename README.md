# AI Text Verification Tool

A sophisticated tool for verifying the authenticity of AI-generated text and detecting which AI model likely generated a piece of text. The tool uses multiple metrics and comparison techniques to provide accurate authorship attribution.

This is an implementation of the paper "Generative Models are Self-Watermarked: Declaring Model Authentication through Re-Generation" https://arxiv.org/pdf/2402.16889

## Features

- **Multi-Model Support**: Verify text against multiple AI models including OpenAI GPT-4, Anthropic Claude, Google Gemini, and Mistral
- **Human Detection**: Optional capability to detect human-written text
- **Multiple Input Methods**: 
  - Manual text entry
  - Direct text generation using supported models
- **Comprehensive Analysis**:
  - BERTScore
  - Cosine Similarity
  - ROUGE-L Score
  - BLEU Score
  - METEOR Score
  - Perplexity Measurements
- **Interactive Visualisations**: View detailed metrics and probability distributions

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JackPayne123/COMP3850_PACE.git
   cd COMP3850_PACE
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required NLTK data:
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```

5. Download Spacy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Configuration

1. Create a `.streamlit` directory in the project root:
   ```bash
   mkdir .streamlit
   ```

2. Create a `secrets.toml` file in the `.streamlit` directory:
   ```bash
   touch .streamlit/secrets.toml
   ```

3. Add your API keys to `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "your-openai-key"
   ANTHROPIC_API_KEY = "your-anthropic-key"
   GEMINI_API_KEY = "your-gemini-key"
   MISTRAL_API_KEY = "your-mistral-key"
   ```

## Running the Application

1. Ensure your virtual environment is activated
2. Run the Streamlit application:
   ```bash
   streamlit run src/app.py
   ```
3. Open your browser and navigate to `http://localhost:8501`

## Usage

1. **Select Input Method**:
   - Choose between manual text entry or AI-generated text
   - For AI generation, select a model and enter a prompt

2. **Configure Options**:
   - Choose regeneration method (Summarize or Paraphrase)
   - Enable/disable human detection
   - Select the model to verify against

3. **Run Verification**:
   - Click "Run Verification" to start the analysis
   - View detailed results including:
     - Authorship probability
     - Metric scores
     - Verification iterations
     - Visual analysis

## Project Structure

```
project_root/
├── src/
│   ├── app.py                 # Main application file
│   ├── models/
│   │   └── model_loader.py    # Model loading utilities
│   ├── utils/
│   │   ├── metrics.py         # Metric calculations
│   │   ├── text_analysis.py   # Text analysis functions
│   │   └── verification.py    # Verification logic
│   └── config/
│       └── settings.py        # Configuration settings
├── requirements.txt           # Project dependencies
└── README.md                 # Project documentation
```
