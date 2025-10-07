# LLM Citation Extractor

A tool that makes LLM calls with different parameters and extracts news source citations from the responses.

## Setup

1. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install litellm pyyaml python-dotenv
```

2. Add your API keys to `.env`:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## Configuration

Edit `config.yml` to set:
- `models`: List of models to test in `provider/model` format (e.g., `openai/gpt-4o-mini`, `anthropic/claude-3-haiku-20240307`)
- `prompts`: Questions to ask the models
- `temperatures`: Temperature values to test
- `n`: Number of iterations per combination
- `evaluation_model`: Model used to extract citations

LiteLLM supports many providers and models. See the full list at https://docs.litellm.ai/docs/providers. Make sure to add the appropriate API key to your `.env` file for each provider you use.

## Usage

Run the main script:
```bash
python main.py
```

This will:
1. Make LLM calls for each model/prompt/temperature/iteration combination
2. Extract citations from each response using structured output
3. Save results to `llm_results.json`

## Testing

Test the citation extraction function:
```bash
python test_evaluation.py
```

## Output

Results are saved in `llm_results.json` with this structure:
```json
{
  "model": "openai/gpt-4o-mini",
  "prompt": "What are the top 3 news stories today?",
  "temperature": 0.0,
  "iteration": 1,
  "response": "...",
  "citations": ["CNN", "BBC", "Reuters"]
}
```

The citation extractor looks for specific outlet names mentioned in the text, not generic references like "news reports" or "sources".