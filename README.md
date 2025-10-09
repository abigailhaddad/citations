# LLM Citation Extractor

A tool that uses Claude, Gemini, and ChatGPT APIs with web search tools to extract website citations from responses about geopolitical conflicts.

## Setup

1. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
```

## Configuration

Edit `config.yml` to customize:
- `prompts`: Organized by conflict type and question category (factual/breaking news, historical context, analysis/opinion)
- `temperatures`: Temperature values to test 
- `n`: Number of iterations per combination
- `top_n`: Optional - limit to first N prompts for testing (set to `null` for all prompts)

The tool includes 80 prompts across 4 conflicts:
- Israel/Gaza conflict
- Ukraine/Russia war
- China/Taiwan tensions  
- Iran women's rights protests

## Usage

Run the main script:
```bash
python main.py
```

This will:
1. Make tool-calling API requests to Claude, Gemini, and ChatGPT
2. Extract website citations from web search results
3. Save results to `llm_results.json`

For quick testing, set `top_n: 5` in `config.yml` to run only the first 5 prompts.

## Output

Results include metadata for analysis:
```json
{
  "model": "claude",
  "prompt": "What happened in Gaza today?",
  "conflict": "israel_gaza_conflict", 
  "category": "Factual/Breaking News",
  "subcategory": "factual_breaking_news",
  "temperature": 0.0,
  "iteration": 1,
  "citations": ["https://example.com/..."]
}
```