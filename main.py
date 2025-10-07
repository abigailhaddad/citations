#!/usr/bin/env python3
"""
LLM News Response Citation Extractor

Makes LLM calls with various parameters and extracts news source citations
from responses using structured output mode.
"""

import os
import json
import yaml
import itertools
from typing import List
from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class CitationResult(BaseModel):
    """Model for citation extraction results"""
    citations: List[str] = Field(
        description="List of news sources or citations mentioned in the response (e.g., Wikipedia, CBS, Reuters, etc.)"
    )

def load_config(config_file="config.yml"):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def make_llm_call(model, prompt, temperature):
    """Make a standard LLM call without structured output"""
    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error making LLM call with {model}: {e}")
        return None

def extract_citations(response_text, evaluation_model):
    """Extract citations from response using structured output"""
    try:
        system_prompt = """You are a citation extractor. Analyze the given text and identify ONLY the specific names of news outlets, publications, and information sources that are explicitly mentioned IN THE PROVIDED TEXT.

Examples of what to extract (only if they appear in the text):
- CNN, BBC, Reuters, Associated Press, New York Times
- Wikipedia, Google News, Yahoo News
- Wall Street Journal, Washington Post, The Guardian
- Specific government agency names (FDA, CDC, etc.)
- Specific academic institution names

DO NOT extract:
- Generic descriptions like "news reports", "media outlets", "sources"
- Phrases like "according to experts" or "studies show"
- Vague references like "recent reports" or "multiple sources"
- Names that are not actually contained in the provided text

CRITICAL: Return ONLY the proper names that are explicitly written in the text you are analyzing. Do not infer or add sources that are not directly mentioned."""

        response = completion(
            model=evaluation_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract all citations and news sources from this text:\n\n{response_text}"}
            ],
            response_format=CitationResult,
            temperature=0.0
        )
        
        # Handle different response formats
        if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content:
            if isinstance(response.choices[0].message.content, str):
                result = json.loads(response.choices[0].message.content)
            else:
                result = response.choices[0].message.content
        else:
            result = response.choices[0].message.model_dump()
            
        return result.get('citations', [])
        
    except Exception as e:
        print(f"Error extracting citations: {e}")
        return []

def main():
    """Main execution function"""
    # Ensure API keys are available
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables or .env file")
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables or .env file")
    
    # Load configuration
    config = load_config()
    
    models = config['models']
    prompts = config['prompts']
    temperatures = config['temperatures']
    n = config['n']
    evaluation_model = config['evaluation_model']
    
    print(f"Starting LLM calls with {len(models)} models, {len(prompts)} prompts, {len(temperatures)} temperatures")
    print(f"Running {n} iterations per combination = {len(models) * len(prompts) * len(temperatures) * n} total calls")
    print(f"Using evaluation model: {evaluation_model}")
    print()
    
    results = []
    
    # Generate all combinations
    for model, prompt, temperature in itertools.product(models, prompts, temperatures):
        print(f"Running {n} iterations for {model} at temp {temperature}")
        print(f"Prompt: {prompt[:50]}...")
        
        for iteration in range(n):
            print(f"  Iteration {iteration + 1}/{n}", end=" ")
            
            # Make the main LLM call
            response = make_llm_call(model, prompt, temperature)
            
            if response:
                # Extract citations using evaluation model
                citations = extract_citations(response, evaluation_model)
                
                result = {
                    'model': model,
                    'prompt': prompt,
                    'temperature': temperature,
                    'iteration': iteration + 1,
                    'response': response,
                    'citations': citations
                }
                
                results.append(result)
                print(f"✓ Found {len(citations)} citations")
            else:
                print("✗ Failed")
    
    # Save results to JSON file
    output_file = "llm_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCompleted! Results saved to {output_file}")
    print(f"Total successful calls: {len(results)}")
    
    # Print summary statistics
    total_citations = sum(len(r['citations']) for r in results)
    print(f"Total citations extracted: {total_citations}")
    
    if results:
        avg_citations = total_citations / len(results)
        print(f"Average citations per response: {avg_citations:.2f}")

if __name__ == "__main__":
    main()