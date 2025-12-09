#!/usr/bin/env python3
"""
LLM Citation Extractor using direct tool calling APIs

Uses Claude, Gemini, and ChatGPT APIs directly
YAML configuration for prompts, temperatures, and iterations.
"""

import os
import json
import yaml
import itertools
import pandas as pd
import requests
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
import anthropic
import openai
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

def resolve_redirect_url(url: str) -> str:
    """Resolve Google redirect URLs to their final destination"""
    if not url.startswith('https://vertexaisearch.cloud.google.com/grounding-api-redirect/'):
        return url
    
    print(f"Resolving redirect: {url[:80]}...")
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            final_url = url
            
            # Capture response events to track redirects - but only for main document navigation
            def handle_response(response):
                nonlocal final_url
                # Only process main document requests, not resources like images/js/css
                if response.request.resource_type == 'document':
                    # If this is a redirect, capture the Location header
                    if response.status in [301, 302, 303, 307, 308]:
                        location = response.headers.get('location')
                        if location:
                            # Make absolute URL if needed
                            if location.startswith('http'):
                                final_url = location
                            else:
                                from urllib.parse import urljoin
                                final_url = urljoin(response.url, location)
                    elif response.url != url:
                        # Final destination
                        final_url = response.url
            
            page.on('response', handle_response)
            
            # Block all resource loading except main document to speed up and avoid noise
            page.route("**/*", lambda route: route.abort() if route.request.resource_type != 'document' else route.continue_())
            
            # Navigate but abort as soon as we get the redirect
            try:
                page.goto(url, timeout=15000)
            except Exception:
                # Even if navigation fails, we might have captured the redirect
                pass
            
            browser.close()
            
            if final_url != url:
                print(f"✓ Resolved to: {final_url}")
                return final_url
            else:
                print(f"✗ No redirect found")
                return url
                
    except Exception as e:
        print(f"✗ Playwright error: {e}")
        return url

def load_config(config_file="config.yml"):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def search_claude(query: str) -> list[str]:
    """Query Claude with web search tool and extract citations"""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": query
            }
        ],
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5
        }]
    )

    # Extract citations from the response
    citations = []
    for block in message.content:
        if block.type == "text" and hasattr(block, 'citations') and block.citations:
            for citation in block.citations:
                citations.append(citation.url)

    return list(set(citations))  # Remove duplicates

def search_gemini(query: str) -> list[str]:
    """Query Google Gemini with Google Search grounding and extract website citations"""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=config
    )

    # Extract citations from grounding metadata
    citations = []
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            if hasattr(candidate.grounding_metadata, 'grounding_chunks'):
                for chunk in candidate.grounding_metadata.grounding_chunks:
                    if hasattr(chunk, 'web') and chunk.web and hasattr(chunk.web, 'uri'):
                        # Resolve Google redirect URLs to actual destinations
                        resolved_url = resolve_redirect_url(chunk.web.uri)
                        citations.append(resolved_url)

    return list(set(citations))  # Remove duplicates

def search_chatgpt(query: str) -> list[str]:
    """Query ChatGPT with web search enabled and extract website citations"""
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.responses.create(
        model="gpt-4o",
        tools=[{"type": "web_search"}],
        tool_choice={"type": "web_search"},
        input=query
    )

    # Extract citations from the response annotations
    citations = []
    if hasattr(response, 'output') and response.output:
        for output_item in response.output:
            if hasattr(output_item, 'content') and output_item.content:
                for content_item in output_item.content:
                    if hasattr(content_item, 'annotations') and content_item.annotations:
                        for annotation in content_item.annotations:
                            if annotation.type == "url_citation" and hasattr(annotation, 'url'):
                                citations.append(annotation.url)

    return list(set(citations))  # Remove duplicates

def main():
    """Main execution function"""
    # Ensure API keys are available
    required_keys = ["ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"]
    for key in required_keys:
        if key not in os.environ:
            print(f"Warning: {key} not found in environment variables")

    # Load configuration
    config = load_config()
    
    # Flatten prompts from nested dict structure
    prompts = []
    for conflict_name, conflict_data in config['prompts'].items():
        for category_name, category_prompts in conflict_data.items():
            for prompt_item in category_prompts:
                prompts.append({
                    'prompt': prompt_item['prompt'],
                    'conflict': conflict_name,
                    'category': prompt_item['category'],
                    'subcategory': category_name
                })
    
    temperatures = config['temperatures'] 
    n = config['n']
    top_n = config.get('top_n')  # Optional parameter
    
    # Limit prompts if top_n is specified
    if top_n is not None:
        prompts = prompts[:top_n]
        print(f"Limited to first {top_n} prompts for testing")
    
    # Define the models to test
    models = [
        ("claude", search_claude),
        ("gemini", search_gemini), 
        ("chatgpt", search_chatgpt)
    ]
    
    total_calls = len(models) * len(prompts) * len(temperatures) * n
    
    print(f"Starting tool calling LLM calls:")
    print(f"  {len(models)} models, {len(prompts)} prompts, {len(temperatures)} temperatures")
    print(f"  Running {n} iterations per combination = {total_calls} total calls")
    print()
    
    results = []
    
    # Run all combinations
    for (model_name, search_func), prompt_data, temperature in itertools.product(models, prompts, temperatures):
        prompt_text = prompt_data['prompt']
        print(f"Running {n} iterations for {model_name} at temp {temperature}")
        print(f"Prompt: {prompt_text[:50]}...")
        
        for iteration in range(n):
            print(f"  Iteration {iteration + 1}/{n}", end=" ")
            
            try:
                # Make the tool calling LLM call (temperature not used for tool calling)
                citations = search_func(prompt_text)
                
                result = {
                    'model': model_name,
                    'prompt': prompt_text,
                    'conflict': prompt_data['conflict'],
                    'category': prompt_data['category'],
                    'subcategory': prompt_data['subcategory'],
                    'temperature': temperature,
                    'iteration': iteration + 1,
                    'response': None,  # Tool calling doesn't return full response text
                    'citations': citations,
                    'tool_calling': True
                }
                
                results.append(result)
                print(f"✓ Found {len(citations)} citations")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
    
    # Save results to JSON file
    output_file = "llm_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create CSV in long format with one row per citation
    csv_rows = []
    for result in results:
        for citation in result['citations']:
            csv_rows.append({
                'model': result['model'],
                'prompt': result['prompt'],
                'conflict': result['conflict'],
                'category': result['category'],
                'subcategory': result['subcategory'],
                'temperature': result['temperature'],
                'iteration': result['iteration'],
                'citation': citation
            })
    
    # Save to CSV using pandas
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        csv_file = "llm_citations.csv"
        df.to_csv(csv_file, index=False)
        print(f"CSV saved to {csv_file}")
    
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