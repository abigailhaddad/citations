#!/usr/bin/env python3
"""
Test Citation Extraction Evaluation

Tests the citation extraction function with known inputs and expected outputs.
"""

import yaml
from main import extract_citations

def load_config(config_file="config.yml"):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

# Test cases with sample texts and expected citations
TEST_CASES = [
    {
        "text": "According to CNN, the conflict began in 2022. BBC reported similar findings, while Reuters confirmed the timeline.",
        "expected": ["CNN", "BBC", "Reuters"],
        "description": "Basic news outlet mentions"
    },
    {
        "text": "The New York Times published an investigation, and The Guardian provided additional context. The Wall Street Journal also covered the story.",
        "expected": ["New York Times", "The Guardian", "Wall Street Journal"],
        "description": "Newspaper citations with articles"
    },
    {
        "text": "Multiple sources suggest that recent reports indicate significant developments. Experts agree on the findings.",
        "expected": [],
        "description": "Generic references - should extract nothing"
    },
    {
        "text": "Wikipedia states that the population is growing. According to Google News aggregation, the trend continues.",
        "expected": ["Wikipedia", "Google News"],
        "description": "Web sources"
    },
    {
        "text": "The CDC released guidelines while the FDA announced new regulations. NATO sources confirmed the information.",
        "expected": ["CDC", "FDA", "NATO"],
        "description": "Government and organization acronyms"
    },
    {
        "text": "Studies show that news outlets have reported extensively on this topic.",
        "expected": [],
        "description": "Vague references without specific names"
    },
    {
        "text": "As reported by Associated Press and confirmed by Al Jazeera, the situation remains fluid. NPR also covered the story.",
        "expected": ["Associated Press", "Al Jazeera", "NPR"],
        "description": "Mixed news sources"
    },
    {
        "text": "The Harvard study, published in Nature, cited previous work from MIT researchers.",
        "expected": ["Harvard", "Nature", "MIT"],
        "description": "Academic sources"
    },
    {
        "text": "According to recent analysis, sources indicate widespread agreement among experts in the field.",
        "expected": [],
        "description": "Academic-style vague references"
    },
    {
        "text": "Fox News and MSNBC offered contrasting perspectives, while CBS News provided balanced coverage.",
        "expected": ["Fox News", "MSNBC", "CBS News"],
        "description": "TV news networks"
    }
]

def run_tests():
    """Run all test cases and report results"""
    config = load_config()
    evaluation_model = config['evaluation_model']
    
    print(f"Testing citation extraction with model: {evaluation_model}")
    print("=" * 60)
    
    total_tests = len(TEST_CASES)
    passed_tests = 0
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\nTest {i}/{total_tests}: {test_case['description']}")
        print(f"Text: {test_case['text']}")
        print(f"Expected: {test_case['expected']}")
        
        # Extract citations
        extracted = extract_citations(test_case['text'], evaluation_model)
        print(f"Extracted: {extracted}")
        
        # Check if results match
        expected_set = set(test_case['expected'])
        extracted_set = set(extracted)
        
        if expected_set == extracted_set:
            print("✅ PASS")
            passed_tests += 1
        else:
            print("❌ FAIL")
            missing = expected_set - extracted_set
            extra = extracted_set - expected_set
            if missing:
                print(f"   Missing: {list(missing)}")
            if extra:
                print(f"   Extra: {list(extra)}")
        
        print("-" * 40)
    
    print(f"\nSUMMARY:")
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    run_tests()