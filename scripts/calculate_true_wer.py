#!/usr/bin/env python3
"""
Calculate true Word Error Rate (WER) ignoring punctuation and capitalization.
This reflects actual word recognition errors, not formatting differences.
"""

import re
import json
from pathlib import Path

def normalize_text_for_wer(text):
    """
    Normalize text for WER calculation by:
    - Converting to lowercase
    - Removing punctuation
    - Normalizing whitespace
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_word_error_rate(reference, hypothesis):
    """
    Calculate WER using Levenshtein distance on word level.
    Only counts actual word differences, ignoring punctuation/capitalization.
    """
    # Normalize both texts
    ref_words = normalize_text_for_wer(reference).split()
    hyp_words = normalize_text_for_wer(hypothesis).split()
    
    # Calculate Levenshtein distance
    m, n = len(ref_words), len(hyp_words)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    
    # Calculate WER
    errors = dp[m][n]
    wer = errors / m if m > 0 else 0
    
    return wer, errors, m

def analyze_evaluation_results():
    """Analyze the evaluation results with proper WER calculation."""
    results_file = Path("/Users/user/Desktop/WhisperX/results/evaluation_results_20250613_190417.json")
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("=== TRUE WORD ERROR RATE ANALYSIS ===\\n")
    
    total_errors = 0
    total_words = 0
    
    for result in data['asr_results']:
        sample_id = result['sample_id']
        ground_truth = result['ground_truth']
        predicted = result['predicted']
        original_wer = result['wer']
        
        # Calculate true WER
        true_wer, errors, word_count = calculate_word_error_rate(ground_truth, predicted)
        
        print(f"Sample {sample_id}:")
        print(f"  Ground truth: {ground_truth}")
        print(f"  Predicted:    {predicted}")
        print(f"  Original WER: {original_wer:.1%}")
        print(f"  True WER:     {true_wer:.1%} ({errors} errors in {word_count} words)")
        
        # Show normalized versions for comparison
        ref_normalized = normalize_text_for_wer(ground_truth)
        hyp_normalized = normalize_text_for_wer(predicted)
        print(f"  Normalized reference: {ref_normalized}")
        print(f"  Normalized hypothesis: {hyp_normalized}")
        print()
        
        total_errors += errors
        total_words += word_count
    
    # Calculate overall true WER
    overall_true_wer = total_errors / total_words if total_words > 0 else 0
    original_avg_wer = data['asr_summary']['average_wer']
    
    print(f"=== SUMMARY ===")
    print(f"Original Average WER: {original_avg_wer:.1%}")
    print(f"True WER (words only): {overall_true_wer:.1%}")
    print(f"Total word errors: {total_errors}")
    print(f"Total words: {total_words}")
    print(f"True ASR Accuracy: {(1 - overall_true_wer):.1%}")
    
    return overall_true_wer

if __name__ == "__main__":
    analyze_evaluation_results()