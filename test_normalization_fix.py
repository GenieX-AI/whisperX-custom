#!/usr/bin/env python3
"""
Test script to verify that feature normalization fixes the gender classification issue.
"""

import sys
import torch
import numpy as np

# Add WhisperX to path
sys.path.insert(0, '/Users/user/Desktop/WhisperX')

from whisperx.gender_classifier import EnhancedGenderClassifier

def test_normalization_fix():
    """Test the normalization fix for feature scaling."""
    print("=== TESTING NORMALIZATION FIX ===\n")
    
    # Initialize classifier
    classifier = EnhancedGenderClassifier(device="cpu")
    
    # Create sample audio (same as before)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_signal = (
        0.3 * np.sin(2 * np.pi * 150 * t) +
        0.2 * np.sin(2 * np.pi * 300 * t) +
        0.1 * np.sin(2 * np.pi * 450 * t) +
        0.05 * np.random.randn(len(t))
    )
    audio_tensor = torch.tensor(audio_signal, dtype=torch.float32)
    
    print("1. BEFORE normalization:")
    # Get original features
    features_original = classifier._extract_audio_features(audio_tensor)
    print(f"   Feature range: [{features_original.min():.2f}, {features_original.max():.2f}]")
    print(f"   Feature mean: {features_original.mean():.4f}, std: {features_original.std():.4f}")
    
    # Test classification with original features
    gender_orig, confidence_orig = classifier.classify_gender(audio_tensor)
    print(f"   Classification: {gender_orig} (confidence: {confidence_orig:.4f})")
    
    print("\n2. AFTER normalization:")
    # Normalize features (z-score normalization)
    features_normalized = (features_original - features_original.mean()) / (features_original.std() + 1e-8)
    print(f"   Normalized feature range: [{features_normalized.min():.4f}, {features_normalized.max():.4f}]")
    print(f"   Normalized feature mean: {features_normalized.mean():.4f}, std: {features_normalized.std():.4f}")
    
    # Test classification with normalized features
    with torch.no_grad():
        features_tensor = torch.tensor(features_normalized, dtype=torch.float32).unsqueeze(0)
        outputs = classifier.model(features_tensor)
        probabilities = outputs.squeeze().cpu().numpy()
        
        male_prob = probabilities[0]
        female_prob = probabilities[1]
        
        if male_prob > female_prob:
            gender_norm = "Male"
            confidence_norm = float(male_prob)
        else:
            gender_norm = "Female"
            confidence_norm = float(female_prob)
            
        print(f"   Classification: {gender_norm} (confidence: {confidence_norm:.4f})")
        print(f"   Raw probabilities: Male={male_prob:.4f}, Female={female_prob:.4f}")
    
    print("\n3. ANALYZING THE BIAS EFFECT:")
    # Test with normalized features through neural network layers
    with torch.no_grad():
        features_tensor = torch.tensor(features_normalized, dtype=torch.float32).unsqueeze(0)
        
        x = features_tensor
        print(f"   Input: range [{x.min():.4f}, {x.max():.4f}]")
        
        # First linear layer
        x = classifier.model.layers[0](x)  # Linear
        print(f"   After 1st linear: range [{x.min():.4f}, {x.max():.4f}]")
        
        x = classifier.model.layers[1](x)  # ReLU
        print(f"   After ReLU: range [{x.min():.4f}, {x.max():.4f}]")
        
        x = classifier.model.layers[2](x)  # Dropout (no effect in eval)
        
        # Second linear layer  
        x = classifier.model.layers[3](x)  # Linear
        print(f"   After 2nd linear: range [{x.min():.4f}, {x.max():.4f}]")
        
        x = classifier.model.layers[4](x)  # ReLU
        print(f"   After ReLU: range [{x.min():.4f}, {x.max():.4f}]")
        
        x = classifier.model.layers[5](x)  # Dropout (no effect in eval)
        
        # Final linear layer (before softmax)
        pre_softmax = classifier.model.layers[6](x)  # Linear with bias [2.0, -2.0]
        print(f"   Before softmax: {pre_softmax.squeeze()}")
        
        # Check the bias effect
        final_layer = classifier.model.layers[6]
        matmul_part = torch.matmul(x, final_layer.weight.t())
        bias_part = final_layer.bias
        print(f"   Matrix multiply part: {matmul_part.squeeze()}")
        print(f"   Bias part: {bias_part}")
        print(f"   Total (matmul + bias): {(matmul_part + bias_part).squeeze()}")
        
        # Apply softmax
        final_output = classifier.model.layers[7](pre_softmax)  # Softmax
        print(f"   After softmax: {final_output.squeeze()}")
    
    print("\n4. COMPARISON:")
    print(f"   Original: {gender_orig} ({confidence_orig:.4f})")
    print(f"   Normalized: {gender_norm} ({confidence_norm:.4f})")
    
    if gender_orig != gender_norm:
        print("   ✅ NORMALIZATION CHANGED THE CLASSIFICATION!")
    else:
        print("   ❌ Normalization didn't change classification")
        
    if abs(confidence_norm - 0.5) < abs(confidence_orig - 0.5):
        print("   ✅ NORMALIZATION MADE CONFIDENCE MORE REASONABLE!")
    else:
        print("   ❌ Confidence still extreme")

if __name__ == "__main__":
    test_normalization_fix()