#!/usr/bin/env python3
"""
Debug script to analyze feature scales in gender classification.
This will help identify exactly where the massive feature values are coming from.
"""

import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Add WhisperX to path
sys.path.insert(0, '/Users/user/Desktop/WhisperX')

from whisperx.gender_classifier import EnhancedGenderClassifier

def analyze_feature_scales():
    """Analyze the scale of features being generated."""
    print("=== DEBUGGING FEATURE SCALE ISSUE ===\n")
    
    # Initialize classifier
    print("1. Initializing gender classifier...")
    classifier = EnhancedGenderClassifier(device="cpu")
    
    # Create sample audio data similar to what we'd get from diarization
    print("2. Creating sample audio data...")
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    
    # Generate realistic audio signal (not just white noise)
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Mix of frequencies similar to human speech
    audio_signal = (
        0.3 * np.sin(2 * np.pi * 150 * t) +  # Fundamental around 150Hz (male voice)
        0.2 * np.sin(2 * np.pi * 300 * t) +  # First harmonic
        0.1 * np.sin(2 * np.pi * 450 * t) +  # Second harmonic
        0.05 * np.random.randn(len(t))       # Some noise
    )
    
    audio_tensor = torch.tensor(audio_signal, dtype=torch.float32)
    
    print(f"   Audio shape: {audio_tensor.shape}")
    print(f"   Audio range: [{audio_tensor.min():.4f}, {audio_tensor.max():.4f}]")
    print(f"   Audio mean: {audio_tensor.mean():.4f}, std: {audio_tensor.std():.4f}")
    
    # Extract features and analyze each component
    print("\n3. Analyzing feature extraction components...")
    
    # Convert to numpy for analysis
    audio_np = audio_tensor.numpy()
    
    print("\n--- Individual Feature Analysis ---")
    
    # 1. F0 estimation
    f0_estimate = classifier._estimate_f0(audio_np)
    f0_features = [f0_estimate, f0_estimate**2, np.log(f0_estimate + 1e-8)]
    print(f"F0 features: {f0_features}")
    print(f"F0 range: [{min(f0_features):.2f}, {max(f0_features):.2f}]")
    
    # 2. Spectral centroid
    spectral_centroid = classifier._compute_spectral_centroid(audio_np)
    centroid_features = [spectral_centroid, spectral_centroid**2]
    print(f"Spectral centroid features: {centroid_features}")
    print(f"Centroid range: [{min(centroid_features):.2f}, {max(centroid_features):.2f}]")
    
    # 3. Energy features
    energy_features = classifier._compute_energy_features(audio_np)
    print(f"Energy features: {energy_features}")
    print(f"Energy range: [{min(energy_features):.6f}, {max(energy_features):.6f}]")
    
    # 4. Spectral rolloff
    rolloff = classifier._compute_spectral_rolloff(audio_np)
    rolloff_features = [rolloff, rolloff**2]
    print(f"Spectral rolloff features: {rolloff_features}")
    print(f"Rolloff range: [{min(rolloff_features):.2f}, {max(rolloff_features):.2f}]")
    
    # 5. Zero crossing rate
    zcr = classifier._compute_zero_crossing_rate(audio_np)
    zcr_features = [zcr, zcr**2]
    print(f"ZCR features: {zcr_features}")
    print(f"ZCR range: [{min(zcr_features):.6f}, {max(zcr_features):.6f}]")
    
    # 6. Mel features
    mel_features = classifier._compute_mel_features(audio_np)
    print(f"Mel features (first 5): {mel_features[:5]}")
    print(f"Mel range: [{min(mel_features):.2f}, {max(mel_features):.2f}]")
    
    # Extract full feature vector
    print("\n--- Full Feature Vector Analysis ---")
    features = classifier._extract_audio_features(audio_tensor)
    
    if features is not None:
        print(f"Feature vector shape: {features.shape}")
        print(f"Feature range: [{features.min():.2f}, {features.max():.2f}]")
        print(f"Feature mean: {features.mean():.4f}, std: {features.std():.4f}")
        
        # Identify problematic features
        large_features = np.where(np.abs(features) > 1000)[0]
        if len(large_features) > 0:
            print(f"\n⚠️  FOUND {len(large_features)} FEATURES WITH MAGNITUDE > 1000:")
            for idx in large_features[:10]:  # Show first 10
                print(f"   Feature[{idx}]: {features[idx]:.2f}")
        
        huge_features = np.where(np.abs(features) > 10000)[0]
        if len(huge_features) > 0:
            print(f"\n❌ FOUND {len(huge_features)} FEATURES WITH MAGNITUDE > 10,000:")
            for idx in huge_features:
                print(f"   Feature[{idx}]: {features[idx]:.2f}")
    
    # Test neural network forward pass
    print("\n4. Testing neural network forward pass...")
    
    if features is not None and classifier.model is not None:
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Check intermediate activations
        with torch.no_grad():
            print(f"Input features range: [{features_tensor.min():.2f}, {features_tensor.max():.2f}]")
            
            # Get the bias values we set
            final_layer_bias = classifier.model.layers[6].bias.data
            print(f"Final layer bias: {final_layer_bias}")
            
            # Forward pass through each layer
            x = features_tensor
            for i, layer in enumerate(classifier.model.layers):
                if isinstance(layer, torch.nn.Linear):
                    x_before = x.clone()
                    x = layer(x)
                    print(f"After layer {i} ({type(layer).__name__}): range [{x.min():.4f}, {x.max():.4f}], mean {x.mean():.4f}")
                    
                    # Check if this linear layer is causing the problem
                    if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
                        linear_output = torch.matmul(x_before, layer.weight.t()) + layer.bias
                        matmul_part = torch.matmul(x_before, layer.weight.t())
                        print(f"   Matrix multiply part: range [{matmul_part.min():.4f}, {matmul_part.max():.4f}]")
                        print(f"   Bias part: {layer.bias}")
                        print(f"   Bias magnitude vs matmul: bias_max={layer.bias.abs().max():.4f}, matmul_max={matmul_part.abs().max():.4f}")
                        
                elif isinstance(layer, torch.nn.ReLU):
                    x = layer(x)
                    print(f"After layer {i} ({type(layer).__name__}): range [{x.min():.4f}, {x.max():.4f}]")
                elif isinstance(layer, torch.nn.Dropout):
                    x = layer(x)  # In eval mode, dropout does nothing
                elif isinstance(layer, torch.nn.Softmax):
                    x = layer(x)
                    print(f"After layer {i} ({type(layer).__name__}): range [{x.min():.4f}, {x.max():.4f}]")
                    print(f"Final probabilities: {x.squeeze()}")
    
    print("\n=== ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    analyze_feature_scales()