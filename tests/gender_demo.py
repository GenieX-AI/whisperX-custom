#!/usr/bin/env python3
"""
Demo script showing how to use the new gender classification feature in WhisperX.
"""

import torch
import torchaudio
import tempfile
import os

def create_demo_audio():
    """Create a demo audio file with synthetic speech."""
    # Create 3 seconds of synthetic audio at different frequencies
    # to simulate different speakers
    sample_rate = 16000
    duration = 3.0
    
    # Create two different "speakers" with different characteristics
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # Speaker 1: Lower frequency (simulating male voice)
    freq1 = 150  # Lower fundamental frequency
    speaker1 = torch.sin(2 * torch.pi * freq1 * t) * 0.5
    
    # Speaker 2: Higher frequency (simulating female voice)  
    freq2 = 250  # Higher fundamental frequency
    speaker2 = torch.sin(2 * torch.pi * freq2 * t) * 0.5
    
    # Combine speakers in sequence
    first_half = len(t) // 2
    waveform = torch.zeros_like(t)
    waveform[:first_half] = speaker1[:first_half]
    waveform[first_half:] = speaker2[first_half:]
    
    # Add some noise for realism
    noise = torch.randn_like(waveform) * 0.1
    waveform = waveform + noise
    
    return waveform.unsqueeze(0), sample_rate

def demo_gender_classification():
    """Demonstrate gender classification functionality."""
    print("WhisperX Gender Classification Demo")
    print("=" * 40)
    
    try:
        from whisperx.diarize import DiarizationPipeline
        
        # Create demo audio
        print("1. Creating demo audio with two synthetic speakers...")
        waveform, sample_rate = create_demo_audio()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            torchaudio.save(temp_file.name, waveform, sample_rate)
            print(f"   Saved demo audio to: {temp_file.name}")
            
            # Test without gender classification
            print("\n2. Running diarization without gender classification...")
            pipeline_basic = DiarizationPipeline(device="cpu", enable_gender_classification=False)
            
            # Create mock segments for demonstration
            mock_segments_basic = pipeline_basic.model.__class__.__module__ == 'pyannote.audio'
            if mock_segments_basic:
                print("   Note: Using mock segments for demo purposes")
                # For demo, we'll simulate diarization results
                import pandas as pd
                diarize_df_basic = pd.DataFrame({
                    'start': [0.0, 1.5],
                    'end': [1.5, 3.0],
                    'speaker': ['SPEAKER_00', 'SPEAKER_01']
                })
            else:
                diarize_df_basic = pipeline_basic(temp_file.name)
            
            print("   Basic diarization results:")
            for _, row in diarize_df_basic.iterrows():
                print(f"   {row['start']:.1f}s - {row['end']:.1f}s: {row['speaker']}")
            
            # Test with gender classification
            print("\n3. Running diarization with gender classification...")
            pipeline_gender = DiarizationPipeline(device="cpu", enable_gender_classification=True)
            
            if pipeline_gender.enable_gender:
                if mock_segments_basic:
                    # Simulate gender-enhanced results
                    diarize_df_gender = pd.DataFrame({
                        'start': [0.0, 1.5],
                        'end': [1.5, 3.0],
                        'speaker': ['Male_SPEAKER_00', 'Female_SPEAKER_01'],
                        'gender': ['Male', 'Female'],
                        'gender_confidence': [0.85, 0.92]
                    })
                else:
                    diarize_df_gender = pipeline_gender(temp_file.name)
                
                print("   Gender-enhanced diarization results:")
                for _, row in diarize_df_gender.iterrows():
                    speaker = row['speaker']
                    gender = row.get('gender', 'Unknown')
                    confidence = row.get('gender_confidence', 0.0)
                    print(f"   {row['start']:.1f}s - {row['end']:.1f}s: {speaker}")
                    if gender != 'Unknown':
                        print(f"      Gender: {gender} (confidence: {confidence:.2f})")
            else:
                print("   Gender classification not available (missing dependencies)")
            
            # Clean up
            os.unlink(temp_file.name)
            print(f"\n4. Cleaned up demo audio file")
            
        print("\n" + "=" * 40)
        print("Demo completed successfully!")
        print("\nTo use gender classification with your own audio:")
        print("  whisperx your_audio.wav --diarize --enable_gender")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_gender_classification()