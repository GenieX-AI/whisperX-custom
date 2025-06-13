#!/usr/bin/env python3
"""
Download real audio samples with transcripts for testing
"""

import os
import requests
import soundfile as sf
from datasets import load_dataset
import json

def download_real_test_samples():
    """Download 2-3 real audio samples with transcripts"""
    print("📥 Downloading Real Audio Test Samples")
    print("=" * 60)
    
    test_data_dir = "/Users/user/Desktop/WhisperX/test_data"
    results_dir = "/Users/user/Desktop/WhisperX/results"
    
    # Ensure directories exist
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    samples_downloaded = []
    
    try:
        # Method 1: Use the demo dataset that loads quickly
        print("📂 Loading LibriSpeech demo samples...")
        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
        
        # Limit to first 3 samples
        max_samples = min(3, len(dataset))
        dataset = dataset.select(range(max_samples))
        
        for i, sample in enumerate(dataset):
            try:
                # Extract data
                audio_array = sample["audio"]["array"].astype('float32')
                sample_rate = sample["audio"]["sampling_rate"]
                transcript = sample["text"]
                speaker_id = sample.get("speaker_id", f"demo_{i}")
                chapter_id = sample.get("chapter_id", f"demo_ch_{i}")
                
                # Create filenames
                audio_filename = f"sample_{i+1}_speaker_{speaker_id}.wav"
                transcript_filename = f"sample_{i+1}_speaker_{speaker_id}.txt"
                
                audio_path = os.path.join(test_data_dir, audio_filename)
                transcript_path = os.path.join(test_data_dir, transcript_filename)
                
                # Save audio file
                sf.write(audio_path, audio_array, sample_rate)
                
                # Save transcript
                with open(transcript_path, 'w') as f:
                    f.write(transcript)
                
                # Create metadata
                metadata = {
                    "audio_file": audio_filename,
                    "transcript_file": transcript_filename,
                    "transcript": transcript,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "duration_seconds": len(audio_array) / sample_rate,
                    "sample_rate": int(sample_rate),
                    "source": "LibriSpeech_Demo"
                }
                
                samples_downloaded.append(metadata)
                
                print(f"✅ Sample {i+1} downloaded:")
                print(f"   Audio: {audio_filename} ({len(audio_array)/sample_rate:.1f}s)")
                print(f"   Text: {transcript[:60]}...")
                print(f"   Speaker: {speaker_id}")
                
            except Exception as e:
                print(f"❌ Failed to download sample {i+1}: {e}")
                continue
        
    except Exception as e:
        print(f"❌ LibriSpeech demo failed: {e}")
        
        # Fallback: Create synthetic samples
        try:
            print("\n🔄 Creating synthetic samples as fallback...")
            import numpy as np
            
            synthetic_data = [
                {"text": "This is a test of the WhisperX speech recognition system.", "freq": 440},
                {"text": "Gender classification and speaker diarization testing sample.", "freq": 330}
            ]
            
            for i, sample_info in enumerate(synthetic_data):
                try:
                    # Generate synthetic audio
                    sample_rate = 16000
                    duration = 3.0
                    t = np.linspace(0, duration, int(sample_rate * duration))
                    
                    freq = sample_info["freq"]
                    audio_array = (np.sin(2 * np.pi * freq * t) * 0.3 +
                                  np.sin(2 * np.pi * freq * 2 * t) * 0.1).astype('float32')
                    
                    transcript = sample_info["text"]
                    
                    audio_filename = f"synthetic_sample_{i+1}.wav"
                    transcript_filename = f"synthetic_sample_{i+1}.txt"
                    
                    audio_path = os.path.join(test_data_dir, audio_filename)
                    transcript_path = os.path.join(test_data_dir, transcript_filename)
                    
                    sf.write(audio_path, audio_array, sample_rate)
                    
                    with open(transcript_path, 'w') as f:
                        f.write(transcript)
                    
                    metadata = {
                        "audio_file": audio_filename,
                        "transcript_file": transcript_filename,
                        "transcript": transcript,
                        "duration_seconds": duration,
                        "sample_rate": sample_rate,
                        "source": "Synthetic"
                    }
                    
                    samples_downloaded.append(metadata)
                    print(f"✅ Synthetic sample {i+1} created")
                    
                except Exception as e:
                    print(f"❌ Failed synthetic sample {i+1}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Synthetic fallback failed: {e}")
    
    # Save metadata file
    if samples_downloaded:
        metadata_path = os.path.join(test_data_dir, "test_samples_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(samples_downloaded, f, indent=2)
        
        print(f"\n✅ Downloaded {len(samples_downloaded)} samples")
        print(f"📁 Test data saved to: {test_data_dir}")
        print(f"📄 Metadata saved to: {metadata_path}")
        
        # Create results structure
        results_structure = {
            "test_run_info": {
                "total_samples": len(samples_downloaded),
                "test_data_dir": test_data_dir,
                "results_dir": results_dir
            },
            "samples": samples_downloaded
        }
        
        results_metadata_path = os.path.join(results_dir, "test_setup.json")
        with open(results_metadata_path, 'w') as f:
            json.dump(results_structure, f, indent=2)
            
        print(f"📊 Results structure created: {results_metadata_path}")
        
        return len(samples_downloaded)
    else:
        print("❌ No samples downloaded successfully")
        return 0

def list_downloaded_samples():
    """List what we have available for testing"""
    test_data_dir = "/Users/user/Desktop/WhisperX/test_data"
    
    print("\n📋 Available Test Samples:")
    print("-" * 40)
    
    if os.path.exists(test_data_dir):
        files = [f for f in os.listdir(test_data_dir) if f.endswith('.wav')]
        
        for audio_file in sorted(files):
            audio_path = os.path.join(test_data_dir, audio_file)
            transcript_file = audio_file.replace('.wav', '.txt')
            transcript_path = os.path.join(test_data_dir, transcript_file)
            
            if os.path.exists(transcript_path):
                with open(transcript_path, 'r') as f:
                    transcript = f.read().strip()
                
                # Get audio info
                try:
                    import soundfile as sf
                    info = sf.info(audio_path)
                    duration = info.duration
                    sample_rate = info.samplerate
                    
                    print(f"🎵 {audio_file}")
                    print(f"   Duration: {duration:.1f}s, Rate: {sample_rate}Hz")
                    print(f"   Text: {transcript[:80]}...")
                    print()
                    
                except Exception as e:
                    print(f"🎵 {audio_file} (info unavailable)")
            else:
                print(f"🎵 {audio_file} (no transcript)")
    else:
        print("No test data directory found")

if __name__ == "__main__":
    print("WhisperX Test Data Setup")
    print("=" * 80)
    
    # Download samples
    num_downloaded = download_real_test_samples()
    
    # List what we have
    list_downloaded_samples()
    
    if num_downloaded > 0:
        print(f"\n🎉 Success! {num_downloaded} samples ready for testing")
        print("🚀 Next: Run WhisperX tests with real audio data")
    else:
        print("\n⚠️ No samples downloaded. Check network connection.")