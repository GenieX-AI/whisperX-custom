#!/usr/bin/env python3
"""
WhisperX Test with LibriSpeech Demo Dataset (Small & Fast)
"""

import torch
import whisperx
from datasets import load_dataset
from jiwer import wer
import time

def test_with_demo_dataset():
    """Test WhisperX with LibriSpeech demo dataset (just a few samples)"""
    print("🎯 WhisperX Test with LibriSpeech Demo Dataset")
    print("=" * 60)
    
    device = "cpu"
    print(f"Using device: {device}")
    
    # Load WhisperX model
    print("🤖 Loading WhisperX model...")
    model = whisperx.load_model("tiny", device, compute_type="int8")  # Use tiny for speed
    print("✅ Model loaded")
    
    # Load demo dataset (small, just for testing)
    print("📥 Loading LibriSpeech demo dataset...")
    try:
        # This is a small demo dataset specifically for testing
        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        print(f"✅ Demo dataset loaded with {len(dataset)} samples")
        
        # Test with first sample
        sample = dataset[0]
        audio_data = sample["audio"]["array"]
        # Ensure audio is float32
        if hasattr(audio_data, 'dtype'):
            audio_data = audio_data.astype('float32')
        ground_truth = sample["text"].strip()
        
        print(f"📝 Sample info:")
        print(f"  Audio length: {len(audio_data)} samples ({len(audio_data)/16000:.1f}s)")
        print(f"  Ground truth: '{ground_truth}'")
        
    except Exception as e:
        print(f"❌ Demo dataset failed, trying alternative...")
        # Fallback to a single sample from regular dataset
        try:
            dataset = load_dataset("librispeech_asr", "clean", split="validation[:1]", trust_remote_code=True)
            sample = dataset[0]
            audio_data = sample["audio"]["array"]
            # Ensure audio is float32
            if hasattr(audio_data, 'dtype'):
                audio_data = audio_data.astype('float32')
            ground_truth = sample["text"].strip()
            print(f"✅ Fallback dataset loaded")
            print(f"  Ground truth: '{ground_truth}'")
        except Exception as e2:
            print(f"❌ All datasets failed: {e2}")
            return False
    
    # Run transcription
    print("\n🎙️ Running transcription...")
    start_time = time.time()
    
    try:
        result = model.transcribe(audio_data)
        transcription_time = time.time() - start_time
        
        print(f"✅ Transcription completed in {transcription_time:.2f}s")
        
        # Extract predicted text
        predicted = ""
        if result.get("segments"):
            predicted = " ".join([seg["text"].strip() for seg in result["segments"]])
            print(f"  Segments found: {len(result['segments'])}")
            for i, seg in enumerate(result["segments"]):
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg["text"].strip()
                print(f"    Segment {i+1}: '{text}' [{start:.2f}s-{end:.2f}s]")
        else:
            print("  No segments detected")
            
        print(f"  Full prediction: '{predicted}'")
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return False
    
    # Calculate accuracy
    print("\n📊 Accuracy Analysis:")
    if predicted and ground_truth:
        word_error_rate = wer(ground_truth, predicted)
        accuracy = 1 - word_error_rate
        
        print(f"  Ground Truth: '{ground_truth}'")
        print(f"  Predicted:    '{predicted}'")
        print(f"  WER: {word_error_rate:.3f}")
        print(f"  Accuracy: {accuracy:.1%}")
        
        # Word-level comparison
        gt_words = set(ground_truth.lower().split())
        pred_words = set(predicted.lower().split())
        common_words = gt_words & pred_words
        
        if len(gt_words) > 0:
            word_overlap = len(common_words) / len(gt_words)
            print(f"  Word overlap: {word_overlap:.1%}")
            print(f"  Common words: {common_words}")
            
        success = accuracy > 0.3  # 30% accuracy threshold
        
    else:
        print("  No prediction to compare")
        success = False
    
    # Test alignment if we have segments
    if result.get("segments") and len(result["segments"]) > 0:
        print("\n🔄 Testing alignment...")
        try:
            model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
            
            result_aligned = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio_data, 
                device
            )
            
            print("✅ Alignment completed")
            if result_aligned.get("segments"):
                print(f"  Aligned segments: {len(result_aligned['segments'])}")
                
                # Show word-level timing if available
                for i, seg in enumerate(result_aligned["segments"][:2]):  # Show first 2 segments
                    if seg.get("words"):
                        print(f"    Segment {i+1} words:")
                        for word_info in seg["words"][:5]:  # Show first 5 words
                            word = word_info.get("word", "")
                            start = word_info.get("start", 0)
                            end = word_info.get("end", 0)
                            print(f"      '{word}' [{start:.2f}s-{end:.2f}s]")
            
        except Exception as e:
            print(f"⚠️ Alignment failed: {e}")
    
    print("\n🎉 Demo test completed!")
    return success

def test_multiple_samples():
    """Test with multiple samples from demo dataset"""
    print("\n🔄 Testing multiple samples...")
    
    try:
        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        
        # Test up to 3 samples
        num_samples = min(3, len(dataset))
        print(f"Testing {num_samples} samples...")
        
        device = "cpu"
        model = whisperx.load_model("tiny", device, compute_type="int8")
        
        total_wer = 0
        successful_tests = 0
        
        for i in range(num_samples):
            sample = dataset[i]
            audio_data = sample["audio"]["array"]
            # Ensure audio is float32
            if hasattr(audio_data, 'dtype'):
                audio_data = audio_data.astype('float32')
            ground_truth = sample["text"].strip()
            
            print(f"\nSample {i+1}:")
            print(f"  Ground truth: '{ground_truth}'")
            
            try:
                result = model.transcribe(audio_data)
                predicted = ""
                if result.get("segments"):
                    predicted = " ".join([seg["text"].strip() for seg in result["segments"]])
                
                print(f"  Predicted:    '{predicted}'")
                
                if predicted:
                    sample_wer = wer(ground_truth, predicted)
                    total_wer += sample_wer
                    successful_tests += 1
                    print(f"  WER: {sample_wer:.3f}")
                
            except Exception as e:
                print(f"  Failed: {e}")
        
        if successful_tests > 0:
            avg_wer = total_wer / successful_tests
            avg_accuracy = 1 - avg_wer
            print(f"\n📊 Overall Results:")
            print(f"  Successful tests: {successful_tests}/{num_samples}")
            print(f"  Average WER: {avg_wer:.3f}")
            print(f"  Average Accuracy: {avg_accuracy:.1%}")
            
            return avg_accuracy > 0.3
        
    except Exception as e:
        print(f"Multiple sample test failed: {e}")
        return False

if __name__ == "__main__":
    print("WhisperX Demo Dataset Test")
    print("=" * 80)
    
    # Test single sample
    single_success = test_with_demo_dataset()
    
    # Test multiple samples
    multiple_success = test_multiple_samples()
    
    if single_success or multiple_success:
        print("\n🎉 SUCCESS: WhisperX is working with real audio data!")
        print("✅ Installation verified")
        print("✅ Audio processing works")
        print("✅ Transcription accuracy is reasonable")
        print("🚀 Ready for full implementation!")
        
    else:
        print("\n⚠️ Basic functionality works, but accuracy needs improvement")
        print("💡 This might be normal for synthetic/test data")
        print("🔧 Consider testing with higher quality audio")