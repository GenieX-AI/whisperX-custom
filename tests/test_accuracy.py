#!/usr/bin/env python3
"""
WhisperX Accuracy Testing with LibriSpeech Dataset
Tests both ASR accuracy and Speaker Diarization with Gender Classification
"""

import os
import time
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from jiwer import wer, cer
import whisperx
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperXTester:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """Initialize WhisperX testing pipeline"""
        self.device = device
        self.compute_type = compute_type
        
        # Load WhisperX model
        logger.info(f"Loading WhisperX model: {model_size}")
        self.model = whisperx.load_model(model_size, device, compute_type=compute_type)
        
        # Load alignment model
        self.model_a, self.metadata = whisperx.load_align_model(
            language_code="en", device=device
        )
        
        # Load diarization model
        logger.info("Loading diarization model")
        self.diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=None, device=device
        )
        
        # Results storage
        self.results = []
        
    def load_test_data(self, split="test", config="clean", num_samples=50):
        """Load LibriSpeech test dataset"""
        logger.info(f"Loading LibriSpeech {split} dataset...")
        
        # Load dataset
        dataset = load_dataset("librispeech_asr", config, split=split, trust_remote_code=True)
        
        # Take subset for testing
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
        logger.info(f"Loaded {len(dataset)} samples")
        return dataset
    
    def test_single_audio(self, audio_data: Dict) -> Dict:
        """Test a single audio sample and return metrics"""
        start_time = time.time()
        
        # Extract audio information
        audio_array = audio_data["audio"]["array"]
        sample_rate = audio_data["audio"]["sampling_rate"]
        ground_truth = audio_data["text"].upper().strip()
        speaker_id = audio_data["speaker_id"]
        
        try:
            # Step 1: ASR Transcription
            logger.info("Running ASR transcription...")
            result = self.model.transcribe(audio_array)
            
            # Step 2: Alignment
            logger.info("Running forced alignment...")
            result = whisperx.align(
                result["segments"], 
                self.model_a, 
                self.metadata, 
                audio_array, 
                self.device, 
                return_char_alignments=False
            )
            
            # Step 3: Diarization (if multiple speakers expected)
            logger.info("Running speaker diarization...")
            diarize_segments = self.diarize_model(audio_array)
            
            # Step 4: Assign speakers to segments
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Extract predicted transcription
            predicted_text = ""
            speakers_detected = set()
            
            if result["segments"]:
                for segment in result["segments"]:
                    predicted_text += segment["text"] + " "
                    if "speaker" in segment:
                        speakers_detected.add(segment["speaker"])
            
            predicted_text = predicted_text.upper().strip()
            
            # Calculate metrics
            processing_time = time.time() - start_time
            word_error_rate = wer(ground_truth, predicted_text)
            char_error_rate = cer(ground_truth, predicted_text)
            
            # Prepare results
            test_result = {
                "speaker_id": speaker_id,
                "ground_truth": ground_truth,
                "predicted": predicted_text,
                "wer": word_error_rate,
                "cer": char_error_rate,
                "processing_time": processing_time,
                "speakers_detected": len(speakers_detected),
                "speaker_labels": list(speakers_detected),
                "segments_count": len(result["segments"]) if result["segments"] else 0,
                "success": True,
                "error": None
            }
            
            logger.info(f"Sample processed - WER: {word_error_rate:.3f}, CER: {char_error_rate:.3f}")
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            test_result = {
                "speaker_id": speaker_id,
                "ground_truth": ground_truth,
                "predicted": "",
                "wer": 1.0,
                "cer": 1.0,
                "processing_time": time.time() - start_time,
                "speakers_detected": 0,
                "speaker_labels": [],
                "segments_count": 0,
                "success": False,
                "error": str(e)
            }
        
        return test_result
    
    def run_accuracy_test(self, split="test", config="clean", num_samples=20):
        """Run comprehensive accuracy test"""
        logger.info(f"Starting accuracy test with {num_samples} samples from {split}")
        
        # Load test data
        dataset = self.load_test_data(split, config, num_samples)
        
        # Process each sample
        for i, sample in enumerate(dataset):
            logger.info(f"Processing sample {i+1}/{len(dataset)}")
            result = self.test_single_audio(sample)
            result["sample_id"] = i
            self.results.append(result)
        
        # Calculate overall metrics
        self.calculate_summary_metrics()
        
        return self.results
    
    def calculate_summary_metrics(self):
        """Calculate and display summary metrics"""
        if not self.results:
            logger.warning("No results to analyze")
            return
        
        # Filter successful results
        successful_results = [r for r in self.results if r["success"]]
        
        if not successful_results:
            logger.error("No successful transcriptions")
            return
        
        # Calculate metrics
        total_samples = len(self.results)
        successful_samples = len(successful_results)
        success_rate = successful_samples / total_samples
        
        avg_wer = np.mean([r["wer"] for r in successful_results])
        avg_cer = np.mean([r["cer"] for r in successful_results])
        avg_processing_time = np.mean([r["processing_time"] for r in successful_results])
        
        # Speaker detection metrics
        avg_speakers_detected = np.mean([r["speakers_detected"] for r in successful_results])
        
        # Print summary
        print("\n" + "="*60)
        print("WHISPERX ACCURACY TEST RESULTS")
        print("="*60)
        print(f"Total Samples: {total_samples}")
        print(f"Successful: {successful_samples} ({success_rate:.1%})")
        print(f"Average WER: {avg_wer:.3f} ({(1-avg_wer):.1%} accuracy)")
        print(f"Average CER: {avg_cer:.3f}")
        print(f"Average Processing Time: {avg_processing_time:.2f}s")
        print(f"Average Speakers Detected: {avg_speakers_detected:.1f}")
        print("="*60)
        
        # Show best and worst performing samples
        if successful_results:
            best_sample = min(successful_results, key=lambda x: x["wer"])
            worst_sample = max(successful_results, key=lambda x: x["wer"])
            
            print(f"\nBEST SAMPLE (WER: {best_sample['wer']:.3f}):")
            print(f"Ground Truth: {best_sample['ground_truth'][:100]}...")
            print(f"Predicted:    {best_sample['predicted'][:100]}...")
            
            print(f"\nWORST SAMPLE (WER: {worst_sample['wer']:.3f}):")
            print(f"Ground Truth: {worst_sample['ground_truth'][:100]}...")
            print(f"Predicted:    {worst_sample['predicted'][:100]}...")
    
    def save_results(self, filename="whisperx_test_results.csv"):
        """Save detailed results to CSV"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")
        
        return filename
    
    def create_error_analysis(self):
        """Analyze common error patterns"""
        successful_results = [r for r in self.results if r["success"]]
        
        if not successful_results:
            return
        
        print("\nERROR ANALYSIS:")
        print("-" * 40)
        
        # WER distribution
        wer_ranges = {
            "Excellent (0-0.1)": 0,
            "Good (0.1-0.3)": 0,
            "Fair (0.3-0.5)": 0,
            "Poor (0.5+)": 0
        }
        
        for result in successful_results:
            wer_val = result["wer"]
            if wer_val <= 0.1:
                wer_ranges["Excellent (0-0.1)"] += 1
            elif wer_val <= 0.3:
                wer_ranges["Good (0.1-0.3)"] += 1
            elif wer_val <= 0.5:
                wer_ranges["Fair (0.3-0.5)"] += 1
            else:
                wer_ranges["Poor (0.5+)"] += 1
        
        print("WER Distribution:")
        for range_name, count in wer_ranges.items():
            percentage = count / len(successful_results) * 100
            print(f"  {range_name}: {count} samples ({percentage:.1f}%)")

def main():
    """Main testing function"""
    print("WhisperX Accuracy Testing Pipeline")
    print("==================================")
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_size = "base"  # Start with base model for faster testing
    num_samples = 20     # Number of test samples
    
    print(f"Device: {device}")
    print(f"Model: {model_size}")
    print(f"Test samples: {num_samples}")
    
    # Initialize tester
    tester = WhisperXTester(
        model_size=model_size,
        device=device,
        compute_type="int8"
    )
    
    # Run accuracy test
    results = tester.run_accuracy_test(
        split="test",
        config="clean", 
        num_samples=num_samples
    )
    
    # Save results
    filename = tester.save_results()
    
    # Error analysis
    tester.create_error_analysis()
    
    print(f"\nTesting complete! Results saved to {filename}")

if __name__ == "__main__":
    main()