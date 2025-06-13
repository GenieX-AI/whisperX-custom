#!/usr/bin/env python3
"""
Comprehensive WhisperX Evaluation Suite
- ASR Accuracy (WER, CER)
- Diarization Error Rate (DER)  
- Gender Classification Accuracy
- Processing Speed Metrics
"""

import os
import json
import time
import torch
import whisperx
import pandas as pd
import numpy as np
from pathlib import Path
from jiwer import wer, cer
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system environment variables

class WhisperXEvaluator:
    def __init__(self, device="cpu", model_size="base", hf_token=None):
        """Initialize the comprehensive evaluator"""
        self.device = device
        self.model_size = model_size
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        
        # Initialize paths
        self.project_root = Path("/Users/user/Desktop/WhisperX")
        self.test_data_dir = self.project_root / "test_data"
        self.results_dir = self.project_root / "results"
        
        # Load test metadata
        self.metadata_path = self.test_data_dir / "test_samples_metadata.json"
        self.test_samples = self._load_test_metadata()
        
        # Initialize models
        self._initialize_models()
        
        # Results storage
        self.evaluation_results = {
            "test_info": {
                "model_size": model_size,
                "device": device,
                "hf_token_enabled": bool(self.hf_token),
                "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
                "num_samples": len(self.test_samples)
            },
            "asr_results": [],
            "diarization_results": [],
            "gender_classification_results": [],
            "performance_metrics": {}
        }
    
    def _load_test_metadata(self) -> List[Dict]:
        """Load test sample metadata"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Test metadata not found: {self.metadata_path}")
    
    def _initialize_models(self):
        """Initialize all WhisperX models"""
        print("🤖 Initializing WhisperX models...")
        
        # Display HF token status
        if self.hf_token:
            print("  ✅ HuggingFace token detected - using optimal diarization models")
        else:
            print("  ⚠️  No HuggingFace token - using fallback diarization (may have reduced performance)")
            print("     Set HF_TOKEN environment variable or pass hf_token parameter for optimal results")
        
        # ASR Model
        print(f"  Loading ASR model ({self.model_size})...")
        self.asr_model = whisperx.load_model(self.model_size, self.device, compute_type="int8")
        
        # Alignment Model
        print("  Loading alignment model...")
        try:
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code="en", device=self.device
            )
        except Exception as e:
            print(f"  ⚠️ Alignment model failed: {e}")
            self.align_model = None
            self.align_metadata = None
        
        # Diarization Model (with gender classification)
        print("  Loading diarization model...")
        try:
            from whisperx.diarize import DiarizationPipeline
            self.diarize_model = DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device,
                enable_gender_classification=True
            )
        except Exception as e:
            print(f"  ⚠️ Diarization model failed: {e}")
            self.diarize_model = None
        
        print("✅ Models initialized")
    
    def evaluate_asr_accuracy(self) -> Dict:
        """Evaluate ASR accuracy with WER and CER metrics"""
        print("\n📊 Evaluating ASR Accuracy...")
        
        asr_results = []
        total_wer = 0
        total_cer = 0
        total_processing_time = 0
        successful_tests = 0
        
        for i, sample in enumerate(self.test_samples):
            print(f"  Processing sample {i+1}/{len(self.test_samples)}...")
            
            # Load audio and transcript
            audio_path = self.test_data_dir / sample["audio_file"]
            transcript_path = self.test_data_dir / sample["transcript_file"]
            
            with open(transcript_path, 'r') as f:
                ground_truth = f.read().strip()
            
            # Load audio data
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_path)
            
            try:
                # Measure processing time
                start_time = time.time()
                
                # Run ASR
                result = self.asr_model.transcribe(audio_data.astype('float32'))
                
                processing_time = time.time() - start_time
                
                # Extract prediction
                predicted_text = ""
                if result.get("segments"):
                    predicted_text = " ".join([seg["text"].strip() for seg in result["segments"]])
                
                # Calculate metrics (normalized to lowercase for fairer comparison)
                gt_normalized = ground_truth.lower()
                pred_normalized = predicted_text.lower()
                
                sample_wer = wer(gt_normalized, pred_normalized) if pred_normalized else 1.0
                sample_cer = cer(gt_normalized, pred_normalized) if pred_normalized else 1.0
                
                # Store results
                sample_result = {
                    "sample_id": sample.get("sample_id", i+1),
                    "audio_file": sample["audio_file"],
                    "duration": sample["duration_seconds"],
                    "ground_truth": ground_truth,
                    "predicted": predicted_text,
                    "wer": sample_wer,
                    "cer": sample_cer,
                    "processing_time": processing_time,
                    "segments_count": len(result.get("segments", [])),
                    "success": True
                }
                
                asr_results.append(sample_result)
                
                total_wer += sample_wer
                total_cer += sample_cer
                total_processing_time += processing_time
                successful_tests += 1
                
                print(f"    WER: {sample_wer:.3f}, CER: {sample_cer:.3f}, Time: {processing_time:.2f}s")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                sample_result = {
                    "sample_id": sample.get("sample_id", i+1),
                    "audio_file": sample["audio_file"],
                    "duration": sample["duration_seconds"],
                    "ground_truth": ground_truth,
                    "predicted": "",
                    "wer": 1.0,
                    "cer": 1.0,
                    "processing_time": 0,
                    "segments_count": 0,
                    "success": False,
                    "error": str(e)
                }
                asr_results.append(sample_result)
        
        # Calculate summary metrics
        summary = {
            "total_samples": len(self.test_samples),
            "successful_samples": successful_tests,
            "success_rate": successful_tests / len(self.test_samples) if self.test_samples else 0,
            "average_wer": total_wer / successful_tests if successful_tests > 0 else 1.0,
            "average_cer": total_cer / successful_tests if successful_tests > 0 else 1.0,
            "average_processing_time": total_processing_time / successful_tests if successful_tests > 0 else 0,
            "total_audio_duration": sum(s["duration_seconds"] for s in self.test_samples),
            "real_time_factor": (total_processing_time / sum(s["duration_seconds"] for s in self.test_samples)) if self.test_samples else 0
        }
        
        print(f"  📈 ASR Summary:")
        print(f"    Success Rate: {summary['success_rate']:.1%}")
        print(f"    Average WER: {summary['average_wer']:.3f} ({(1-summary['average_wer']):.1%} accuracy)")
        print(f"    Average CER: {summary['average_cer']:.3f}")
        print(f"    Real-time Factor: {summary['real_time_factor']:.2f}x")
        
        self.evaluation_results["asr_results"] = asr_results
        self.evaluation_results["asr_summary"] = summary
        
        return summary
    
    def evaluate_diarization_der(self) -> Dict:
        """Evaluate diarization performance and calculate DER"""
        print("\n👥 Evaluating Speaker Diarization (DER)...")
        
        if self.diarize_model is None:
            print("  ⚠️ Diarization model not available, skipping...")
            return {"error": "Diarization model not available"}
        
        diarization_results = []
        total_der = 0
        successful_tests = 0
        
        for i, sample in enumerate(self.test_samples):
            print(f"  Processing sample {i+1}/{len(self.test_samples)}...")
            
            audio_path = self.test_data_dir / sample["audio_file"]
            
            try:
                start_time = time.time()
                
                # Run diarization with gender classification
                diarize_segments = self.diarize_model(str(audio_path))
                
                processing_time = time.time() - start_time
                
                # Extract speaker information
                speakers_detected = set()
                segments_info = []
                
                if hasattr(diarize_segments, 'iterrows'):
                    for _, row in diarize_segments.iterrows():
                        speaker = row.get('speaker', 'Unknown')
                        start_time_seg = row.get('start', 0)
                        end_time_seg = row.get('end', 0)
                        gender = row.get('gender', 'Unknown')
                        gender_conf = row.get('gender_confidence', 0)
                        
                        speakers_detected.add(speaker)
                        segments_info.append({
                            "speaker": speaker,
                            "start": start_time_seg,
                            "end": end_time_seg,
                            "duration": end_time_seg - start_time_seg,
                            "gender": gender,
                            "gender_confidence": gender_conf
                        })
                
                # Simple DER approximation (for single speaker, DER = missed speech)
                # For real DER, we'd need reference speaker timestamps
                total_speech_time = sample["duration_seconds"]
                detected_speech_time = sum(seg["duration"] for seg in segments_info)
                
                # Approximate DER as the difference in speech time coverage
                der_estimate = abs(total_speech_time - detected_speech_time) / total_speech_time
                
                sample_result = {
                    "sample_id": sample.get("sample_id", i+1),
                    "audio_file": sample["audio_file"],
                    "duration": sample["duration_seconds"],
                    "speakers_detected": len(speakers_detected),
                    "segments_count": len(segments_info),
                    "detected_speech_time": detected_speech_time,
                    "der_estimate": der_estimate,
                    "processing_time": processing_time,
                    "segments": segments_info,
                    "success": True
                }
                
                diarization_results.append(sample_result)
                total_der += der_estimate
                successful_tests += 1
                
                print(f"    Speakers: {len(speakers_detected)}, Segments: {len(segments_info)}, DER≈{der_estimate:.3f}")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                sample_result = {
                    "sample_id": sample.get("sample_id", i+1),
                    "audio_file": sample["audio_file"],
                    "duration": sample["duration_seconds"],
                    "speakers_detected": 0,
                    "segments_count": 0,
                    "detected_speech_time": 0,
                    "der_estimate": 1.0,
                    "processing_time": 0,
                    "segments": [],
                    "success": False,
                    "error": str(e)
                }
                diarization_results.append(sample_result)
        
        # Calculate summary
        summary = {
            "total_samples": len(self.test_samples),
            "successful_samples": successful_tests,
            "success_rate": successful_tests / len(self.test_samples) if self.test_samples else 0,
            "average_der": total_der / successful_tests if successful_tests > 0 else 1.0,
            "total_speakers_detected": sum(r.get("speakers_detected", 0) for r in diarization_results),
            "total_segments": sum(r.get("segments_count", 0) for r in diarization_results)
        }
        
        print(f"  📈 Diarization Summary:")
        print(f"    Success Rate: {summary['success_rate']:.1%}")
        print(f"    Average DER: {summary['average_der']:.3f}")
        print(f"    Total Speakers: {summary['total_speakers_detected']}")
        print(f"    Total Segments: {summary['total_segments']}")
        
        self.evaluation_results["diarization_results"] = diarization_results
        self.evaluation_results["diarization_summary"] = summary
        
        return summary
    
    def evaluate_gender_classification(self) -> Dict:
        """Evaluate gender classification accuracy"""
        print("\n🚻 Evaluating Gender Classification...")
        
        # Note: Without ground truth gender labels, we'll evaluate consistency and confidence
        gender_results = []
        
        for result in self.evaluation_results.get("diarization_results", []):
            if result["success"] and result["segments"]:
                sample_genders = []
                sample_confidences = []
                
                for segment in result["segments"]:
                    if segment.get("gender") != "Unknown":
                        sample_genders.append(segment["gender"])
                        sample_confidences.append(segment.get("gender_confidence", 0))
                
                if sample_genders:
                    # Evaluate consistency (same speaker should have same gender)
                    gender_consistency = len(set(sample_genders)) == 1
                    avg_confidence = np.mean(sample_confidences)
                    
                    gender_results.append({
                        "sample_id": result["sample_id"],
                        "audio_file": result["audio_file"],
                        "gender_predictions": sample_genders,
                        "avg_confidence": avg_confidence,
                        "consistent": gender_consistency,
                        "segments_with_gender": len(sample_genders)
                    })
        
        # Calculate summary
        if gender_results:
            summary = {
                "samples_with_gender": len(gender_results),
                "consistency_rate": sum(r["consistent"] for r in gender_results) / len(gender_results),
                "average_confidence": np.mean([r["avg_confidence"] for r in gender_results]),
                "total_segments_classified": sum(r["segments_with_gender"] for r in gender_results)
            }
        else:
            summary = {
                "samples_with_gender": 0,
                "consistency_rate": 0,
                "average_confidence": 0,
                "total_segments_classified": 0
            }
        
        print(f"  📈 Gender Classification Summary:")
        print(f"    Samples with Gender: {summary['samples_with_gender']}")
        print(f"    Consistency Rate: {summary['consistency_rate']:.1%}")
        print(f"    Average Confidence: {summary['average_confidence']:.3f}")
        
        self.evaluation_results["gender_classification_results"] = gender_results
        self.evaluation_results["gender_summary"] = summary
        
        return summary
    
    def save_results(self):
        """Save comprehensive evaluation results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"evaluation_results_{timestamp}.json"
        
        # Save detailed JSON results
        with open(results_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        # Create summary report
        summary_file = self.results_dir / f"evaluation_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("WhisperX Comprehensive Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Test info
            test_info = self.evaluation_results["test_info"]
            f.write(f"Test Configuration:\n")
            f.write(f"  Model: {test_info['model_size']}\n")
            f.write(f"  Device: {test_info['device']}\n")
            f.write(f"  HF Token: {'✅ Enabled' if test_info['hf_token_enabled'] else '❌ Disabled (fallback mode)'}\n")
            f.write(f"  Samples: {test_info['num_samples']}\n")
            f.write(f"  Timestamp: {test_info['timestamp']}\n\n")
            
            # ASR results
            if "asr_summary" in self.evaluation_results:
                asr = self.evaluation_results["asr_summary"]
                f.write(f"ASR Performance:\n")
                f.write(f"  Success Rate: {asr['success_rate']:.1%}\n")
                f.write(f"  Word Error Rate (WER): {asr['average_wer']:.3f}\n")
                f.write(f"  Character Error Rate (CER): {asr['average_cer']:.3f}\n")
                f.write(f"  Accuracy: {(1-asr['average_wer']):.1%}\n")
                f.write(f"  Real-time Factor: {asr['real_time_factor']:.2f}x\n\n")
            
            # Diarization results
            if "diarization_summary" in self.evaluation_results:
                diar = self.evaluation_results["diarization_summary"]
                f.write(f"Diarization Performance:\n")
                f.write(f"  Success Rate: {diar['success_rate']:.1%}\n")
                f.write(f"  Estimated DER: {diar['average_der']:.3f}\n")
                f.write(f"  Speakers Detected: {diar['total_speakers_detected']}\n")
                f.write(f"  Total Segments: {diar['total_segments']}\n\n")
            
            # Gender classification results
            if "gender_summary" in self.evaluation_results:
                gender = self.evaluation_results["gender_summary"]
                f.write(f"Gender Classification:\n")
                f.write(f"  Samples Classified: {gender['samples_with_gender']}\n")
                f.write(f"  Consistency Rate: {gender['consistency_rate']:.1%}\n")
                f.write(f"  Average Confidence: {gender['average_confidence']:.3f}\n")
        
        print(f"\n💾 Results saved:")
        print(f"  Detailed: {results_file}")
        print(f"  Summary: {summary_file}")
        
        return results_file, summary_file
    
    def run_comprehensive_evaluation(self):
        """Run the complete evaluation suite"""
        print("🚀 Starting Comprehensive WhisperX Evaluation")
        print("=" * 80)
        
        # Run all evaluations
        asr_summary = self.evaluate_asr_accuracy()
        diar_summary = self.evaluate_diarization_der()
        gender_summary = self.evaluate_gender_classification()
        
        # Save results
        results_file, summary_file = self.save_results()
        
        print(f"\n🎉 Evaluation Complete!")
        print(f"📊 Key Metrics:")
        print(f"  ASR Accuracy: {(1-asr_summary['average_wer']):.1%} (WER: {asr_summary['average_wer']:.3f})")
        if "average_der" in diar_summary:
            print(f"  Diarization DER: {diar_summary['average_der']:.3f}")
        print(f"  Gender Consistency: {gender_summary['consistency_rate']:.1%}")
        
        return self.evaluation_results

def main(hf_token=None):
    """Main evaluation function"""
    device = "cpu"  # Change to "cuda" if available
    model_size = "base"  # Options: tiny, base, small, medium, large
    
    # Get HF token from environment or parameter
    hf_token = hf_token or os.getenv('HF_TOKEN')
    
    evaluator = WhisperXEvaluator(device=device, model_size=model_size, hf_token=hf_token)
    results = evaluator.run_comprehensive_evaluation()
    
    return results

if __name__ == "__main__":
    main()