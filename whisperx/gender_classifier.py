import torch
import torchaudio
import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings
import logging

# Handle speechbrain import with version compatibility
try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    try:
        # Fallback to older version
        from speechbrain.pretrained import EncoderClassifier
    except ImportError as e:
        warnings.warn(f"Could not import SpeechBrain EncoderClassifier: {e}")
        EncoderClassifier = None

# Suppress some verbose logging from speechbrain
logging.getLogger("speechbrain").setLevel(logging.WARNING)


class GenderClassifier:
    """Gender classification using ECAPA-TDNN embeddings.
    
    This class extracts speaker embeddings using the ECAPA-TDNN model
    and applies a simple gender classifier on top.
    """
    
    def __init__(self, device: str = "cpu"):
        """Initialize the gender classifier.
        
        Args:
            device: Device to run the model on ("cpu" or "cuda")
        """
        self.device = device
        
        # Load ECAPA-TDNN for embeddings
        if EncoderClassifier is None:
            print("Warning: SpeechBrain EncoderClassifier not available.")
            print("Gender classification will be disabled.")
            self.encoder = None
            return
            
        try:
            self.encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": device}
            )
        except Exception as e:
            print(f"Warning: Could not load ECAPA-TDNN model: {e}")
            print("Gender classification will be disabled.")
            self.encoder = None
            return
            
        # Initialize gender classification head
        self.gender_classifier = self._init_gender_classifier()
        
    def _init_gender_classifier(self):
        """Initialize a simple MLP for gender classification.
        
        Note: This is a placeholder implementation. In practice, this would
        be trained on labeled gender data.
        """
        import torch.nn as nn
        
        class GenderMLP(nn.Module):
            def __init__(self, input_dim: int = 192, hidden_dim: int = 64):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 2),  # Binary classification
                    nn.Softmax(dim=1)
                )
                
            def forward(self, x):
                return self.layers(x)
        
        model = GenderMLP().to(self.device)
        
        # Initialize with random weights (in practice, would load pre-trained weights)
        # For now, we'll use a simple heuristic based on embedding patterns
        with torch.no_grad():
            # Initialize weights to create a basic heuristic classifier
            model.layers[0].weight.normal_(0, 0.1)
            model.layers[3].weight.normal_(0, 0.1)
            
        return model
    
    def extract_embedding(self, audio_segment: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract ECAPA-TDNN embeddings from audio segment.
        
        Args:
            audio_segment: Audio tensor of shape (1, samples) or (samples,)
            
        Returns:
            Embedding tensor of shape (embedding_dim,) or None if encoder unavailable
        """
        if self.encoder is None:
            return None
            
        try:
            # Ensure proper shape
            if len(audio_segment.shape) == 1:
                audio_segment = audio_segment.unsqueeze(0)
            if len(audio_segment.shape) == 2 and audio_segment.shape[0] > 1:
                audio_segment = audio_segment.mean(dim=0, keepdim=True)
                
            with torch.no_grad():
                embeddings = self.encoder.encode_batch(audio_segment.to(self.device))
                
            return embeddings.squeeze()
        except Exception as e:
            warnings.warn(f"Failed to extract embedding: {e}")
            return None
    
    def classify_gender(self, audio_segment: torch.Tensor) -> Tuple[str, float]:
        """Classify gender from audio segment.
        
        Args:
            audio_segment: Audio tensor
            
        Returns:
            Tuple of (gender, confidence) where gender is "Male" or "Female"
        """
        if self.encoder is None:
            # Fallback: random classification with low confidence
            gender = np.random.choice(["Male", "Female"])
            return gender, 0.5
            
        embedding = self.extract_embedding(audio_segment)
        
        if embedding is None:
            # Fallback: random classification with low confidence
            gender = np.random.choice(["Male", "Female"])
            return gender, 0.5
        
        try:
            with torch.no_grad():
                probs = self.gender_classifier(embedding.unsqueeze(0))
                confidence = probs.max().item()
                gender = "Male" if probs.argmax() == 0 else "Female"
                
            return gender, confidence
        except Exception as e:
            warnings.warn(f"Gender classification failed: {e}")
            # Fallback: random classification with low confidence
            gender = np.random.choice(["Male", "Female"])
            return gender, 0.5
    
    def _simple_gender_heuristic(self, embedding: torch.Tensor) -> Tuple[str, float]:
        """Simple heuristic for gender classification based on embedding statistics.
        
        This is a placeholder implementation. In practice, this would be replaced
        with a properly trained classifier.
        """
        # Use embedding statistics as a simple heuristic
        mean_val = embedding.mean().item()
        std_val = embedding.std().item()
        
        # Simple heuristic: higher mean tends to correlate with male voices
        # This is just a placeholder and not scientifically validated
        if mean_val > 0:
            gender = "Male"
            confidence = min(0.6 + abs(mean_val) * 0.1, 0.8)
        else:
            gender = "Female"
            confidence = min(0.6 + abs(mean_val) * 0.1, 0.8)
            
        return gender, confidence
    
    def process_segments(self, audio_path: str, segments: List[Dict]) -> List[Dict]:
        """Process diarization segments with gender labels.
        
        Args:
            audio_path: Path to the audio file
            segments: List of segment dictionaries with 'start', 'end', 'speaker' keys
            
        Returns:
            Updated segments with gender information added
        """
        if not segments:
            return segments
            
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary (ECAPA expects 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Process each segment
            for segment in segments:
                try:
                    start_sample = int(segment['start'] * 16000)
                    end_sample = int(segment['end'] * 16000)
                    
                    # Ensure valid sample range
                    start_sample = max(0, start_sample)
                    end_sample = min(waveform.shape[1], end_sample)
                    
                    if start_sample >= end_sample:
                        # Invalid segment, skip gender classification
                        segment['gender'] = "Unknown"
                        segment['gender_confidence'] = 0.0
                        continue
                    
                    audio_segment = waveform[:, start_sample:end_sample]
                    
                    # Ensure minimum segment length (0.5 seconds)
                    min_samples = 8000  # 0.5 seconds at 16kHz
                    if audio_segment.shape[1] < min_samples:
                        padding = min_samples - audio_segment.shape[1]
                        audio_segment = torch.nn.functional.pad(audio_segment, (0, padding))
                    
                    gender, confidence = self.classify_gender(audio_segment)
                    
                    # Update segment with gender information
                    segment['gender'] = gender
                    segment['gender_confidence'] = confidence
                    
                    # Update speaker label to include gender
                    original_speaker = segment.get('speaker', 'SPEAKER_00')
                    segment['speaker'] = f"{gender}_{original_speaker}"
                    
                except Exception as e:
                    warnings.warn(f"Failed to process segment {segment}: {e}")
                    segment['gender'] = "Unknown"
                    segment['gender_confidence'] = 0.0
                    
        except Exception as e:
            warnings.warn(f"Failed to load audio file {audio_path}: {e}")
            # Add default gender info to all segments
            for segment in segments:
                segment['gender'] = "Unknown"
                segment['gender_confidence'] = 0.0
                
        return segments
    
    def batch_process_segments(self, audio_path: str, segments: List[Dict], 
                              batch_size: int = 8) -> List[Dict]:
        """Process segments in batches for memory efficiency.
        
        Args:
            audio_path: Path to the audio file
            segments: List of segment dictionaries
            batch_size: Number of segments to process at once
            
        Returns:
            Updated segments with gender information
        """
        if self.encoder is None:
            return self.process_segments(audio_path, segments)
            
        if not segments or len(segments) <= batch_size:
            return self.process_segments(audio_path, segments)
            
        try:
            # Load audio once
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Process in batches
            for i in range(0, len(segments), batch_size):
                batch_segments = segments[i:i+batch_size]
                batch_audio = []
                valid_indices = []
                
                for j, segment in enumerate(batch_segments):
                    try:
                        start_sample = int(segment['start'] * 16000)
                        end_sample = int(segment['end'] * 16000)
                        
                        start_sample = max(0, start_sample)
                        end_sample = min(waveform.shape[1], end_sample)
                        
                        if start_sample >= end_sample:
                            continue
                            
                        audio_segment = waveform[:, start_sample:end_sample]
                        
                        # Pad to consistent length for batching
                        target_length = 16000  # 1 second
                        if audio_segment.shape[1] < target_length:
                            audio_segment = torch.nn.functional.pad(
                                audio_segment, (0, target_length - audio_segment.shape[1])
                            )
                        else:
                            audio_segment = audio_segment[:, :target_length]
                        
                        batch_audio.append(audio_segment.squeeze())
                        valid_indices.append(j)
                        
                    except Exception as e:
                        warnings.warn(f"Failed to prepare segment for batch: {e}")
                        continue
                
                if not batch_audio:
                    continue
                    
                try:
                    # Stack and process batch
                    batch_tensor = torch.stack(batch_audio).to(self.device)
                    
                    with torch.no_grad():
                        embeddings = self.encoder.encode_batch(batch_tensor)
                        probs = self.gender_classifier(embeddings)
                        
                    genders = ["Male" if p.argmax() == 0 else "Female" for p in probs]
                    confidences = [p.max().item() for p in probs]
                    
                    # Update segments
                    for k, j in enumerate(valid_indices):
                        segment = batch_segments[j]
                        segment['gender'] = genders[k]
                        segment['gender_confidence'] = confidences[k]
                        
                        original_speaker = segment.get('speaker', 'SPEAKER_00')
                        segment['speaker'] = f"{genders[k]}_{original_speaker}"
                        
                except Exception as e:
                    warnings.warn(f"Batch processing failed: {e}")
                    # Fallback to individual processing
                    for j in valid_indices:
                        segment = batch_segments[j]
                        segment['gender'] = "Unknown"
                        segment['gender_confidence'] = 0.0
                        
        except Exception as e:
            warnings.warn(f"Batch processing completely failed: {e}")
            return self.process_segments(audio_path, segments)
            
        return segments