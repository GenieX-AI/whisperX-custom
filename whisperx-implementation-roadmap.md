# WhisperX v3 Enhancement: Complete Implementation Roadmap

## Executive Summary

This roadmap provides a comprehensive implementation guide for enhancing WhisperX v3 with advanced speaker embedding and gender classification capabilities. The three-phase approach progressively builds from minimal integration to full architectural restructuring, targeting legal-grade transcription requirements with <5% Diarization Error Rate (DER) and <5% Word Error Rate (WER).

---

## Phase 1: ECAPA Speaker Embedding Integration for Gender Classification

### 1.1 Technical Architecture

#### Current WhisperX v3 Architecture Analysis
```python
# WhisperX v3 Core Components
whisperx/
├── alignment.py      # Forced alignment with wav2vec2
├── asr.py           # Whisper ASR integration
├── audio.py         # Audio preprocessing
├── diarize.py       # Pyannote diarization wrapper
├── transcribe.py    # Main transcription pipeline
└── utils.py         # Utility functions
```

#### Integration Points
1. **Primary Integration**: `diarize.py` - Extend DiarizationPipeline class
2. **Secondary Integration**: `transcribe.py` - Modify pipeline flow
3. **Output Integration**: `alignment.py` - Update segment labeling

### 1.2 Detailed Implementation Steps

#### Step 1: Environment Setup
```bash
# Clone WhisperX and install dependencies
git clone https://github.com/m-bain/whisperX.git
cd whisperX
pip install -e .

# Install additional dependencies
pip install speechbrain==0.5.16
pip install torch==2.0.1+cu118
pip install librosa==0.10.1
```

#### Step 2: Create Gender Classification Module
```python
# whisperx/gender_classifier.py
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import numpy as np
from typing import Tuple, Dict, List

class GenderClassifier:
    def __init__(self, device="cuda"):
        self.device = device
        # Load ECAPA-TDNN for embeddings
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        
        # Initialize gender classification head
        self.gender_classifier = self._init_gender_classifier()
        
    def _init_gender_classifier(self):
        """Initialize a simple MLP for gender classification"""
        import torch.nn as nn
        
        class GenderMLP(nn.Module):
            def __init__(self, input_dim=192, hidden_dim=64):
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
        # Load pre-trained weights (train separately on VoxCeleb)
        # model.load_state_dict(torch.load("models/gender_classifier.pth"))
        return model
    
    def extract_embedding(self, audio_segment: torch.Tensor) -> torch.Tensor:
        """Extract ECAPA-TDNN embeddings from audio segment"""
        with torch.no_grad():
            embeddings = self.encoder.encode_batch(audio_segment)
        return embeddings.squeeze()
    
    def classify_gender(self, audio_segment: torch.Tensor) -> Tuple[str, float]:
        """Classify gender from audio segment"""
        embedding = self.extract_embedding(audio_segment)
        
        with torch.no_grad():
            probs = self.gender_classifier(embedding.unsqueeze(0))
            confidence = probs.max().item()
            gender = "Male" if probs.argmax() == 0 else "Female"
            
        return gender, confidence
    
    def process_segments(self, audio_path: str, segments: List[Dict]) -> List[Dict]:
        """Process diarization segments with gender labels"""
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary (ECAPA expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Process each segment
        for segment in segments:
            start_sample = int(segment['start'] * 16000)
            end_sample = int(segment['end'] * 16000)
            
            audio_segment = waveform[:, start_sample:end_sample]
            
            # Ensure minimum segment length (0.5 seconds)
            if audio_segment.shape[1] < 8000:
                padding = 8000 - audio_segment.shape[1]
                audio_segment = torch.nn.functional.pad(audio_segment, (0, padding))
            
            gender, confidence = self.classify_gender(audio_segment)
            
            # Update segment with gender information
            segment['gender'] = gender
            segment['gender_confidence'] = confidence
            segment['speaker'] = f"{gender} Speaker {segment['speaker']}"
            
        return segments
```

#### Step 3: Integrate with WhisperX Pipeline
```python
# Modify whisperx/diarize.py
import whisperx.gender_classifier as gc

class DiarizationPipeline:
    def __init__(self, 
                 use_auth_token=None, 
                 device: Optional[Union[str, torch.device]] = "cpu",
                 enable_gender_classification: bool = True):
        
        # Existing initialization
        self.model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=use_auth_token
        ).to(device)
        
        # Add gender classifier
        self.enable_gender = enable_gender_classification
        if self.enable_gender:
            self.gender_classifier = gc.GenderClassifier(device=device)
    
    def __call__(self, audio: Union[str, np.ndarray], 
                 min_speakers=None, max_speakers=None):
        # Run original diarization
        diarization = self.model(
            audio, 
            min_speakers=min_speakers, 
            max_speakers=max_speakers
        )
        
        # Convert to segments format
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        # Apply gender classification if enabled
        if self.enable_gender and isinstance(audio, str):
            segments = self.gender_classifier.process_segments(audio, segments)
        
        return {"segments": segments}
```

### 1.3 Performance Optimization

#### Memory Management
```python
# Batch processing for multiple segments
def batch_process_segments(audio_path: str, segments: List[Dict], 
                          batch_size: int = 16) -> List[Dict]:
    """Process segments in batches for memory efficiency"""
    waveform, sample_rate = torchaudio.load(audio_path)
    
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Process in batches
    for i in range(0, len(segments), batch_size):
        batch_segments = segments[i:i+batch_size]
        batch_audio = []
        
        for segment in batch_segments:
            start_sample = int(segment['start'] * 16000)
            end_sample = int(segment['end'] * 16000)
            audio_segment = waveform[:, start_sample:end_sample]
            
            # Pad to consistent length
            target_length = 16000  # 1 second
            if audio_segment.shape[1] < target_length:
                audio_segment = torch.nn.functional.pad(
                    audio_segment, (0, target_length - audio_segment.shape[1])
                )
            else:
                audio_segment = audio_segment[:, :target_length]
            
            batch_audio.append(audio_segment)
        
        # Stack and process batch
        batch_tensor = torch.stack([a.squeeze(0) for a in batch_audio])
        embeddings = self.encoder.encode_batch(batch_tensor)
        
        # Classify batch
        with torch.no_grad():
            probs = self.gender_classifier(embeddings)
            genders = ["Male" if p.argmax() == 0 else "Female" 
                      for p in probs]
            confidences = [p.max().item() for p in probs]
        
        # Update segments
        for j, segment in enumerate(batch_segments):
            segment['gender'] = genders[j]
            segment['gender_confidence'] = confidences[j]
            segment['speaker'] = f"{genders[j]} Speaker {segment['speaker']}"
    
    return segments
```

### 1.4 Testing and Validation

#### Unit Tests
```python
# tests/test_gender_classifier.py
import pytest
import torch
import torchaudio
from whisperx.gender_classifier import GenderClassifier

class TestGenderClassifier:
    @pytest.fixture
    def classifier(self):
        return GenderClassifier(device="cpu")
    
    @pytest.fixture
    def sample_audio(self):
        # Generate synthetic audio for testing
        sample_rate = 16000
        duration = 2.0
        frequency = 440.0
        
        t = torch.linspace(0, duration, int(sample_rate * duration))
        waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
        
        return waveform, sample_rate
    
    def test_embedding_extraction(self, classifier, sample_audio):
        waveform, _ = sample_audio
        embedding = classifier.extract_embedding(waveform)
        
        assert embedding.shape == (192,)  # ECAPA-TDNN embedding dimension
        assert not torch.isnan(embedding).any()
    
    def test_gender_classification(self, classifier, sample_audio):
        waveform, _ = sample_audio
        gender, confidence = classifier.classify_gender(waveform)
        
        assert gender in ["Male", "Female"]
        assert 0.0 <= confidence <= 1.0
    
    def test_batch_processing(self, classifier):
        # Test batch processing efficiency
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_00"}
        ]
        
        # Mock audio file processing
        processed = classifier.process_segments("test.wav", segments)
        
        assert len(processed) == len(segments)
        assert all('gender' in s for s in processed)
        assert all('gender_confidence' in s for s in processed)
```

#### Performance Benchmarks
```python
# benchmarks/benchmark_phase1.py
import time
import torch
from whisperx import Pipeline
import matplotlib.pyplot as plt

def benchmark_gender_classification():
    """Benchmark gender classification performance"""
    results = {
        'segment_length': [],
        'processing_time': [],
        'memory_usage': []
    }
    
    classifier = GenderClassifier(device="cuda")
    
    for duration in [0.5, 1.0, 2.0, 5.0, 10.0]:
        # Generate test audio
        audio = torch.randn(1, int(16000 * duration))
        
        # Measure processing time
        torch.cuda.synchronize()
        start_time = time.time()
        
        _ = classifier.classify_gender(audio)
        
        torch.cuda.synchronize()
        processing_time = time.time() - start_time
        
        # Measure memory usage
        memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        results['segment_length'].append(duration)
        results['processing_time'].append(processing_time)
        results['memory_usage'].append(memory_usage)
        
        torch.cuda.reset_peak_memory_stats()
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(results['segment_length'], results['processing_time'])
    ax1.set_xlabel('Segment Length (seconds)')
    ax1.set_ylabel('Processing Time (seconds)')
    ax1.set_title('Processing Time vs Segment Length')
    
    ax2.plot(results['segment_length'], results['memory_usage'])
    ax2.set_xlabel('Segment Length (seconds)')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage vs Segment Length')
    
    plt.tight_layout()
    plt.savefig('benchmarks/phase1_performance.png')
    
    return results
```

### 1.5 Deployment Configuration

#### Docker Configuration
```dockerfile
# Dockerfile.phase1
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements_phase1.txt .
RUN pip3 install -r requirements_phase1.txt

# Copy WhisperX with modifications
COPY whisperx/ ./whisperx/
COPY models/ ./models/

# Download pre-trained models
RUN python3 -c "from speechbrain.pretrained import EncoderClassifier; \
    EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', \
    savedir='pretrained_models/spkrec-ecapa-voxceleb')"

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=1

# Entry point
CMD ["python3", "-m", "whisperx.transcribe"]
```

#### Requirements File
```txt
# requirements_phase1.txt
torch==2.0.1+cu118
torchaudio==2.0.2+cu118
speechbrain==0.5.16
pyannote.audio==2.1.1
openai-whisper==20230918
faster-whisper==0.9.0
transformers==4.35.0
librosa==0.10.1
soundfile==0.12.1
ffmpeg-python==0.2.0
pytest==7.4.3
matplotlib==3.8.0
numpy==1.24.3
pandas==2.0.3
```

---

## Phase 2: Enhanced Pipeline with Voice Recognition Embeddings

### 2.1 Multi-Model Architecture

#### Architecture Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                    Audio Input (16kHz)                           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Feature Extraction                              │
│            (Mel-spectrogram / Raw waveform)                      │
└──────────┬──────────────┴──────────────┬────────────────────────┘
           │                             │
           ▼                             ▼
┌──────────────────────┐      ┌──────────────────────────────────┐
│   Whisper ASR        │      │   Speaker Recognition Pipeline    │
│   (Transcription)    │      │  ┌────────────────────────────┐  │
└──────────┬───────────┘      │  │ ECAPA-TDNN (Baseline)      │  │
           │                  │  ├────────────────────────────┤  │
           │                  │  │ WavLM-TDNN (Robustness)    │  │
           │                  │  ├────────────────────────────┤  │
           │                  │  │ ResNet34 (Efficiency)      │  │
           │                  │  └────────────────────────────┘  │
           │                  └──────────────┬────────────────────┘
           │                                 │
           ▼                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Fusion Layer                                  │
│         (Attention-based weighting + Confidence scoring)         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Enhanced Output                                  │
│    (Transcript + Speaker ID + Gender + Confidence Scores)       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementation Details

#### Multi-Model Speaker Recognition Module
```python
# whisperx/multi_model_speaker.py
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from speechbrain.pretrained import EncoderClassifier
import numpy as np
from dataclasses import dataclass

@dataclass
class SpeakerEmbedding:
    """Container for speaker embeddings from multiple models"""
    ecapa: torch.Tensor
    wavlm: Optional[torch.Tensor] = None
    resnet: Optional[torch.Tensor] = None
    fusion: Optional[torch.Tensor] = None

class MultiModelSpeakerRecognition:
    def __init__(self, 
                 models: List[str] = ["ecapa", "wavlm", "resnet"],
                 device: str = "cuda",
                 fusion_method: str = "attention"):
        
        self.device = device
        self.models = {}
        self.fusion_method = fusion_method
        
        # Initialize models
        if "ecapa" in models:
            self.models["ecapa"] = self._load_ecapa()
        if "wavlm" in models:
            self.models["wavlm"] = self._load_wavlm()
        if "resnet" in models:
            self.models["resnet"] = self._load_resnet()
        
        # Initialize fusion layer
        self.fusion_layer = self._init_fusion_layer()
        
        # Speaker database for identification
        self.speaker_database = {}
        self.speaker_threshold = 0.7
    
    def _load_ecapa(self):
        """Load ECAPA-TDNN model"""
        return EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa",
            run_opts={"device": self.device}
        )
    
    def _load_wavlm(self):
        """Load WavLM-based speaker model"""
        return EncoderClassifier.from_hparams(
            source="microsoft/wavlm-base-plus-sv",
            savedir="pretrained_models/wavlm",
            run_opts={"device": self.device}
        )
    
    def _load_resnet(self):
        """Load ResNet34 speaker model"""
        return EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-resnet-voxceleb",
            savedir="pretrained_models/resnet",
            run_opts={"device": self.device}
        )
    
    def _init_fusion_layer(self):
        """Initialize attention-based fusion layer"""
        class AttentionFusion(nn.Module):
            def __init__(self, input_dims: Dict[str, int], output_dim: int = 256):
                super().__init__()
                
                # Project each embedding to common dimension
                self.projections = nn.ModuleDict({
                    name: nn.Linear(dim, output_dim)
                    for name, dim in input_dims.items()
                })
                
                # Attention mechanism
                self.attention = nn.MultiheadAttention(
                    embed_dim=output_dim,
                    num_heads=8,
                    batch_first=True
                )
                
                # Output projection
                self.output_proj = nn.Sequential(
                    nn.Linear(output_dim, output_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(output_dim, output_dim)
                )
                
                # Confidence estimation
                self.confidence_head = nn.Sequential(
                    nn.Linear(output_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, embeddings: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
                # Project embeddings
                projected = []
                for name, embedding in embeddings.items():
                    if name in self.projections:
                        proj = self.projections[name](embedding)
                        projected.append(proj.unsqueeze(1))
                
                # Stack for attention
                stacked = torch.cat(projected, dim=1)  # [batch, num_models, dim]
                
                # Apply attention
                attended, _ = self.attention(stacked, stacked, stacked)
                
                # Pool and project
                pooled = attended.mean(dim=1)  # [batch, dim]
                output = self.output_proj(pooled)
                
                # Estimate confidence
                confidence = self.confidence_head(output)
                
                return output, confidence
        
        # Define input dimensions for each model
        input_dims = {
            "ecapa": 192,
            "wavlm": 256,
            "resnet": 256
        }
        
        fusion = AttentionFusion(input_dims).to(self.device)
        # Load pre-trained fusion weights if available
        # fusion.load_state_dict(torch.load("models/fusion_layer.pth"))
        
        return fusion
    
    def extract_embeddings(self, audio: torch.Tensor) -> SpeakerEmbedding:
        """Extract embeddings from all models"""
        embeddings = {}
        
        with torch.no_grad():
            if "ecapa" in self.models:
                embeddings["ecapa"] = self.models["ecapa"].encode_batch(audio)
            
            if "wavlm" in self.models:
                embeddings["wavlm"] = self.models["wavlm"].encode_batch(audio)
            
            if "resnet" in self.models:
                embeddings["resnet"] = self.models["resnet"].encode_batch(audio)
        
        # Apply fusion if multiple models
        if len(embeddings) > 1:
            fusion_emb, confidence = self.fusion_layer(embeddings)
            return SpeakerEmbedding(
                ecapa=embeddings.get("ecapa"),
                wavlm=embeddings.get("wavlm"),
                resnet=embeddings.get("resnet"),
                fusion=fusion_emb
            )
        else:
            # Single model case
            return SpeakerEmbedding(
                ecapa=embeddings.get("ecapa"),
                wavlm=embeddings.get("wavlm"),
                resnet=embeddings.get("resnet")
            )
    
    def identify_speaker(self, embedding: torch.Tensor) -> Tuple[str, float]:
        """Identify speaker from embedding using cosine similarity"""
        if not self.speaker_database:
            return "Unknown", 0.0
        
        max_similarity = -1
        identified_speaker = "Unknown"
        
        for speaker_id, stored_embedding in self.speaker_database.items():
            similarity = torch.nn.functional.cosine_similarity(
                embedding.unsqueeze(0),
                stored_embedding.unsqueeze(0)
            ).item()
            
            if similarity > max_similarity:
                max_similarity = similarity
                identified_speaker = speaker_id
        
        if max_similarity < self.speaker_threshold:
            # New speaker
            new_id = f"Speaker_{len(self.speaker_database):02d}"
            self.speaker_database[new_id] = embedding
            return new_id, max_similarity
        
        return identified_speaker, max_similarity
    
    def cluster_speakers(self, embeddings: List[torch.Tensor], 
                        method: str = "spectral") -> List[int]:
        """Cluster speaker embeddings"""
        from sklearn.cluster import SpectralClustering, DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        # Convert to numpy
        X = torch.stack(embeddings).cpu().numpy()
        
        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method == "spectral":
            # Spectral clustering
            clustering = SpectralClustering(
                n_clusters=None,
                affinity='nearest_neighbors',
                n_neighbors=5,
                assign_labels='discretize'
            )
        else:  # DBSCAN
            clustering = DBSCAN(
                eps=0.5,
                min_samples=2,
                metric='cosine'
            )
        
        labels = clustering.fit_predict(X_scaled)
        return labels.tolist()
```

#### Integration with WhisperX Pipeline
```python
# Modify whisperx/transcribe.py
from whisperx.multi_model_speaker import MultiModelSpeakerRecognition

def transcribe_with_speaker_recognition(
    audio_file: str,
    model_name: str = "large-v2",
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 16,
    language: Optional[str] = None,
    enable_speaker_recognition: bool = True,
    speaker_models: List[str] = ["ecapa", "wavlm"],
    **kwargs
):
    """
    Enhanced transcription with multi-model speaker recognition
    """
    # Load WhisperX model
    model = whisperx.load_model(
        model_name, 
        device, 
        compute_type=compute_type,
        language=language
    )
    
    # Transcribe audio
    result = model.transcribe(
        audio_file, 
        batch_size=batch_size,
        **kwargs
    )
    
    # Load alignment model
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], 
        device=device
    )
    
    # Align whisper output
    result = whisperx.align(
        result["segments"], 
        model_a, 
        metadata, 
        audio_file, 
        device
    )
    
    # Diarization
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=kwargs.get("hf_token"), 
        device=device
    )
    diarize_segments = diarize_model(audio_file)
    
    # Enhanced speaker recognition
    if enable_speaker_recognition:
        speaker_model = MultiModelSpeakerRecognition(
            models=speaker_models,
            device=device
        )
        
        # Process each segment
        waveform, sample_rate = torchaudio.load(audio_file)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        enhanced_segments = []
        for segment in diarize_segments["segments"]:
            # Extract audio segment
            start_sample = int(segment["start"] * 16000)
            end_sample = int(segment["end"] * 16000)
            audio_segment = waveform[:, start_sample:end_sample]
            
            # Extract embeddings
            speaker_embedding = speaker_model.extract_embeddings(audio_segment)
            
            # Use fusion embedding if available
            embedding = speaker_embedding.fusion if speaker_embedding.fusion is not None \
                       else speaker_embedding.ecapa
            
            # Identify speaker
            speaker_id, confidence = speaker_model.identify_speaker(embedding)
            
            # Update segment
            segment["speaker_id"] = speaker_id
            segment["speaker_confidence"] = confidence
            segment["embeddings"] = {
                "ecapa": speaker_embedding.ecapa.cpu().numpy().tolist() 
                        if speaker_embedding.ecapa is not None else None,
                "wavlm": speaker_embedding.wavlm.cpu().numpy().tolist() 
                        if speaker_embedding.wavlm is not None else None,
                "fusion": speaker_embedding.fusion.cpu().numpy().tolist() 
                         if speaker_embedding.fusion is not None else None
            }
            
            enhanced_segments.append(segment)
        
        diarize_segments["segments"] = enhanced_segments
    
    # Assign speaker labels
    result = whisperx.assign_word_speakers(
        diarize_segments, 
        result
    )
    
    return result
```

### 2.3 Performance Optimization Strategies

#### GPU Memory Management
```python
# whisperx/optimization/memory_manager.py
import torch
import gc
from contextlib import contextmanager
from typing import Optional

class GPUMemoryManager:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.reserved_memory = 0.1  # Reserve 10% of GPU memory
        
    @contextmanager
    def managed_inference(self):
        """Context manager for memory-efficient inference"""
        try:
            # Clear cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            yield
            
        finally:
            # Clean up after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    def get_optimal_batch_size(self, model_size: str = "large") -> int:
        """Calculate optimal batch size based on available GPU memory"""
        if not torch.cuda.is_available():
            return 1
        
        # Get GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved = total_memory * self.reserved_memory
        available = total_memory - reserved
        
        # Estimate memory requirements per sample
        memory_per_sample = {
            "base": 1.5e9,    # 1.5 GB
            "small": 2.0e9,   # 2.0 GB
            "medium": 3.0e9,  # 3.0 GB
            "large": 4.0e9,   # 4.0 GB
            "large-v2": 4.5e9 # 4.5 GB
        }
        
        required = memory_per_sample.get(model_size, 4.0e9)
        optimal_batch = int(available / required)
        
        return max(1, optimal_batch)
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply optimization techniques to model"""
        # Enable mixed precision
        model = model.half()
        
        # Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Compile model with torch.compile (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode="reduce-overhead")
        
        return model

# Usage example
memory_manager = GPUMemoryManager()

with memory_manager.managed_inference():
    # Run inference
    batch_size = memory_manager.get_optimal_batch_size("large-v2")
    result = transcribe_with_speaker_recognition(
        audio_file="audio.wav",
        batch_size=batch_size
    )
```

#### Parallel Processing Pipeline
```python
# whisperx/optimization/parallel_pipeline.py
import asyncio
import concurrent.futures
from typing import List, Dict, Any
import multiprocessing as mp

class ParallelProcessingPipeline:
    def __init__(self, 
                 num_workers: Optional[int] = None,
                 use_gpu_pool: bool = True):
        
        self.num_workers = num_workers or mp.cpu_count()
        self.use_gpu_pool = use_gpu_pool
        
        # Create process pool
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers
        )
        
        # GPU assignment for workers
        if use_gpu_pool and torch.cuda.device_count() > 1:
            self.gpu_assignments = {
                i: i % torch.cuda.device_count() 
                for i in range(self.num_workers)
            }
        else:
            self.gpu_assignments = {i: 0 for i in range(self.num_workers)}
    
    async def process_batch_async(self, 
                                 audio_files: List[str],
                                 model_config: Dict[str, Any]) -> List[Dict]:
        """Process multiple audio files in parallel"""
        loop = asyncio.get_event_loop()
        
        # Create tasks for each file
        tasks = []
        for i, audio_file in enumerate(audio_files):
            worker_id = i % self.num_workers
            gpu_id = self.gpu_assignments[worker_id]
            
            task = loop.run_in_executor(
                self.executor,
                self._process_single_file,
                audio_file,
                model_config,
                gpu_id
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        return results
    
    def _process_single_file(self, 
                           audio_file: str,
                           model_config: Dict[str, Any],
                           gpu_id: int) -> Dict:
        """Process single file on assigned GPU"""
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Import here to ensure correct GPU assignment
        import whisperx
        
        # Process file
        result = whisperx.transcribe_with_speaker_recognition(
            audio_file=audio_file,
            device=f"cuda:{gpu_id}",
            **model_config
        )
        
        return result
    
    def process_directory(self, 
                         input_dir: str,
                         output_dir: str,
                         model_config: Dict[str, Any],
                         file_pattern: str = "*.wav"):
        """Process all audio files in directory"""
        from pathlib import Path
        import json
        
        # Find all audio files
        audio_files = list(Path(input_dir).glob(file_pattern))
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process in batches
        batch_size = self.num_workers * 2
        results = []
        
        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i:i+batch_size]
            
            # Process batch
            batch_results = asyncio.run(
                self.process_batch_async(
                    [str(f) for f in batch],
                    model_config
                )
            )
            
            # Save results
            for audio_file, result in zip(batch, batch_results):
                output_file = Path(output_dir) / f"{audio_file.stem}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                results.append({
                    "input": str(audio_file),
                    "output": str(output_file),
                    "status": "completed"
                })
        
        return results
```

### 2.4 Testing Framework

#### Integration Tests
```python
# tests/test_phase2_integration.py
import pytest
import torch
import numpy as np
from pathlib import Path
import json

class TestPhase2Integration:
    @pytest.fixture
    def test_audio_dir(self):
        """Create test audio files"""
        test_dir = Path("tests/data/phase2")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate test audio files
        sample_rate = 16000
        duration = 10.0
        
        for i in range(3):
            # Generate audio with different characteristics
            t = np.linspace(0, duration, int(sample_rate * duration))
            frequency = 200 + i * 100  # Different frequencies
            audio = np.sin(2 * np.pi * frequency * t)
            
            # Add noise
            noise = np.random.normal(0, 0.01, audio.shape)
            audio += noise
            
            # Save
            import soundfile as sf
            filename = test_dir / f"test_speaker_{i}.wav"
            sf.write(filename, audio, sample_rate)
        
        return test_dir
    
    def test_multi_model_extraction(self):
        """Test multi-model embedding extraction"""
        from whisperx.multi_model_speaker import MultiModelSpeakerRecognition
        
        # Initialize with multiple models
        speaker_model = MultiModelSpeakerRecognition(
            models=["ecapa", "wavlm"],
            device="cpu"  # Use CPU for testing
        )
        
        # Create test audio
        audio = torch.randn(1, 16000)  # 1 second of audio
        
        # Extract embeddings
        embeddings = speaker_model.extract_embeddings(audio)
        
        # Verify embeddings
        assert embeddings.ecapa is not None
        assert embeddings.wavlm is not None
        assert embeddings.fusion is not None
        
        # Check dimensions
        assert embeddings.ecapa.shape[-1] == 192  # ECAPA dimension
        assert embeddings.fusion.shape[-1] == 256  # Fusion dimension
    
    def test_speaker_identification(self):
        """Test speaker identification and clustering"""
        from whisperx.multi_model_speaker import MultiModelSpeakerRecognition
        
        speaker_model = MultiModelSpeakerRecognition(
            models=["ecapa"],
            device="cpu"
        )
        
        # Generate embeddings for different "speakers"
        embeddings = []
        for i in range(5):
            # Simulate different speakers with different embeddings
            if i < 2:
                # First two are same speaker
                base = torch.randn(192)
                noise = torch.randn(192) * 0.1
                embedding = base + noise
            else:
                # Others are different speakers
                embedding = torch.randn(192)
            
            embeddings.append(embedding)
        
        # Test clustering
        labels = speaker_model.cluster_speakers(embeddings)
        
        # Verify clustering results
        assert len(labels) == 5
        assert labels[0] == labels[1]  # Same speaker
        assert len(set(labels)) >= 3    # At least 3 different speakers
    
    def test_parallel_processing(self, test_audio_dir):
        """Test parallel processing pipeline"""
        from whisperx.optimization.parallel_pipeline import ParallelProcessingPipeline
        
        pipeline = ParallelProcessingPipeline(num_workers=2)
        
        model_config = {
            "model_name": "base",  # Use smaller model for testing
            "compute_type": "int8",
            "enable_speaker_recognition": True,
            "speaker_models": ["ecapa"]
        }
        
        # Process test directory
        output_dir = test_audio_dir / "output"
        results = pipeline.process_directory(
            str(test_audio_dir),
            str(output_dir),
            model_config,
            file_pattern="*.wav"
        )
        
        # Verify results
        assert len(results) >= 3
        assert all(r["status"] == "completed" for r in results)
        
        # Check output files
        output_files = list(output_dir.glob("*.json"))
        assert len(output_files) >= 3
        
        # Verify JSON structure
        with open(output_files[0]) as f:
            data = json.load(f)
            assert "segments" in data
            assert "language" in data
    
    def test_memory_management(self):
        """Test GPU memory management"""
        from whisperx.optimization.memory_manager import GPUMemoryManager
        
        manager = GPUMemoryManager()
        
        # Test batch size calculation
        batch_size = manager.get_optimal_batch_size("large")
        assert batch_size >= 1
        
        # Test memory context manager
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        with manager.managed_inference():
            # Simulate memory allocation
            if torch.cuda.is_available():
                dummy = torch.randn(1000, 1000).cuda()
        
        # Memory should be cleared after context
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        assert final_memory <= initial_memory + 1e6  # Allow small overhead
```

#### Performance Benchmarks
```python
# benchmarks/benchmark_phase2.py
import time
import psutil
import GPUtil
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class Phase2Benchmarker:
    def __init__(self):
        self.results = {
            "model_combination": [],
            "processing_time": [],
            "accuracy": [],
            "memory_usage": [],
            "gpu_utilization": []
        }
    
    def benchmark_model_combinations(self, test_audio: str):
        """Benchmark different model combinations"""
        model_combinations = [
            ["ecapa"],
            ["wavlm"],
            ["ecapa", "wavlm"],
            ["ecapa", "resnet"],
            ["ecapa", "wavlm", "resnet"]
        ]
        
        for models in model_combinations:
            print(f"Benchmarking: {models}")
            
            # Initialize model
            speaker_model = MultiModelSpeakerRecognition(
                models=models,
                device="cuda"
            )
            
            # Measure performance
            start_time = time.time()
            initial_memory = torch.cuda.memory_allocated()
            
            # Process audio
            result = transcribe_with_speaker_recognition(
                audio_file=test_audio,
                speaker_models=models
            )
            
            processing_time = time.time() - start_time
            memory_usage = (torch.cuda.memory_allocated() - initial_memory) / 1024**2
            
            # Get GPU utilization
            gpus = GPUtil.getGPUs()
            gpu_util = gpus[0].load * 100 if gpus else 0
            
            # Calculate accuracy (would need ground truth in real scenario)
            accuracy = self._calculate_accuracy(result)
            
            # Store results
            self.results["model_combination"].append("-".join(models))
            self.results["processing_time"].append(processing_time)
            self.results["accuracy"].append(accuracy)
            self.results["memory_usage"].append(memory_usage)
            self.results["gpu_utilization"].append(gpu_util)
    
    def _calculate_accuracy(self, result: Dict) -> float:
        """Calculate accuracy metrics"""
        # In real implementation, compare with ground truth
        # For now, return mock accuracy
        return np.random.uniform(0.85, 0.95)
    
    def plot_results(self, save_path: str = "benchmarks/phase2_results.png"):
        """Plot benchmark results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Processing time
        sns.barplot(
            x="model_combination", 
            y="processing_time",
            data=self.results,
            ax=axes[0, 0]
        )
        axes[0, 0].set_title("Processing Time by Model Combination")
        axes[0, 0].set_xlabel("Model Combination")
        axes[0, 0].set_ylabel("Time (seconds)")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Accuracy
        sns.barplot(
            x="model_combination",
            y="accuracy",
            data=self.results,
            ax=axes[0, 1]
        )
        axes[0, 1].set_title("Accuracy by Model Combination")
        axes[0, 1].set_xlabel("Model Combination")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Memory usage
        sns.barplot(
            x="model_combination",
            y="memory_usage",
            data=self.results,
            ax=axes[1, 0]
        )
        axes[1, 0].set_title("Memory Usage by Model Combination")
        axes[1, 0].set_xlabel("Model Combination")
        axes[1, 0].set_ylabel("Memory (MB)")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # GPU utilization
        sns.barplot(
            x="model_combination",
            y="gpu_utilization",
            data=self.results,
            ax=axes[1, 1]
        )
        axes[1, 1].set_title("GPU Utilization by Model Combination")
        axes[1, 1].set_xlabel("Model Combination")
        axes[1, 1].set_ylabel("GPU Utilization (%)")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path)
        
        return self.results
    
    def generate_report(self, output_path: str = "benchmarks/phase2_report.md"):
        """Generate comprehensive benchmark report"""
        report = f"""# Phase 2 Benchmark Report

## Executive Summary
Benchmarked {len(self.results['model_combination'])} different model combinations
for speaker recognition enhancement in WhisperX.

## Results Summary

| Model Combination | Processing Time (s) | Accuracy | Memory (MB) | GPU Util (%) |
|------------------|-------------------|----------|-------------|--------------|
"""
        
        for i in range(len(self.results['model_combination'])):
            report += f"| {self.results['model_combination'][i]} | "
            report += f"{self.results['processing_time'][i]:.2f} | "
            report += f"{self.results['accuracy'][i]:.3f} | "
            report += f"{self.results['memory_usage'][i]:.1f} | "
            report += f"{self.results['gpu_utilization'][i]:.1f} |\n"
        
        report += f"""
## Recommendations

1. **Best Performance**: {self._get_best_model('processing_time', min)}
2. **Best Accuracy**: {self._get_best_model('accuracy', max)}
3. **Most Efficient**: {self._get_best_balanced_model()}

## Hardware Requirements

- **Minimum**: NVIDIA GPU with 6GB VRAM (GTX 1060)
- **Recommended**: NVIDIA GPU with 12GB VRAM (RTX 3060)
- **Optimal**: NVIDIA GPU with 24GB VRAM (RTX 3090/4090)
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
    
    def _get_best_model(self, metric: str, func):
        """Get best model for specific metric"""
        idx = func(range(len(self.results[metric])), 
                  key=lambda i: self.results[metric][i])
        return self.results['model_combination'][idx]
    
    def _get_best_balanced_model(self):
        """Get best balanced model (accuracy vs performance)"""
        # Normalize metrics
        norm_time = np.array(self.results['processing_time'])
        norm_time = (norm_time - norm_time.min()) / (norm_time.max() - norm_time.min())
        
        norm_acc = np.array(self.results['accuracy'])
        norm_acc = (norm_acc - norm_acc.min()) / (norm_acc.max() - norm_acc.min())
        
        # Combined score (lower is better)
        scores = norm_time - norm_acc
        best_idx = np.argmin(scores)
        
        return self.results['model_combination'][best_idx]

# Run benchmarks
if __name__ == "__main__":
    benchmarker = Phase2Benchmarker()
    benchmarker.benchmark_model_combinations("test_audio.wav")
    benchmarker.plot_results()
    benchmarker.generate_report()
```

### 2.5 Production Deployment

#### Kubernetes Configuration
```yaml
# k8s/phase2-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: whisperx-phase2
  namespace: whisperx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: whisperx-phase2
  template:
    metadata:
      labels:
        app: whisperx-phase2
    spec:
      nodeSelector:
        accelerator: nvidia-gpu
      containers:
      - name: whisperx
        image: whisperx:phase2-v1.0.0
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: OMP_NUM_THREADS
          value: "4"
        - name: WHISPERX_CACHE_DIR
          value: "/models"
        volumeMounts:
        - name: model-cache
          mountPath: /models
        - name: audio-storage
          mountPath: /audio
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: audio-storage
        persistentVolumeClaim:
          claimName: audio-storage-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: whisperx-phase2-service
  namespace: whisperx
spec:
  selector:
    app: whisperx-phase2
  ports:
  - port: 80
    targetPort: 8080
    name: http
  - port: 9090
    targetPort: 9090
    name: metrics
  type: LoadBalancer
```

#### API Server Implementation
```python
# api/server.py
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
import tempfile
from typing import Optional, List
import redis
import json

app = FastAPI(title="WhisperX Phase 2 API")

# Redis for job queue
redis_client = redis.Redis(host='redis', port=6379, db=0)

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    global speaker_model, whisper_model
    
    # Load models
    speaker_model = MultiModelSpeakerRecognition(
        models=["ecapa", "wavlm"],
        device="cuda"
    )
    
    whisper_model = whisperx.load_model(
        "large-v2",
        device="cuda",
        compute_type="float16"
    )

@app.post("/transcribe")
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = None,
    speaker_models: Optional[List[str]] = ["ecapa", "wavlm"],
    async_mode: bool = False
):
    """Transcribe audio with speaker recognition"""
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    if async_mode:
        # Queue job for background processing
        background_tasks.add_task(
            process_transcription_job,
            job_id,
            tmp_path,
            language,
            speaker_models
        )
        
        return JSONResponse({
            "job_id": job_id,
            "status": "queued",
            "message": "Job queued for processing"
        })
    else:
        # Process synchronously
        result = await process_transcription(
            tmp_path,
            language,
            speaker_models
        )
        
        return JSONResponse({
            "job_id": job_id,
            "status": "completed",
            "result": result
        })

async def process_transcription(
    audio_path: str,
    language: Optional[str],
    speaker_models: List[str]
) -> dict:
    """Process transcription with speaker recognition"""
    
    result = transcribe_with_speaker_recognition(
        audio_file=audio_path,
        model_name="large-v2",
        device="cuda",
        language=language,
        speaker_models=speaker_models
    )
    
    return result

async def process_transcription_job(
    job_id: str,
    audio_path: str,
    language: Optional[str],
    speaker_models: List[str]
):
    """Process transcription job in background"""
    
    # Update job status
    redis_client.hset(
        f"job:{job_id}",
        mapping={
            "status": "processing",
            "audio_path": audio_path
        }
    )
    
    try:
        # Process transcription
        result = await process_transcription(
            audio_path,
            language,
            speaker_models
        )
        
        # Store result
        redis_client.hset(
            f"job:{job_id}",
            mapping={
                "status": "completed",
                "result": json.dumps(result)
            }
        )
        
    except Exception as e:
        # Store error
        redis_client.hset(
            f"job:{job_id}",
            mapping={
                "status": "failed",
                "error": str(e)
            }
        )
    
    finally:
        # Clean up temp file
        import os
        os.unlink(audio_path)

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and result"""
    
    job_data = redis_client.hgetall(f"job:{job_id}")
    
    if not job_data:
        return JSONResponse(
            status_code=404,
            content={"error": "Job not found"}
        )
    
    # Decode Redis data
    response = {
        "job_id": job_id,
        "status": job_data.get(b"status", b"").decode(),
    }
    
    if response["status"] == "completed":
        response["result"] = json.loads(
            job_data.get(b"result", b"{}").decode()
        )
    elif response["status"] == "failed":
        response["error"] = job_data.get(b"error", b"").decode()
    
    return JSONResponse(response)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    # Check if models are loaded
    if 'speaker_model' in globals() and 'whisper_model' in globals():
        return {"status": "ready"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready"}
        )
```

---

## Phase 3: Full Pipeline Restructuring with End-to-End Neural Diarization

### 3.1 EEND Architecture Implementation

#### Core EEND Model
```python
# whisperx/eend/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module for EEND"""
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, heads, seq_len, d_k]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.W_o(context)
        
        return output

class ConformerBlock(nn.Module):
    """Conformer block for enhanced EEND"""
    
    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 4,
                 conv_kernel_size: int = 31,
                 ff_expansion: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        # Feed-forward module 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Convolution module
        self.conv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Conv1d(d_model, d_model * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, conv_kernel_size, 
                     padding=conv_kernel_size // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(dropout)
        )
        
        # Feed-forward module 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First feed-forward
        x = x + 0.5 * self.ff1(x)
        
        # Self-attention
        x = x + self.self_attn(self.attn_norm(x))
        
        # Convolution (need to transpose for Conv1d)
        conv_input = x.transpose(1, 2)
        x = x + self.conv(conv_input).transpose(1, 2)
        
        # Second feed-forward
        x = x + 0.5 * self.ff2(x)
        
        # Final layer norm
        x = self.final_norm(x)
        
        return x

class EEND(nn.Module):
    """End-to-End Neural Diarization Model"""
    
    def __init__(self,
                 input_dim: int = 345,  # 23 mel-bins * 15 context frames
                 hidden_dim: int = 256,
                 num_blocks: int = 4,
                 num_heads: int = 4,
                 max_speakers: int = 20,
                 use_conformer: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_speakers = max_speakers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Encoder blocks
        if use_conformer:
            self.encoder = nn.ModuleList([
                ConformerBlock(hidden_dim, num_heads)
                for _ in range(num_blocks)
            ])
        else:
            # Use transformer blocks
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='relu',
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_blocks
            )
        
        # Speaker detection head
        self.speaker_detection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_speakers),
            nn.Sigmoid()
        )
        
        # Number of speakers estimation
        self.num_speakers_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_speakers + 1)  # 0 to max_speakers
        )
        
    def forward(self, 
                x: torch.Tensor,
                num_speakers: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input features [batch, time, feat_dim]
            num_speakers: Known number of speakers (optional)
            
        Returns:
            Dictionary containing:
            - 'diarization': Speaker activity [batch, time, max_speakers]
            - 'num_speakers': Estimated number of speakers [batch]
        """
        # Input projection
        x = self.input_proj(x)
        
        # Encode
        if isinstance(self.encoder, nn.ModuleList):
            # Conformer blocks
            for block in self.encoder:
                x = block(x)
        else:
            # Transformer encoder
            x = self.encoder(x)
        
        # Speaker detection
        diarization = self.speaker_detection(x)
        
        # Estimate number of speakers if not provided
        if num_speakers is None:
            # Global pooling
            pooled = x.mean(dim=1)  # [batch, hidden_dim]
            num_speakers_logits = self.num_speakers_head(pooled)
            num_speakers_pred = num_speakers_logits.argmax(dim=-1)
        else:
            num_speakers_pred = torch.tensor([num_speakers] * x.size(0))
        
        # Mask out speakers beyond estimated number
        batch_size, seq_len, _ = diarization.shape
        for b in range(batch_size):
            n_spk = num_speakers_pred[b].item()
            if n_spk < self.max_speakers:
                diarization[b, :, n_spk:] = 0
        
        return {
            'diarization': diarization,
            'num_speakers': num_speakers_pred
        }

class StreamingEEND(nn.Module):
    """Streaming version of EEND for real-time processing"""
    
    def __init__(self,
                 base_model: EEND,
                 chunk_size: int = 100,  # 1 second chunks at 100Hz
                 context_size: int = 50):  # 0.5 second context
        super().__init__()
        
        self.base_model = base_model
        self.chunk_size = chunk_size
        self.context_size = context_size
        
        # Buffer for context
        self.register_buffer('context_buffer', None)
        
    def forward(self, 
                x: torch.Tensor,
                reset: bool = False) -> Dict[str, torch.Tensor]:
        """
        Process audio chunk
        
        Args:
            x: Input chunk [batch, chunk_time, feat_dim]
            reset: Reset context buffer
            
        Returns:
            Diarization results for current chunk
        """
        if reset or self.context_buffer is None:
            self.context_buffer = torch.zeros(
                x.size(0), self.context_size, x.size(2),
                device=x.device, dtype=x.dtype
            )
        
        # Concatenate context with current chunk
        x_with_context = torch.cat([self.context_buffer, x], dim=1)
        
        # Process through base model
        results = self.base_model(x_with_context)
        
        # Extract results for current chunk (excluding context)
        chunk_results = {
            'diarization': results['diarization'][:, self.context_size:],
            'num_speakers': results['num_speakers']
        }
        
        # Update context buffer
        self.context_buffer = x[:, -self.context_size:].detach()
        
        return chunk_results
```

#### Integration with WhisperX
```python
# whisperx/eend/integration.py
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import whisperx
from whisperx.eend.models import EEND, StreamingEEND

class WhisperXEENDPipeline:
    """Integrated WhisperX + EEND pipeline"""
    
    def __init__(self,
                 whisper_model: str = "large-v2",
                 eend_checkpoint: str = "models/eend_conformer.pt",
                 device: str = "cuda",
                 use_streaming: bool = False):
        
        self.device = device
        self.use_streaming = use_streaming
        
        # Load Whisper model
        self.whisper_model = whisperx.load_model(
            whisper_model,
            device=device,
            compute_type="float16"
        )
        
        # Load EEND model
        self.eend_model = self._load_eend_model(eend_checkpoint)
        
        # Feature extractor for EEND
        self.feature_extractor = self._init_feature_extractor()
        
    def _load_eend_model(self, checkpoint_path: str) -> nn.Module:
        """Load pre-trained EEND model"""
        # Initialize model
        model = EEND(
            input_dim=345,
            hidden_dim=256,
            num_blocks=4,
            num_heads=4,
            max_speakers=8,
            use_conformer=True
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Wrap in streaming module if needed
        if self.use_streaming:
            model = StreamingEEND(model)
        
        return model
    
    def _init_feature_extractor(self):
        """Initialize feature extractor for EEND"""
        import torchaudio.transforms as T
        
        # Mel-spectrogram extractor
        return T.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,  # 10ms hop
            n_mels=23,
            f_min=20,
            f_max=8000
        ).to(self.device)
    
    def extract_features(self, 
                        audio: torch.Tensor,
                        context_frames: int = 15) -> torch.Tensor:
        """Extract features for EEND"""
        # Compute mel-spectrogram
        mel_spec = self.feature_extractor(audio)
        
        # Convert to log scale
        log_mel = torch.log(mel_spec + 1e-8)
        
        # Transpose to [time, mel_bins]
        log_mel = log_mel.transpose(1, 2)
        
        # Add context frames
        batch_size, time_steps, mel_bins = log_mel.shape
        
        # Pad for context
        padding = context_frames // 2
        padded = F.pad(log_mel, (0, 0, padding, padding), mode='reflect')
        
        # Extract context windows
        features = []
        for t in range(time_steps):
            context_window = padded[:, t:t+context_frames, :]
            # Flatten context window
            features.append(context_window.reshape(batch_size, -1))
        
        features = torch.stack(features, dim=1)  # [batch, time, feat_dim]
        
        return features
    
    def transcribe_and_diarize(self,
                              audio_path: str,
                              num_speakers: Optional[int] = None,
                              language: Optional[str] = None) -> Dict:
        """
        Transcribe and diarize audio using integrated pipeline
        
        Args:
            audio_path: Path to audio file
            num_speakers: Known number of speakers (optional)
            language: Language code (optional)
            
        Returns:
            Dictionary with transcription and diarization results
        """
        # Load audio
        audio, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Transcribe with Whisper
        transcription = self.whisper_model.transcribe(
            audio_path,
            language=language
        )
        
        # Extract features for EEND
        features = self.extract_features(audio)
        
        # Run EEND
        with torch.no_grad():
            eend_output = self.eend_model(
                features.unsqueeze(0),  # Add batch dimension
                num_speakers=num_speakers
            )
        
        # Post-process diarization
        diarization = eend_output['diarization'].squeeze(0).cpu().numpy()
        num_speakers_est = eend_output['num_speakers'].item()
        
        # Convert frame-level to time-based diarization
        frame_duration = 0.01  # 10ms per frame
        time_diarization = self._frames_to_time(diarization, frame_duration)
        
        # Align transcription with diarization
        aligned_result = self._align_transcription_diarization(
            transcription,
            time_diarization,
            num_speakers_est
        )
        
        return aligned_result
    
    def _frames_to_time(self,
                       diarization: np.ndarray,
                       frame_duration: float) -> List[Dict]:
        """Convert frame-level diarization to time segments"""
        segments = []
        num_frames, num_speakers = diarization.shape
        
        # Threshold for voice activity
        threshold = 0.5
        
        for speaker_idx in range(num_speakers):
            speaker_activity = diarization[:, speaker_idx] > threshold
            
            # Find continuous segments
            in_segment = False
            start_frame = 0
            
            for frame_idx in range(num_frames):
                if speaker_activity[frame_idx] and not in_segment:
                    # Start of segment
                    start_frame = frame_idx
                    in_segment = True
                elif not speaker_activity[frame_idx] and in_segment:
                    # End of segment
                    segments.append({
                        'speaker': f"SPEAKER_{speaker_idx:02d}",
                        'start': start_frame * frame_duration,
                        'end': frame_idx * frame_duration
                    })
                    in_segment = False
            
            # Handle last segment
            if in_segment:
                segments.append({
                    'speaker': f"SPEAKER_{speaker_idx:02d}",
                    'start': start_frame * frame_duration,
                    'end': num_frames * frame_duration
                })
        
        # Sort by start time
        segments.sort(key=lambda x: x['start'])
        
        return segments
    
    def _align_transcription_diarization(self,
                                       transcription: Dict,
                                       diarization: List[Dict],
                                       num_speakers: int) -> Dict:
        """Align transcription segments with speaker diarization"""
        # Create speaker timeline
        timeline = []
        
        for segment in transcription['segments']:
            # Find overlapping speaker segments
            seg_start = segment['start']
            seg_end = segment['end']
            
            overlapping_speakers = []
            for spk_seg in diarization:
                # Check for overlap
                if (spk_seg['start'] < seg_end and 
                    spk_seg['end'] > seg_start):
                    overlap_duration = min(seg_end, spk_seg['end']) - \
                                     max(seg_start, spk_seg['start'])
                    overlapping_speakers.append({
                        'speaker': spk_seg['speaker'],
                        'overlap': overlap_duration
                    })
            
            # Assign to speaker with maximum overlap
            if overlapping_speakers:
                best_speaker = max(overlapping_speakers, 
                                 key=lambda x: x['overlap'])
                speaker = best_speaker['speaker']
            else:
                speaker = "UNKNOWN"
            
            # Create aligned segment
            timeline.append({
                'start': seg_start,
                'end': seg_end,
                'text': segment['text'],
                'speaker': speaker,
                'words': segment.get('words', [])
            })
        
        return {
            'segments': timeline,
            'language': transcription['language'],
            'num_speakers': num_speakers,
            'diarization': diarization
        }
```

### 3.2 Training Pipeline for EEND

```python
# whisperx/eend/training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Dict, List, Tuple
import wandb

class DiarizationDataset(Dataset):
    """Dataset for EEND training"""
    
    def __init__(self,
                 data_dir: str,
                 split: str = "train",
                 max_speakers: int = 8,
                 chunk_duration: float = 10.0):
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_speakers = max_speakers
        self.chunk_duration = chunk_duration
        
        # Load metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> List[Dict]:
        """Load dataset metadata"""
        import json
        
        metadata_file = self.data_dir / f"{self.split}.json"
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        return metadata
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training sample"""
        sample_info = self.metadata[idx]
        
        # Load audio
        audio, sr = torchaudio.load(
            self.data_dir / sample_info['audio_path']
        )
        
        # Load diarization labels
        labels = self._load_labels(sample_info['label_path'])
        
        # Extract random chunk
        chunk_audio, chunk_labels = self._extract_chunk(
            audio, labels, sr
        )
        
        # Extract features
        features = self._extract_features(chunk_audio)
        
        return {
            'features': features,
            'labels': chunk_labels,
            'num_speakers': len(sample_info['speakers'])
        }
    
    def _load_labels(self, label_path: str) -> torch.Tensor:
        """Load frame-level speaker labels"""
        # Implementation depends on label format
        # Return tensor of shape [time, max_speakers]
        pass
    
    def _extract_chunk(self, 
                      audio: torch.Tensor,
                      labels: torch.Tensor,
                      sample_rate: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract random chunk from audio and labels"""
        chunk_samples = int(self.chunk_duration * sample_rate)
        total_samples = audio.shape[-1]
        
        if total_samples <= chunk_samples:
            # Pad if necessary
            return audio, labels
        
        # Random start position
        start = torch.randint(0, total_samples - chunk_samples, (1,)).item()
        end = start + chunk_samples
        
        # Extract chunk
        audio_chunk = audio[:, start:end]
        
        # Corresponding label frames
        start_frame = int(start / sample_rate * 100)  # 100Hz frame rate
        end_frame = int(end / sample_rate * 100)
        label_chunk = labels[start_frame:end_frame]
        
        return audio_chunk, label_chunk
    
    def _extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features from audio chunk"""
        # Use same feature extraction as inference
        # Return tensor of shape [time, feat_dim]
        pass

class EENDLightningModule(pl.LightningModule):
    """PyTorch Lightning module for EEND training"""
    
    def __init__(self,
                 model_config: Dict,
                 learning_rate: float = 1e-4,
                 warmup_steps: int = 1000):
        super().__init__()
        
        # Initialize model
        self.model = EEND(**model_config)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        
        # Metrics
        self.train_loss = []
        self.val_loss = []
        
    def forward(self, features: torch.Tensor, 
                num_speakers: Optional[int] = None) -> Dict:
        return self.model(features, num_speakers)
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        features = batch['features']
        labels = batch['labels']
        num_speakers = batch['num_speakers']
        
        # Forward pass
        outputs = self(features, num_speakers)
        
        # Compute loss
        diarization_loss = self.criterion(
            outputs['diarization'],
            labels
        )
        
        # Log metrics
        self.log('train_loss', diarization_loss, prog_bar=True)
        
        return diarization_loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        features = batch['features']
        labels = batch['labels']
        num_speakers = batch['num_speakers']
        
        # Forward pass
        outputs = self(features, num_speakers)
        
        # Compute loss
        val_loss = self.criterion(
            outputs['diarization'],
            labels
        )
        
        # Compute metrics
        der = self._compute_der(outputs['diarization'], labels)
        
        # Log metrics
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_der', der, prog_bar=True)
        
        return val_loss
    
    def _compute_der(self, 
                    predictions: torch.Tensor,
                    targets: torch.Tensor,
                    threshold: float = 0.5) -> float:
        """Compute Diarization Error Rate"""
        # Convert to binary
        pred_binary = (predictions > threshold).float()
        
        # Compute confusion matrix elements
        intersection = (pred_binary * targets).sum()
        pred_total = pred_binary.sum()
        target_total = targets.sum()
        
        # DER = (FA + MISS + CONF) / TOTAL
        false_alarm = pred_total - intersection
        missed = target_total - intersection
        
        # Simplified DER (without speaker confusion)
        der = (false_alarm + missed) / (target_total + 1e-8)
        
        return der.item()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                return 1.0
        
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

# Training script
def train_eend_model(config: Dict):
    """Train EEND model"""
    
    # Initialize wandb
    wandb.init(project="whisperx-eend", config=config)
    
    # Create datasets
    train_dataset = DiarizationDataset(
        data_dir=config['data_dir'],
        split='train',
        max_speakers=config['max_speakers']
    )
    
    val_dataset = DiarizationDataset(
        data_dir=config['data_dir'],
        split='val',
        max_speakers=config['max_speakers']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Initialize model
    model = EENDLightningModule(
        model_config=config['model'],
        learning_rate=config['learning_rate']
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu',
        devices=config['num_gpus'],
        strategy='ddp' if config['num_gpus'] > 1 else None,
        precision=16,
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        logger=pl.loggers.WandbLogger(),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath='checkpoints',
                filename='eend-{epoch:02d}-{val_der:.4f}',
                monitor='val_der',
                mode='min',
                save_top_k=3
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_der',
                patience=10,
                mode='min'
            )
        ]
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'config': config
    }, 'models/eend_final.pt')
```

### 3.3 Real-time Streaming Implementation

```python
# whisperx/streaming/server.py
import asyncio
import websockets
import json
import torch
import numpy as np
from collections import deque
from typing import Optional, Dict, Any
import threading
import queue

class StreamingTranscriptionServer:
    """WebSocket server for real-time transcription with diarization"""
    
    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 8765,
                 model_config: Dict[str, Any] = None):
        
        self.host = host
        self.port = port
        
        # Initialize models
        self.pipeline = WhisperXEENDPipeline(
            whisper_model=model_config.get('whisper_model', 'base'),
            eend_checkpoint=model_config.get('eend_checkpoint'),
            device=model_config.get('device', 'cuda'),
            use_streaming=True
        )
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=16000 * 30)  # 30 seconds max
        self.processing_queue = queue.Queue()
        
        # Processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_worker,
            daemon=True
        )
        self.processing_thread.start()
        
        # Results cache
        self.results_cache = deque(maxlen=100)
        
    async def handle_connection(self, websocket, path):
        """Handle WebSocket connection"""
        print(f"Client connected from {websocket.remote_address}")
        
        try:
            async for message in websocket:
                # Parse message
                data = json.loads(message)
                message_type = data.get('type')
                
                if message_type == 'audio':
                    # Handle audio data
                    await self._handle_audio(websocket, data)
                
                elif message_type == 'config':
                    # Update configuration
                    await self._handle_config(websocket, data)
                
                elif message_type == 'control':
                    # Handle control commands
                    await self._handle_control(websocket, data)
                
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected from {websocket.remote_address}")
        except Exception as e:
            print(f"Error handling connection: {e}")
        
    async def _handle_audio(self, websocket, data: Dict):
        """Handle incoming audio data"""
        # Decode audio data (base64)
        import base64
        audio_bytes = base64.b64decode(data['audio'])
        
        # Convert to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Normalize to [-1, 1]
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Add to buffer
        self.audio_buffer.extend(audio_float)
        
        # Process if enough data
        if len(self.audio_buffer) >= 16000:  # 1 second of audio
            # Extract chunk
            chunk = np.array(list(self.audio_buffer)[:16000])
            
            # Add to processing queue
            self.processing_queue.put({
                'audio': chunk,
                'websocket': websocket,
                'timestamp': data.get('timestamp', 0)
            })
    
    async def _handle_config(self, websocket, data: Dict):
        """Handle configuration updates"""
        config = data.get('config', {})
        
        # Update model configuration
        if 'language' in config:
            self.pipeline.language = config['language']
        
        if 'num_speakers' in config:
            self.pipeline.num_speakers = config['num_speakers']
        
        # Send acknowledgment
        await websocket.send(json.dumps({
            'type': 'config_ack',
            'status': 'updated'
        }))
    
    async def _handle_control(self, websocket, data: Dict):
        """Handle control commands"""
        command = data.get('command')
        
        if command == 'reset':
            # Reset buffers and state
            self.audio_buffer.clear()
            self.results_cache.clear()
            
            # Reset EEND context
            if hasattr(self.pipeline.eend_model, 'context_buffer'):
                self.pipeline.eend_model.context_buffer = None
            
            await websocket.send(json.dumps({
                'type': 'control_ack',
                'command': 'reset',
                'status': 'completed'
            }))
        
        elif command == 'get_results':
            # Send accumulated results
            results = list(self.results_cache)
            await websocket.send(json.dumps({
                'type': 'results',
                'data': results
            }))
    
    def _processing_worker(self):
        """Background worker for audio processing"""
        while True:
            try:
                # Get item from queue
                item = self.processing_queue.get(timeout=1.0)
                
                if item is None:
                    break
                
                # Process audio chunk
                audio_chunk = torch.tensor(item['audio']).unsqueeze(0)
                websocket = item['websocket']
                timestamp = item['timestamp']
                
                # Extract features
                features = self.pipeline.extract_features(audio_chunk)
                
                # Run streaming EEND
                with torch.no_grad():
                    diarization_result = self.pipeline.eend_model(features)
                
                # Run ASR on chunk
                transcription = self._transcribe_chunk(audio_chunk)
                
                # Combine results
                result = self._combine_results(
                    transcription,
                    diarization_result,
                    timestamp
                )
                
                # Cache result
                self.results_cache.append(result)
                
                # Send to client
                asyncio.run(websocket.send(json.dumps({
                    'type': 'transcription',
                    'data': result
                })))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def _transcribe_chunk(self, audio_chunk: torch.Tensor) -> Dict:
        """Transcribe audio chunk with Whisper"""
        # Convert to appropriate format
        audio_np = audio_chunk.squeeze().cpu().numpy()
        
        # Use Whisper for transcription
        result = self.pipeline.whisper_model.transcribe(
            audio_np,
            language=getattr(self.pipeline, 'language', None)
        )
        
        return result
    
    def _combine_results(self,
                        transcription: Dict,
                        diarization: Dict,
                        timestamp: float) -> Dict:
        """Combine transcription and diarization results"""
        segments = []
        
        for segment in transcription.get('segments', []):
            # Find corresponding speaker from diarization
            seg_start = segment['start'] + timestamp
            seg_end = segment['end'] + timestamp
            
            # Get speaker probabilities for this segment
            start_frame = int(segment['start'] * 100)
            end_frame = int(segment['end'] * 100)
            
            speaker_probs = diarization['diarization'][
                0, start_frame:end_frame
            ].mean(dim=0)
            
            # Assign to most likely speaker
            speaker_idx = speaker_probs.argmax().item()
            
            segments.append({
                'start': seg_start,
                'end': seg_end,
                'text': segment['text'],
                'speaker': f"SPEAKER_{speaker_idx:02d}",
                'confidence': speaker_probs[speaker_idx].item()
            })
        
        return {
            'timestamp': timestamp,
            'segments': segments,
            'num_speakers': diarization['num_speakers'].item()
        }
    
    async def start_server(self):
        """Start WebSocket server"""
        async with websockets.serve(
            self.handle_connection,
            self.host,
            self.port
        ):
            print(f"Streaming server started on {self.host}:{self.port}")
            await asyncio.Future()  # Run forever

# Client example
class StreamingTranscriptionClient:
    """Client for streaming transcription"""
    
    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        
    async def connect(self):
        """Connect to server"""
        self.websocket = await websockets.connect(self.server_url)
        print(f"Connected to {self.server_url}")
    
    async def send_audio_file(self, audio_path: str, chunk_duration: float = 1.0):
        """Send audio file in chunks"""
        import soundfile as sf
        
        # Load audio
        audio, sample_rate = sf.read(audio_path)
        
        # Resample if necessary
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=16000
            )
            sample_rate = 16000
        
        # Send configuration
        await self.websocket.send(json.dumps({
            'type': 'config',
            'config': {
                'language': 'en',
                'num_speakers': None  # Auto-detect
            }
        }))
        
        # Send audio in chunks
        chunk_samples = int(chunk_duration * sample_rate)
        timestamp = 0.0
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i+chunk_samples]
            
            # Convert to int16
            chunk_int16 = (chunk * 32768).astype(np.int16)
            
            # Encode as base64
            import base64
            chunk_bytes = chunk_int16.tobytes()
            chunk_base64 = base64.b64encode(chunk_bytes).decode()
            
            # Send chunk
            await self.websocket.send(json.dumps({
                'type': 'audio',
                'audio': chunk_base64,
                'timestamp': timestamp
            }))
            
            timestamp += chunk_duration
            
            # Simulate real-time streaming
            await asyncio.sleep(chunk_duration)
        
        # Get final results
        await self.websocket.send(json.dumps({
            'type': 'control',
            'command': 'get_results'
        }))
        
        # Receive results
        response = await self.websocket.recv()
        results = json.loads(response)
        
        return results
    
    async def close(self):
        """Close connection"""
        if self.websocket:
            await self.websocket.close()
```

### 3.4 Performance Benchmarks and Optimization

```python
# benchmarks/benchmark_phase3.py
import time
import torch
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    model_name: str
    audio_duration: float
    processing_time: float
    der: float
    wer: float
    memory_usage: float
    gpu_utilization: float
    latency: float

class Phase3Benchmarker:
    """Comprehensive benchmarking for Phase 3"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.test_datasets = {
            'ami': 'data/ami_test',
            'callhome': 'data/callhome_test',
            'dihard': 'data/dihard_test',
            'voxconverse': 'data/voxconverse_test'
        }
    
    def benchmark_all_models(self):
        """Benchmark all EEND variants"""
        models = [
            {
                'name': 'EEND-Transformer',
                'config': {
                    'use_conformer': False,
                    'num_blocks': 4,
                    'hidden_dim': 256
                }
            },
            {
                'name': 'EEND-Conformer',
                'config': {
                    'use_conformer': True,
                    'num_blocks': 4,
                    'hidden_dim': 256
                }
            },
            {
                'name': 'EEND-Streaming',
                'config': {
                    'use_conformer': True,
                    'num_blocks': 4,
                    'hidden_dim': 256,
                    'streaming': True,
                    'chunk_size': 100
                }
            }
        ]
        
        for model_config in models:
            print(f"Benchmarking {model_config['name']}...")
            
            for dataset_name, dataset_path in self.test_datasets.items():
                result = self.benchmark_model_on_dataset(
                    model_config,
                    dataset_name,
                    dataset_path
                )
                self.results.append(result)
    
    def benchmark_model_on_dataset(self,
                                  model_config: Dict,
                                  dataset_name: str,
                                  dataset_path: str) -> BenchmarkResult:
        """Benchmark single model on dataset"""
        # Initialize pipeline
        pipeline = WhisperXEENDPipeline(
            whisper_model="large-v2",
            eend_checkpoint=f"models/{model_config['name'].lower()}.pt",
            use_streaming=model_config['config'].get('streaming', False)
        )
        
        # Load test files
        test_files = list(Path(dataset_path).glob("*.wav"))
        
        total_duration = 0
        total_processing_time = 0
        der_scores = []
        wer_scores = []
        memory_usage = []
        latencies = []
        
        for audio_file in test_files[:10]:  # Limit for benchmarking
            # Load ground truth
            ground_truth = self._load_ground_truth(audio_file)
            
            # Measure performance
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            # Process file
            result = pipeline.transcribe_and_diarize(str(audio_file))
            
            processing_time = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            # Calculate metrics
            der = self._calculate_der(result, ground_truth)
            wer = self._calculate_wer(result, ground_truth)
            
            # Audio duration
            import librosa
            duration = librosa.get_duration(filename=str(audio_file))
            
            # Latency (for streaming)
            if model_config['config'].get('streaming', False):
                latency = processing_time / duration
            else:
                latency = processing_time
            
            # Accumulate results
            total_duration += duration
            total_processing_time += processing_time
            der_scores.append(der)
            wer_scores.append(wer)
            memory_usage.append(peak_memory)
            latencies.append(latency)
        
        # GPU utilization (mock for example)
        gpu_util = self._get_gpu_utilization()
        
        return BenchmarkResult(
            model_name=f"{model_config['name']}_{dataset_name}",
            audio_duration=total_duration,
            processing_time=total_processing_time,
            der=np.mean(der_scores),
            wer=np.mean(wer_scores),
            memory_usage=np.mean(memory_usage),
            gpu_utilization=gpu_util,
            latency=np.mean(latencies)
        )
    
    def _calculate_der(self, prediction: Dict, ground_truth: Dict) -> float:
        """Calculate Diarization Error Rate"""
        from pyannote.metrics.diarization import DiarizationErrorRate
        
        metric = DiarizationErrorRate()
        
        # Convert to pyannote format
        # Implementation details...
        
        return metric.compute()
    
    def _calculate_wer(self, prediction: Dict, ground_truth: Dict) -> float:
        """Calculate Word Error Rate"""
        from jiwer import wer
        
        # Extract text
        pred_text = " ".join([s['text'] for s in prediction['segments']])
        true_text = ground_truth.get('text', '')
        
        return wer(true_text, pred_text)
    
    def _get_gpu_utilization(self) -> float:
        """Get average GPU utilization"""
        import GPUtil
        
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].load * 100
        return 0.0
    
    def _load_ground_truth(self, audio_file: Path) -> Dict:
        """Load ground truth annotations"""
        # Load corresponding annotation file
        annotation_file = audio_file.with_suffix('.json')
        
        if annotation_file.exists():
            import json
            with open(annotation_file) as f:
                return json.load(f)
        
        return {}
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        # Convert results to DataFrame
        df = pd.DataFrame([
            {
                'Model': r.model_name,
                'Duration (s)': r.audio_duration,
                'Processing Time (s)': r.processing_time,
                'RTF': r.processing_time / r.audio_duration,
                'DER (%)': r.der * 100,
                'WER (%)': r.wer * 100,
                'Memory (GB)': r.memory_usage,
                'GPU (%)': r.gpu_utilization,
                'Latency (s)': r.latency
            }
            for r in self.results
        ])
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # DER by model and dataset
        der_pivot = df.pivot_table(
            values='DER (%)',
            index='Model',
            aggfunc='mean'
        )
        der_pivot.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Diarization Error Rate by Model')
        axes[0, 0].set_ylabel('DER (%)')
        
        # WER comparison
        wer_pivot = df.pivot_table(
            values='WER (%)',
            index='Model',
            aggfunc='mean'
        )
        wer_pivot.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Word Error Rate by Model')
        axes[0, 1].set_ylabel('WER (%)')
        
        # Real-time factor
        rtf_pivot = df.pivot_table(
            values='RTF',
            index='Model',
            aggfunc='mean'
        )
        rtf_pivot.plot(kind='bar', ax=axes[0, 2])
        axes[0, 2].set_title('Real-Time Factor by Model')
        axes[0, 2].set_ylabel('RTF')
        axes[0, 2].axhline(y=1.0, color='r', linestyle='--', label='Real-time')
        
        # Memory usage
        memory_pivot = df.pivot_table(
            values='Memory (GB)',
            index='Model',
            aggfunc='mean'
        )
        memory_pivot.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Memory Usage by Model')
        axes[1, 0].set_ylabel('Memory (GB)')
        
        # GPU utilization
        gpu_pivot = df.pivot_table(
            values='GPU (%)',
            index='Model',
            aggfunc='mean'
        )
        gpu_pivot.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('GPU Utilization by Model')
        axes[1, 1].set_ylabel('GPU (%)')
        
        # Latency (for streaming)
        streaming_df = df[df['Model'].str.contains('Streaming')]
        if not streaming_df.empty:
            streaming_df.plot(
                x='Model',
                y='Latency (s)',
                kind='bar',
                ax=axes[1, 2]
            )
            axes[1, 2].set_title('Streaming Latency')
            axes[1, 2].set_ylabel('Latency (s)')
        
        plt.tight_layout()
        plt.savefig('benchmarks/phase3_results.png')
        
        # Generate markdown report
        report = f"""# Phase 3 Benchmark Report

## Executive Summary

Benchmarked {len(self.results)} model-dataset combinations for end-to-end neural diarization.

## Overall Performance

### Best Models by Metric:
- **Lowest DER**: {df.loc[df['DER (%)'].idxmin(), 'Model']} ({df['DER (%)'].min():.2f}%)
- **Lowest WER**: {df.loc[df['WER (%)'].idxmin(), 'Model']} ({df['WER (%)'].min():.2f}%)
- **Fastest Processing**: {df.loc[df['RTF'].idxmin(), 'Model']} (RTF: {df['RTF'].min():.2f})
- **Most Memory Efficient**: {df.loc[df['Memory (GB)'].idxmin(), 'Model']} ({df['Memory (GB)'].min():.2f} GB)

## Detailed Results

{df.to_markdown(index=False)}

## Legal-Grade Performance Analysis

For legal transcription requirements (DER < 5%, WER < 5%):
- **Qualifying Models**: {', '.join(df[(df['DER (%)'] < 5) & (df['WER (%)'] < 5)]['Model'].unique())}
- **Average Processing Time**: {df[(df['DER (%)'] < 5) & (df['WER (%)'] < 5)]['Processing Time (s)'].mean():.2f}s
- **Average Memory Required**: {df[(df['DER (%)'] < 5) & (df['WER (%)'] < 5)]['Memory (GB)'].mean():.2f} GB

## Recommendations

1. **For Accuracy**: Use EEND-Conformer with full context
2. **For Speed**: Use EEND-Streaming with appropriate chunk size
3. **For Production**: Deploy EEND-Conformer with GPU acceleration

## Hardware Requirements

### Minimum (RTF < 1.0):
- NVIDIA V100 GPU (16GB VRAM)
- 32GB System RAM
- 8-core CPU

### Recommended (RTF < 0.5):
- NVIDIA A100 GPU (40GB VRAM)
- 64GB System RAM
- 16-core CPU

### Optimal (RTF < 0.25):
- 2x NVIDIA A100 GPUs
- 128GB System RAM
- 32-core CPU
"""
        
        with open('benchmarks/phase3_report.md', 'w') as f:
            f.write(report)
        
        return df
```

### 3.5 Production Deployment Configuration

```yaml
# k8s/phase3-production.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: whisperx-production
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: whisperx-config
  namespace: whisperx-production
data:
  config.yaml: |
    whisper:
      model: "large-v2"
      compute_type: "float16"
      device: "cuda"
    eend:
      checkpoint: "/models/eend_conformer_final.pt"
      max_speakers: 8
      streaming:
        enabled: true
        chunk_size: 100
        context_size: 50
    api:
      max_file_size: "500MB"
      timeout: 300
      rate_limit: 100
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: whisperx-eend
  namespace: whisperx-production
spec:
  serviceName: whisperx-eend-service
  replicas: 3
  selector:
    matchLabels:
      app: whisperx-eend
  template:
    metadata:
      labels:
        app: whisperx-eend
    spec:
      containers:
      - name: whisperx
        image: whisperx:phase3-v1.0.0
        resources:
          requests:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: 1
          limits:
            memory: "64Gi"
            cpu: "16"
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: PYTORCH_CUDA_ALLOC_CONF
          value: "max_split_size_mb:512"
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8765
          name: websocket
        - containerPort: 9090
          name: metrics
        volumeMounts:
        - name: models
          mountPath: /models
        - name: config
          mountPath: /config
        - name: cache
          mountPath: /cache
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 120
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - python3
            - -c
            - "import torch; assert torch.cuda.is_available()"
          initialDelaySeconds: 60
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: whisperx-config
  volumeClaimTemplates:
  - metadata:
      name: models
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
  - metadata:
      name: cache
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 200Gi
---
apiVersion: v1
kind: Service
metadata:
  name: whisperx-eend-service
  namespace: whisperx-production
spec:
  selector:
    app: whisperx-eend
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: websocket
    port: 8765
    targetPort: 8765
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: whisperx-eend-hpa
  namespace: whisperx-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: whisperx-eend
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: gpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### 3.6 Monitoring and Observability

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from functools import wraps
import time

# Define metrics
transcription_requests = Counter(
    'whisperx_transcription_requests_total',
    'Total number of transcription requests',
    ['model', 'status']
)

transcription_duration = Histogram(
    'whisperx_transcription_duration_seconds',
    'Transcription processing duration',
    ['model', 'audio_duration_bucket']
)

active_connections = Gauge(
    'whisperx_active_websocket_connections',
    'Number of active WebSocket connections'
)

model_accuracy = Gauge(
    'whisperx_model_accuracy',
    'Model accuracy metrics',
    ['model', 'metric_type']
)

gpu_memory_usage = Gauge(
    'whisperx_gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['gpu_index']
)

# Decorator for timing
def track_transcription(model_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                transcription_requests.labels(
                    model=model_name,
                    status='success'
                ).inc()
                
                # Determine audio duration bucket
                audio_duration = kwargs.get('audio_duration', 0)
                if audio_duration < 60:
                    bucket = 'short'
                elif audio_duration < 300:
                    bucket = 'medium'
                else:
                    bucket = 'long'
                
                transcription_duration.labels(
                    model=model_name,
                    audio_duration_bucket=bucket
                ).observe(time.time() - start_time)
                
                return result
                
            except Exception as e:
                transcription_requests.labels(
                    model=model_name,
                    status='error'
                ).inc()
                raise e
        
        return wrapper
    return decorator

# Metrics endpoint
from flask import Flask, Response

app = Flask(__name__)

@app.route('/metrics')
def metrics():
    # Update GPU metrics
    update_gpu_metrics()
    
    # Generate metrics
    return Response(generate_latest(), mimetype='text/plain')

def update_gpu_metrics():
    """Update GPU memory usage metrics"""
    import GPUtil
    
    gpus = GPUtil.getGPUs()
    for i, gpu in enumerate(gpus):
        gpu_memory_usage.labels(gpu_index=str(i)).set(
            gpu.memoryUsed * 1024 * 1024 * 1024  # Convert to bytes
        )

# Grafana dashboard configuration
GRAFANA_DASHBOARD = {
    "dashboard": {
        "title": "WhisperX EEND Monitoring",
        "panels": [
            {
                "title": "Transcription Request Rate",
                "targets": [{
                    "expr": "rate(whisperx_transcription_requests_total[5m])"
                }]
            },
            {
                "title": "Processing Duration by Model",
                "targets": [{
                    "expr": "histogram_quantile(0.95, whisperx_transcription_duration_seconds)"
                }]
            },
            {
                "title": "GPU Memory Usage",
                "targets": [{
                    "expr": "whisperx_gpu_memory_usage_bytes / 1024 / 1024 / 1024"
                }]
            },
            {
                "title": "Model Accuracy",
                "targets": [{
                    "expr": "whisperx_model_accuracy"
                }]
            }
        ]
    }
}
```

## Final Recommendations and Best Practices

### Architecture Selection Guide

#### For Legal-Grade Transcription
- **Primary**: EEND-Conformer with full WhisperX integration
- **Accuracy Target**: DER < 5%, WER < 5%
- **Hardware**: NVIDIA A100 GPU minimum
- **Configuration**: Non-streaming mode with full context

#### For Real-Time Applications
- **Primary**: Streaming EEND with WebSocket server
- **Latency Target**: < 300ms
- **Hardware**: NVIDIA V100 GPU minimum
- **Configuration**: Chunk size 1s, context 0.5s

#### For High-Volume Processing
- **Primary**: Kubernetes deployment with auto-scaling
- **Throughput Target**: 100+ concurrent sessions
- **Hardware**: Multi-GPU cluster
- **Configuration**: Load balancing with Redis queue

### Implementation Priority

1. **Phase 1 First**: Immediate 20-30% improvement in speaker identification
   - Low complexity, high impact
   - 2-3 weeks implementation

2. **Phase 2 for Robustness**: Multi-model fusion for challenging audio
   - Medium complexity, significant accuracy gains
   - 4-6 weeks implementation

3. **Phase 3 for Excellence**: State-of-the-art performance
   - High complexity, best-in-class results
   - 8-12 weeks implementation

### Quality Assurance Checklist

- [ ] Unit tests for all components (>90% coverage)
- [ ] Integration tests for full pipeline
- [ ] Performance benchmarks on standard datasets
- [ ] Security audit for API endpoints
- [ ] Documentation for all APIs and configurations
- [ ] Monitoring and alerting setup
- [ ] Disaster recovery procedures
- [ ] Compliance verification (GDPR, HIPAA if applicable)

### Continuous Improvement Strategy

1. **Monthly Model Updates**: Retrain on accumulated data
2. **Quarterly Architecture Review**: Evaluate new research
3. **Continuous Monitoring**: Track accuracy degradation
4. **User Feedback Loop**: Implement correction mechanisms
5. **A/B Testing Framework**: Compare model versions

This comprehensive roadmap provides a clear path from current WhisperX v3 capabilities to a state-of-the-art legal-grade transcription system with advanced speaker identification and gender classification.