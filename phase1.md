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