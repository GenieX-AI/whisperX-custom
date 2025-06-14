#!/usr/bin/env python3
"""
Generate WhisperX pipeline diagrams using various Python libraries.
Choose from: matplotlib, plotly, graphviz, or diagrams.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_matplotlib_pipeline():
    """Create pipeline diagram using matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#e1f5fe',
        'processing': '#f3e5f5', 
        'whisper': '#2d2d2d',
        'gender': '#e8f5e8',
        'highlight': '#ffeb3b',
        'output': '#fff3e0'
    }
    
    # Component definitions (x, y, width, height, text, color)
    components = [
        (0.5, 7, 1.5, 0.8, 'Input Audio\n🎵 WAV/MP3', colors['input']),
        (0.5, 5.8, 1.5, 0.8, 'Voice Activity\nDetection', colors['processing']),
        (0.5, 4.6, 1.5, 0.8, 'Cut & Merge\nSegments', colors['processing']),
        (2.5, 4.6, 1.5, 0.8, 'Batch\nProcessing', colors['processing']),
        (4.5, 4.6, 1.5, 0.8, 'Whisper ASR\nTranscription', colors['whisper']),
        (6.5, 4.6, 1.5, 0.8, 'Phoneme\nModel', colors['whisper']),
        (6.5, 3.4, 1.5, 0.8, 'Forced\nAlignment', colors['whisper']),
        (4.5, 3.4, 1.5, 0.8, 'Speaker\nDiarization', colors['gender']),
        (2.5, 3.4, 1.5, 0.8, 'ECAPA-TDNN\nFeatures', colors['gender']),
        (0.5, 3.4, 1.5, 0.8, 'Z-Score\nNormalization', colors['highlight']),
        (0.5, 2.2, 1.5, 0.8, 'Gender\nClassification', colors['gender']),
        (2.5, 1, 3, 1, 'Enhanced Output\nMale_SPEAKER_00\nFemale_SPEAKER_01', colors['output'])
    ]
    
    # Draw components
    for x, y, w, h, text, color in components:
        # Create rounded rectangle
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(box)
        
        # Add text
        text_color = 'white' if color == colors['whisper'] else 'black'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=9, fontweight='bold', color=text_color)
    
    # Draw arrows (start_x, start_y, end_x, end_y)
    arrows = [
        (1.25, 7, 1.25, 6.6),      # Input → VAD
        (1.25, 5.8, 1.25, 5.4),   # VAD → Cut&Merge
        (2, 5, 2.5, 5),            # Cut&Merge → Batch
        (4, 5, 4.5, 5),            # Batch → Whisper
        (6, 5, 6.5, 5),            # Whisper → Phoneme
        (7.25, 4.6, 7.25, 4.2),   # Phoneme → Alignment
        (6.5, 3.8, 6, 3.8),       # Alignment → Diarization
        (4.5, 3.8, 4, 3.8),       # Diarization → ECAPA
        (2.5, 3.8, 2, 3.8),       # ECAPA → Z-Score
        (1.25, 3.4, 1.25, 3),     # Z-Score → Gender
        (1.25, 2.2, 2.5, 1.8),    # Gender → Output
    ]
    
    for start_x, start_y, end_x, end_y in arrows:
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#333'))
    
    # Add title
    ax.text(5, 7.5, 'WhisperX Enhanced Pipeline with Gender Classification', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add performance metrics box
    metrics_text = """Performance Metrics:
• Word Error Rate (WER): 3.4%
• ASR Accuracy: 96.6%
• Diarization Error Rate (DER): 15.6%
• Gender Consistency: 100%
• Gender Confidence: 98.2%
• Processing Speed: 8x real-time"""
    
    ax.text(8, 2, metrics_text, ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/Users/user/Desktop/WhisperX/figures/enhanced_pipeline_matplotlib.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Pipeline diagram saved as: figures/enhanced_pipeline_matplotlib.png")

def create_graphviz_code():
    """Generate Graphviz DOT code for the pipeline."""
    dot_code = '''
digraph WhisperXPipeline {
    rankdir=TB;
    node [shape=box, style=rounded];
    
    // Title and metrics
    labelloc="t";
    label="WhisperX Enhanced Pipeline\nWER: 3.4% | ASR: 96.6% | DER: 15.6% | Gender: 100% | Confidence: 98.2%";
    
    // Styling
    input [fillcolor="#e1f5fe", style="rounded,filled", label="Input Audio\\nWAV/MP3"];
    vad [fillcolor="#f3e5f5", style="rounded,filled", label="Voice Activity\\nDetection"];
    cutmerge [fillcolor="#f3e5f5", style="rounded,filled", label="Cut & Merge\\nSegments"];
    batch [fillcolor="#f3e5f5", style="rounded,filled", label="Batch\\nProcessing"];
    whisper [fillcolor="#2d2d2d", fontcolor="white", style="rounded,filled", label="Whisper ASR\\nTranscription"];
    phoneme [fillcolor="#2d2d2d", fontcolor="white", style="rounded,filled", label="Phoneme\\nModel"];
    alignment [fillcolor="#2d2d2d", fontcolor="white", style="rounded,filled", label="Forced\\nAlignment"];
    diarization [fillcolor="#e8f5e8", style="rounded,filled", label="Speaker\\nDiarization"];
    ecapa [fillcolor="#e8f5e8", style="rounded,filled", label="ECAPA-TDNN\\nFeatures"];
    zscore [fillcolor="#ffeb3b", style="rounded,filled", label="Z-Score\\nNormalization\\n⚡ Phase 1 Fix"];
    gender [fillcolor="#e8f5e8", style="rounded,filled", label="Gender\\nClassification"];
    output [fillcolor="#fff3e0", style="rounded,filled", label="Enhanced Output\\nMale_SPEAKER_00\\nFemale_SPEAKER_01"];
    
    // Connections
    input -> vad -> cutmerge -> batch -> whisper -> phoneme -> alignment;
    alignment -> diarization -> ecapa -> zscore -> gender -> output;
}
'''
    
    with open('/Users/user/Desktop/WhisperX/figures/pipeline.dot', 'w') as f:
        f.write(dot_code)
    
    print("Graphviz DOT file created: figures/pipeline.dot")
    print("To generate PNG: dot -Tpng figures/pipeline.dot -o figures/pipeline_graphviz.png")

def create_ascii_diagram():
    """Create ASCII art version of the pipeline."""
    ascii_pipeline = '''
WhisperX Enhanced Pipeline with Gender Classification
═══════════════════════════════════════════════════════

┌─────────────┐
│ Input Audio │
│  🎵 WAV/MP3 │
└──────┬──────┘
       │
┌──────▼──────┐
│    Voice    │
│  Activity   │
│ Detection   │
└──────┬──────┘
       │
┌──────▼──────┐
│ Cut & Merge │
│  Segments   │
└──────┬──────┘
       │
┌──────▼──────┐
│    Batch    │
│ Processing  │
└──────┬──────┘
       │
┌──────▼──────┐
│ Whisper ASR │
│Transcription│
└──────┬──────┘
       │
┌──────▼──────┐
│  Phoneme    │
│    Model    │
└──────┬──────┘
       │
┌──────▼──────┐
│   Forced    │
│ Alignment   │
└──────┬──────┘
       │
┌──────▼──────┐
│   Speaker   │
│Diarization  │
└──────┬──────┘
       │
┌──────▼──────┐
│ ECAPA-TDNN  │
│  Features   │
└──────┬──────┘
       │
┌──────▼──────┐
│   Z-Score   │  ⚡ PHASE 1 FIX
│Normalization│
└──────┬──────┘
       │
┌──────▼──────┐
│   Gender    │
│Classification│
└──────┬──────┘
       │
┌──────▼──────┐
│  Enhanced   │
│   Output    │
│Male_SPEAKER │
│Female_SPKR  │
└─────────────┘

Performance: WER 3.4% | ASR 96.6% | DER 15.6% | Gender 100% | Confidence 98.2%
'''
    
    with open('/Users/user/Desktop/WhisperX/figures/pipeline_ascii.txt', 'w') as f:
        f.write(ascii_pipeline)
    
    print("ASCII pipeline saved: figures/pipeline_ascii.txt")

if __name__ == "__main__":
    print("Generating WhisperX pipeline diagrams...")
    
    # Generate different formats
    create_matplotlib_pipeline()
    create_graphviz_code() 
    create_ascii_diagram()
    
    print("\nAll diagrams generated! Choose your preferred format:")
    print("1. Matplotlib PNG: figures/enhanced_pipeline_matplotlib.png")
    print("2. Mermaid Markdown: figures/enhanced_pipeline.md")
    print("3. Graphviz DOT: figures/pipeline.dot")
    print("4. ASCII Text: figures/pipeline_ascii.txt")