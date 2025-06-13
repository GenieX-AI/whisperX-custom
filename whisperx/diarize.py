import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
from typing import Optional, Union
import torch
import warnings

from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.types import TranscriptionResult, AlignedTranscriptionResult

# Import gender classifier with fallback
try:
    from whisperx.gender_classifier import GenderClassifier
    GENDER_CLASSIFICATION_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Gender classification not available: {e}")
    GENDER_CLASSIFICATION_AVAILABLE = False
    GenderClassifier = None

class DiarizationPipeline:
    def __init__(
        self,
        model_name=None,
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
        enable_gender_classification: bool = True,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        model_config = model_name or "pyannote/speaker-diarization-3.1"
        self.model = Pipeline.from_pretrained(model_config, use_auth_token=use_auth_token).to(device)
        # Initialize gender classifier if requested and available
        self.enable_gender = enable_gender_classification and GENDER_CLASSIFICATION_AVAILABLE
        if self.enable_gender:
            try:
                device_str = "cuda" if device.type == "cuda" else "cpu"
                self.gender_classifier = GenderClassifier(device=device_str)
                print("Gender classification enabled")
            except Exception as e:
                warnings.warn(f"Failed to initialize gender classifier: {e}")
                self.gender_classifier = None
                self.enable_gender = False
        else:
            self.gender_classifier = None
            if enable_gender_classification and not GENDER_CLASSIFICATION_AVAILABLE:
                print("Gender classification requested but not available")

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        audio_path = None
        if isinstance(audio, str):
            audio_path = audio  # Store path for gender classification
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        segments = self.model(audio_data, num_speakers = num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        # Apply gender classification if enabled and audio path available
        if self.enable_gender and self.gender_classifier is not None and audio_path is not None:
            try:
                # Convert dataframe to segments format for gender classification
                segments_list = []
                for _, row in diarize_df.iterrows():
                    segments_list.append({
                        'start': row['start'],
                        'end': row['end'],
                        'speaker': row['speaker']
                    })
                # Apply gender classification
                print("Applying gender classification...")
                enhanced_segments = self.gender_classifier.process_segments(audio_path, segments_list)
                # Update dataframe with gender information
                for i, segment in enumerate(enhanced_segments):
                    if i < len(diarize_df):
                        diarize_df.iloc[i, diarize_df.columns.get_loc('speaker')] = segment['speaker']
                        # Add gender columns if they don't exist
                        if 'gender' not in diarize_df.columns:
                            diarize_df['gender'] = None
                        if 'gender_confidence' not in diarize_df.columns:
                            diarize_df['gender_confidence'] = None
                        diarize_df.iloc[i, diarize_df.columns.get_loc('gender')] = segment.get('gender', 'Unknown')
                        diarize_df.iloc[i, diarize_df.columns.get_loc('gender_confidence')] = segment.get('gender_confidence', 0.0)
            except Exception as e:
                warnings.warn(f"Gender classification failed: {e}")
        return diarize_df


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
    fill_nearest=False,
) -> dict:
    transcript_segments = transcript_result["segments"]
    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'], seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])
        # remove no hit, otherwise we look for closest (even negative intersection...)
        if not fill_nearest:
            dia_tmp = diarize_df[diarize_df['intersection'] > 0]
        else:
            dia_tmp = diarize_df
        if len(dia_tmp) > 0:
            # sum over speakers
            speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
            seg["speaker"] = speaker
        
        # assign speaker to words
        if 'words' in seg:
            for word in seg['words']:
                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(diarize_df['start'], word['start'])
                    # remove no hit
                    if not fill_nearest:
                        dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                    else:
                        dia_tmp = diarize_df
                    if len(dia_tmp) > 0:
                        # sum over speakers
                        speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                        word["speaker"] = speaker
        
    return transcript_result            


class Segment:
    def __init__(self, start:int, end:int, speaker:Optional[str]=None):
        self.start = start
        self.end = end
        self.speaker = speaker
