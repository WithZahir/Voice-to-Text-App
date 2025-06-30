
# diarizer.py

from pyannote.audio import Pipeline
import torch
import os

# Set your Hugging Face token here
HF_TOKEN = os.getenv("HF_TOKEN")


# Load pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)

def diarize_audio(audio_path):
    diarization = pipeline(audio_path)

    # Parse speaker segments into list of tuples
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        })
    return speaker_segments
