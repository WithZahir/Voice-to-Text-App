# transcriber.py

import whisper
import os
import subprocess
from diarizer import diarize_audio
import tempfile

# Inject ffmpeg into PATH

os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

model = whisper.load_model("base")

def transcribe_with_diarization(audio_path):
    # Get speaker segments
    segments = diarize_audio(audio_path)

    # Transcribe entire audio
    result = model.transcribe(audio_path, word_timestamps=True)
    words = result.get("segments", [])

    # Build speaker-wise transcript
    transcript = ""
    for segment in segments:
        speaker = segment["speaker"]
        start = segment["start"]
        end = segment["end"]

        # Get words in this time range
        speaker_text = ""
        for seg in words:
            if seg["start"] >= start and seg["end"] <= end:
                speaker_text += seg["text"].strip() + " "

        if speaker_text.strip():
            transcript += f"{speaker}: {speaker_text.strip()}\n\n"

    # Save to file
    output_path = "speaker_transcript.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    return transcript, output_path
