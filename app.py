import gradio as gr
from transcriber import transcribe_with_diarization

def process_audio(file):
    transcript, output_path = transcribe_with_diarization(file)
    return transcript, output_path  # transcript = preview, path = download link

iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Textbox(label="Transcript", lines=10, max_lines=20),
        gr.File(label="Download Transcript File")
    ],
    title="Voice-to-Text with Speaker Diarization"
)

iface.launch()
