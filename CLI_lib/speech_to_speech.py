import gradio as gr
import speech_recognition as sr
import threading
import queue
import time

def transcribe_speech(audio, state):
    if state is None:
        state = {"recognizer": sr.Recognizer(), "queue": queue.Queue()}
    
    recognizer = state["recognizer"]
    text_queue = state["queue"]

    if audio is not None:
        try:
            text = recognizer.recognize_google(audio)
            text_queue.put(text)
        except sr.UnknownValueError:
            text_queue.put("")
        except sr.RequestError:
            text_queue.put("API unavailable")

    all_text = []
    while not text_queue.empty():
        all_text.append(text_queue.get())

    return " ".join(all_text), state

def clear_output():
    return "", None

with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Live Speech Recognition")
    audio_input = gr.Audio(source="microphone", type="numpy", streaming=True)
    text_output = gr.Markdown("", elem_id="large-text-output")
    clear_button = gr.Button("Clear Output")

    state = gr.State()

    audio_input.stream(
        fn=transcribe_speech,
        inputs=[audio_input, state],
        outputs=[text_output, state],
    )

    clear_button.click(
        fn=clear_output,
        outputs=[text_output, state],
    )

    gr.Markdown("""
    ## Instructions:
    1. Click the microphone icon to start recording.
    2. Speak clearly into your microphone.
    3. Watch as your speech is transcribed in real-time!
    4. Click the "Clear Output" button to reset the transcription.
    """)

if __name__ == "__main__":
    demo.launch()