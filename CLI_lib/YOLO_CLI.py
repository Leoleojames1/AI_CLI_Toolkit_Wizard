import gradio as gr
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
import json
import numpy as np
from PIL import ImageGrab
import cv2
import ollama
from ultralytics import YOLO
import re
import multiprocessing
import logging
import os
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
class Vision:
    def __init__(self, model):
        self.model = model
        ensure_dir("model_view_output/")
        
    def start_vision(self):
        logging.getLogger('ultralytics').setLevel(logging.WARNING)
        model = YOLO(self.model, task='detect')
        labels = list(model.names.values())
    
        while True:
            label_count = {label: 0 for label in labels}
            label_dict = {label: [] for label in labels}
            screen = np.array(ImageGrab.grab())
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
            results = model(screen)

            for result in results:
                for box in result.boxes:
                    label = model.names[int(box.cls)]  
                    coords = box.xyxy[0].tolist()
                    label_count[label] += 1
                    label_dict[label].append(coords)

                for key in label_dict:
                    file_path = f"model_view_output/{key}.json"
                    ensure_dir(file_path)
                    with open(file_path, "w") as f:
                        json.dump(label_dict[key], f, indent=4)

                file_path = "model_view_output/labels.json"
                ensure_dir(file_path)
                with open(file_path, "w") as f:
                    json.dump(label_count, f, indent=2)
            
                time.sleep(0.1)

class Api:
    def __init__(self):
        ensure_dir("model_view_output/")
        
    def get_screen_labels(self):
        try:
            with open("model_view_output/labels.json", "r") as f:
                labels = json.load(f)
            return labels
        except FileNotFoundError:
            return {}

    def get_positions_from_label(self, label):
        try: 
            with open(f"model_view_output/{key}.json", "r") as f:
                coords_list = json.load(f)
            return coords_list
        except FileNotFoundError:
            return []

    def get_screen_with_boxes(self):
        screen = np.array(ImageGrab.grab())
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        
        labels = self.get_screen_labels()
        for label, count in labels.items():
            positions = self.get_positions_from_label(label)
            for pos in positions:
                x1, y1, x2, y2 = map(int, pos)
                cv2.rectangle(screen, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(screen, f"{label}: {count}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return screen

    def remove_non_ascii(self, text):
        return re.sub(r'[^\x00-\x7F]+', '', text)

    def updating_text(self, model):
        model = YOLO(model, task='detect')
        labels = list(model.names.values())

        while True:
            file_path = "model_view_output/text.json"
            ensure_dir(file_path)
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                with open(file_path, "w") as f:
                    json.dump([], f)
            
            with open(file_path, "r") as f:
                try:
                    coords_list = json.load(f)
                except json.JSONDecodeError:
                    coords_list = []
        
            names = []
            for coords in coords_list:
                names.append(f"Object at {coords}")  # Placeholder for OCR

            names = [name.replace('\n', '') for name in names]
            names = [self.remove_non_ascii(name) for name in names]

            position_names = {text: [] for text in names}

            for i, key in enumerate(names):
                position_names[key] = coords_list[i]

            with open("model_view_output/sumirize.json", "w") as f:
                json.dump(position_names, f, indent=3)
            time.sleep(1)

api = Api()

def get_labels():
    return api.get_screen_labels()

def get_screen_with_boxes():
    return api.get_screen_with_boxes()

def chat(message):
    response = ollama.chat(model='llama3.1:8b', messages=[
        {
            'role': 'user',
            'content': message,
        },
    ])
    return response['message']['content']

def start_vision_process(model):
    vision_instance = Vision(model)
    vision_instance.start_vision()

def start_updating(model):
    api = Api()
    api.updating_text(model=model)

def update_screen():
    screen = get_screen_with_boxes()
    labels = get_labels()
    label_text = "\n".join([f"{label}: {count}" for label, count in labels.items()])
    return screen, label_text

def gradio_chat(message, history):
    response = chat(message)
    history.append((message, response))
    return "", history

# Gradio Interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            image_output = gr.Image(label="Screen with Bounding Boxes")
        with gr.Column(scale=1):
            label_output = gr.Textbox(label="Detected Objects", lines=10)
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(gradio_chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

    demo.load(update_screen, outputs=[image_output, label_output])

if __name__ == '__main__':
    vision_model = "Computer_Vision_1.3.0.onnx"  # Make sure this file exists
    vision_process = multiprocessing.Process(target=start_vision_process, args=(vision_model,))
    updating_process = multiprocessing.Process(target=start_updating, args=(vision_model,))
    vision_process.start()
    updating_process.start()
    
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=5000)