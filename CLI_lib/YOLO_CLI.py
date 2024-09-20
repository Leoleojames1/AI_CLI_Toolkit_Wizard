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
import threading

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
        self.screen = None
        self.labels = {}
        self.lock = threading.Lock()
        
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
            
            with self.lock:
                self.screen = screen
                self.labels = label_count

            time.sleep(0.1)

class Api:
    def __init__(self, vision_instance):
        ensure_dir("model_view_output/")
        self.vision = vision_instance

    def get_screen_with_boxes(self):
        try:
            with self.vision.lock:
                screen = self.vision.screen.copy()
                labels = self.vision.labels.copy()
            
            for label, count in labels.items():
                positions = self.get_positions_from_label(label)
                for pos in positions:
                    x1, y1, x2, y2 = map(int, pos)
                    cv2.rectangle(screen, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(screen, f"{label}: {count}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            return screen, labels
        except Exception as e:
            logging.error(f"Error in get_screen_with_boxes: {str(e)}")
            return np.zeros((100, 100, 3), dtype=np.uint8), {}

    def get_positions_from_label(self, label):
        try: 
            with open(f"model_view_output/{label}.json", "r") as f:
                coords_list = json.load(f)
            return coords_list
        except FileNotFoundError:
            return []

def chat(message):
    try:
        response = ollama.chat(model='llama3.1:8b', messages=[
            {
                'role': 'user',
                'content': message,
            },
        ])
        return response['message']['content']
    except Exception as e:
        logging.error(f"Error in chat function: {str(e)}")
        return "Sorry, I encountered an error while processing your request."

def start_vision_process(vision_instance):
    vision_instance.start_vision()

def update_screen(api_instance):
    try:
        screen, labels = api_instance.get_screen_with_boxes()
        label_text = "\n".join([f"{label}: {count}" for label, count in labels.items()])
        return screen, label_text
    except Exception as e:
        logging.error(f"Error in update_screen: {str(e)}")
        return np.zeros((100, 100, 3), dtype=np.uint8), "Error occurred while updating screen"

def gradio_chat(message, history):
    response = chat(message)
    history.append((message, response))
    return "", history

# Gradio Interface
def create_gradio_interface(api_instance):
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

        def update_periodically():
            screen, label_text = update_screen(api_instance)
            return screen, label_text

        demo.load(update_periodically, outputs=[image_output, label_output])
        demo.add_periodic_callback(update_periodically, 1, outputs=[image_output, label_output])

    return demo

if __name__ == '__main__':
    vision_model = "Computer_Vision_1.3.0.onnx"  # Make sure this file exists
    vision_instance = Vision(vision_model)
    api_instance = Api(vision_instance)
    
    vision_process = multiprocessing.Process(target=start_vision_process, args=(vision_instance,))
    vision_process.start()
    
    demo = create_gradio_interface(api_instance)
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=5000)