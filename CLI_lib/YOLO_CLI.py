import gradio as gr
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
import queue

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class Vision:
    def __init__(self, model):
        self.model = model
        ensure_dir("model_view_output/")
        
    def start_vision(self, queue):
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
            
            queue.put((screen, label_count))
            time.sleep(0.1)

class Api:
    def __init__(self, queue):
        ensure_dir("model_view_output/")
        self.queue = queue

    def get_screen_with_boxes(self):
        try:
            screen, labels = self.queue.get(timeout=1)
            
            for label, count in labels.items():
                positions = self.get_positions_from_label(label)
                for pos in positions:
                    x1, y1, x2, y2 = map(int, pos)
                    cv2.rectangle(screen, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(screen, f"{label}: {count}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            return screen, labels
        except queue.Empty:
            logging.warning("Queue is empty, returning default values")
            return np.zeros((100, 100, 3), dtype=np.uint8), {}
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

def start_vision_process(model, queue):
    vision_instance = Vision(model)
    vision_instance.start_vision(queue)

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
    def update_output():
        screen, label_text = update_screen(api_instance)
        return screen, label_text

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

        demo.load(update_output, outputs=[image_output, label_output], every=1)

    return demo

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Add this line for Windows support
    vision_model = "Computer_Vision_1.3.0.onnx"  # Make sure this file exists
    
    data_queue = multiprocessing.Queue()
    
    vision_process = multiprocessing.Process(target=start_vision_process, args=(vision_model, data_queue))
    vision_process.start()
    
    api_instance = Api(data_queue)
    
    demo = create_gradio_interface(api_instance)
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=5000)