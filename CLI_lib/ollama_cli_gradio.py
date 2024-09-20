#!/usr/bin/env python3

"""
🤖 Ollama CLI-Gradio Chatbot

This script provides a command-line interface (CLI) and Gradio web interface for interacting with Ollama models.
It allows users to select a model, chat with it, upload files, and maintain conversation history.

Key Features:
1. Select and chat with Ollama models
2. Upload and process files during conversations
3. Maintain conversation history using JSON
4. CLI and Gradio web interface options

Usage examples:
    python ollama_cli_gradio.py chat --model llama2
    python ollama_cli_gradio.py list-models
    python ollama_cli_gradio.py gradio

Customize this script by:
1. Adding more Ollama-related commands
2. Extending file processing capabilities
3. Implementing additional features in the Gradio interface
"""

import argparse
import json
import logging
import os
from typing import List, Dict

import gradio as gr
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HISTORY_FILE = "chat_history.json"

def load_chat_history() -> List[Dict[str, str]]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_chat_history(history: List[Dict[str, str]]):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

def get_ollama_models() -> List[str]:
    try:
        models = ollama.list()
        return [model['name'] for model in models['models']]
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {e}")
        return []

def chat_with_ollama(model: str, message: str, history: List[Dict[str, str]]) -> str:
    try:
        response = ollama.chat(model=model, messages=history + [{"role": "user", "content": message}])
        return response['message']['content']
    except Exception as e:
        logger.error(f"Error chatting with Ollama: {e}")
        return f"Error: {e}"

def process_file(file_path: str) -> str:
    try:
        with open(file_path, "r") as f:
            content = f.read()
        return f"File contents:\n\n{content}"
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return f"Error processing file: {e}"

def run_chat(model: str):
    history = load_chat_history()
    print(f"Chatting with {model}. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = chat_with_ollama(model, user_input, history)
        print(f"{model}: {response}")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
    save_chat_history(history)

def run_list_models():
    models = get_ollama_models()
    print("Available Ollama models:")
    for model in models:
        print(f"- {model}")

def create_gradio_interface():
    models = get_ollama_models()
    history = load_chat_history()

    with gr.Blocks() as interface:
        gr.Markdown("# Ollama Chatbot")
        
        with gr.Row():
            model_dropdown = gr.Dropdown(choices=models, label="Select Ollama Model", value=models[0] if models else None)
        
        chatbot = gr.Chatbot(value=[(msg["content"], None) if msg["role"] == "user" else (None, msg["content"]) for msg in history], height=400)
        msg = gr.Textbox(label="Message")
        clear = gr.Button("Clear")
        
        file_upload = gr.File(label="Upload File")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history, model):
            user_message = history[-1][0]
            bot_message = chat_with_ollama(model, user_message, [{"role": "user" if h[0] else "assistant", "content": h[0] or h[1]} for h in history[:-1]])
            history[-1][1] = bot_message
            save_chat_history([{"role": "user" if h[0] else "assistant", "content": h[0] or h[1]} for h in history])
            return history

        def process_file_gradio(file):
            if file:
                return process_file(file.name)
            return "No file uploaded."

        msg.submit(user, [msg, chatbot], [msg, chatbot]).then(
            bot, [chatbot, model_dropdown], chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)
        file_upload.upload(process_file_gradio, file_upload, chatbot)

    return interface

def create_cli_parser():
    parser = argparse.ArgumentParser(description="🤖 Ollama CLI-Gradio Chatbot")
    parser.add_argument("gradio", action="store_true", help="Run the Gradio interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    chat_parser = subparsers.add_parser("chat", help="Chat with an Ollama model")
    chat_parser.add_argument("--model", type=str, required=True, help="Name of the Ollama model to use")

    subparsers.add_parser("list-models", help="List available Ollama models")

    return parser

if __name__ == "__main__":
    parser = create_cli_parser()
    args = parser.parse_args()

    if args.gradio:
        interface = create_gradio_interface()
        interface.launch()
    elif args.command == "chat":
        run_chat(args.model)
    elif args.command == "list-models":
        run_list_models()
    else:
        parser.print_help()
        exit(1)