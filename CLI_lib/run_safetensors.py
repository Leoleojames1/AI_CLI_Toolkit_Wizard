import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import gradio as gr

def get_model_and_lora_paths(base_dir):
    model_paths = []
    lora_paths = []
    for root, dirs, files in os.walk(base_dir):
        if 'config.json' in files and 'tokenizer_config.json' in files:
            model_paths.append(root)
        if 'adapter_config.json' in files:
            lora_paths.append(root)
    return model_paths, lora_paths

def setup_model(base_model_path, lora_adapter_path=None):
    print(f"Loading base model from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    if lora_adapter_path:
        print(f"Loading LoRA adapter from {lora_adapter_path}")
        peft_config = PeftConfig.from_pretrained(lora_adapter_path)
        model = PeftModel(base_model, peft_config)
    else:
        model = base_model

    return model, tokenizer

def generate_response(user_input, model, tokenizer):
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def launch_gradio():
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.dirname(script_dir)
    model_paths, lora_paths = get_model_and_lora_paths(base_dir)

    chat_history = []
    model = None
    tokenizer = None

    def load_model(base_model_path, lora_adapter_path):
        nonlocal model, tokenizer
        model, tokenizer = setup_model(base_model_path, lora_adapter_path if lora_adapter_path != "None" else None)
        return "Model loaded successfully!"

    def chat_interface(user_input):
        if not model or not tokenizer:
            return "Please load a model first.", ""

        response = generate_response(user_input, model, tokenizer)
        chat_history.append(("User", user_input))
        chat_history.append(("Assistant", response))
        
        formatted_history = ""
        for speaker, message in chat_history:
            formatted_history += f"{speaker}: {message}\n\n"
        
        return formatted_history, ""

    def clear_history():
        chat_history.clear()
        return "", ""

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# LLaMA 3.1 Chat with Optional LoRA")
        
        with gr.Row():
            with gr.Column(scale=1):
                base_model_dropdown = gr.Dropdown(choices=model_paths, label="Select Base Model")
                lora_adapter_dropdown = gr.Dropdown(choices=["None"] + lora_paths, label="Select LoRA Adapter (Optional)")
                load_button = gr.Button("Load Model")
            
            with gr.Column(scale=2):
                chat_output = gr.Textbox(label="Chat History", interactive=False, lines=20)
                user_input = gr.Textbox(label="Your Message", placeholder="Type your message here...", lines=3)
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")

        load_button.click(load_model, inputs=[base_model_dropdown, lora_adapter_dropdown], outputs=gr.Textbox())
        submit_btn.click(chat_interface, inputs=user_input, outputs=[chat_output, user_input])
        clear_btn.click(clear_history, outputs=[chat_output, user_input])

    demo.launch()

if __name__ == "__main__":
    launch_gradio()
