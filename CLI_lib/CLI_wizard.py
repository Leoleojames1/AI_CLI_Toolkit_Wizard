#!/usr/bin/env python3

"""
🧙‍♂️ Enhanced CLI Wizard for Model Management 🪄

This script provides a visually appealing and interactive command-line interface
for managing and interfacing with other CLIs related to model conversion and processing.

Key Features:
1. Colorful and emoji-rich interface
2. ASCII art for enhanced visual appeal
3. Interactive mode with user-friendly prompts
4. Loading spinners for long-running operations
5. Interfaces with Unsloth and LlamaCpp CLIs
6. Model listing and organization features

Usage examples:
    python cli_wizard.py unsloth --model_name mymodel --train_dataset train.parquet
    python cli_wizard.py llamacpp --input model.safetensors --output model.gguf
    python cli_wizard.py list_models
    python cli_wizard.py organize --model mymodel
    python cli_wizard.py interactive

Customize this CLI by adding new features and enhancing the visual elements!
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
import gradio as gr
from colorama import init, Fore, Style
from halo import Halo
from huggingface_hub import login

# Initialize colorama for cross-platform color support
init()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Define paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_GIT_DIR = PROJECT_ROOT.parent
MODEL_FORGE_DIR = PROJECT_ROOT / "model_forge"
UNSLOTH_OUTPUT_DIR = MODEL_FORGE_DIR / "unsloth"
LLAMACPP_OUTPUT_DIR = MODEL_FORGE_DIR / "llamacpp"

print(Fore.YELLOW + "=" * 80 + Style.RESET_ALL)
# ASCII Art
CLI_WIZARD_LOGO = """
   ▄████████  ▄█        ▄████████  ▄█        ▄█           ███      ▄██████▄   ▄██████▄   ▄█          ▄█   ▄█▄  ▄█      ███     
  ███    ███ ███       ███    ███ ███       ███       ▀█████████▄ ███    ███ ███    ███ ███         ███ ▄███▀ ███  ▀█████████▄ 
  ███    ███ ███▌      ███    █▀  ███       ███▌         ▀███▀▀██ ███    ███ ███    ███ ███         ███▐██▀   ███▌    ▀███▀▀██ 
  ███    ███ ███▌      ███        ███       ███▌          ███   ▀ ███    ███ ███    ███ ███        ▄█████▀    ███▌     ███   ▀ 
▀███████████ ███▌      ███        ███       ███▌          ███     ███    ███ ███    ███ ███       ▀▀█████▄    ███▌     ███     
  ███    ███ ███       ███    █▄  ███       ███           ███     ███    ███ ███    ███ ███         ███▐██▄   ███      ███     
  ███    ███ ███       ███    ███ ███▌    ▄ ███           ███     ███    ███ ███    ███ ███▌    ▄   ███ ▀███▄ ███      ███     
  ███    █▀  █▀        ████████▀  █████▄▄██ █▀           ▄████▀    ▀██████▀   ▀██████▀  █████▄▄██   ███   ▀█▀ █▀      ▄████▀                                                                                               
"""
print(Fore.YELLOW + "=" * 80 + Style.RESET_ALL)

WIZARD_ASCII = """
              _,-'|
           ,-'._  |
 .||,      |####\ |
\.`',/     \####| |
= ,. =      |###| |
/ || \    ,-'\#/,'`.
  ||     ,'   `,,. `.
  ,|____,' , ,;' \| |
 (3|\    _/|/'   _| |
  ||/,-''  | >-'' _,\\
  ||'      ==\ ,-'  ,'
  ||       |  V \ ,|
  ||       |    |` |
  ||       |    |   \\
  ||       |    \    \\
  ||       |     |    \\
  ||       |      \_,-'
  ||       |___,,--")_\\
  ||         |_|   ccc/
  ||        ccc/
  ||                hjm
"""

def print_fancy_header():
    print(Fore.CYAN + CLI_WIZARD_LOGO + Style.RESET_ALL)
    print(Fore.MAGENTA + WIZARD_ASCII + Style.RESET_ALL)
    print(Fore.YELLOW + "=" * 80 + Style.RESET_ALL)
    print(Fore.GREEN + "Welcome to the Enhanced CLI Wizard for Model Management! 🧙‍♂️✨" + Style.RESET_ALL)
    print(Fore.YELLOW + "=" * 80 + Style.RESET_ALL)

def run_unsloth(args):
    """
    Interface with the Unsloth CLI for model processing.
    
    Args:
        args: Parsed command-line arguments
    """
    logger.info(Fore.CYAN + f"🦥 Running Unsloth CLI for model: {args.model_name}" + Style.RESET_ALL)
    
    unsloth_script = SCRIPT_DIR / "unsloth-cli-2.py"
    unsloth_command = [
        sys.executable, str(unsloth_script),
        "train",
        "--model_name", str(MODEL_GIT_DIR / args.model_name),
        "--train_dataset", args.train_dataset,
        "--output_dir", str(UNSLOTH_OUTPUT_DIR / args.model_name)
    ]
    
    if args.validation_dataset:
        unsloth_command.extend(["--validation_dataset", args.validation_dataset])
    if args.test_dataset:
        unsloth_command.extend(["--test_dataset", args.test_dataset])
    
    with Halo(text='Processing with Unsloth', spinner='dots') as spinner:
        try:
            subprocess.run(unsloth_command, check=True)
            spinner.succeed(Fore.GREEN + f"✅ Unsloth processing completed for {args.model_name}" + Style.RESET_ALL)
        except subprocess.CalledProcessError as e:
            spinner.fail(Fore.RED + f"❌ Unsloth processing failed for {args.model_name}: {e}" + Style.RESET_ALL)

def run_llamacpp(args):
    """
    Interface with the LlamaCpp CLI for GGUF conversion.
    
    Args:
        args: Parsed command-line arguments
    """
    if not args.model_name:
        raise ValueError("Model name is required")
    
    logger.info(Fore.CYAN + f"🦙 Running LlamaCpp CLI for conversion: {args.model_name}" + Style.RESET_ALL)
    
    llamacpp_script = SCRIPT_DIR / "llamacpp_tools_CLI.py"
    input_dir = MODEL_GIT_DIR
    output_dir = LLAMACPP_OUTPUT_DIR
    
    llamacpp_command = [
        sys.executable,
        str(llamacpp_script),
        "convert",
        "--input_dir", str(input_dir),
        "--model_name", args.model_name,
        "--output_dir", str(output_dir),
        "--quant_type", args.quant_type
    ]
    
    # Debug: Print the command being executed
    print("Debug: Executing command:", " ".join(llamacpp_command))
    
    with Halo(text='Converting to GGUF', spinner='dots') as spinner:
        try:
            result = subprocess.run(llamacpp_command, check=True, capture_output=True, text=True)
            print("Debug: Subprocess output:", result.stdout)
            print("Debug: Subprocess error:", result.stderr)
            spinner.succeed(Fore.GREEN + f"✅ LlamaCpp conversion completed for {args.model_name}" + Style.RESET_ALL)
        except subprocess.CalledProcessError as e:
            error_msg = Fore.RED + f"❌ LlamaCpp conversion failed for {args.model_name}: {e}" + Style.RESET_ALL
            print("Debug: Subprocess error output:", e.output)
            print("Debug: Subprocess error:", e.stderr)
            spinner.fail(error_msg)
            raise RuntimeError(error_msg)
        
def list_models(args=None):
    """
    List all models in the model_git folder.
    
    Args:
        args: Parsed command-line arguments (optional)
    """
    logger.info(Fore.YELLOW + "📋 Listing models in model_git folder:" + Style.RESET_ALL)
    models = []
    for model in MODEL_GIT_DIR.iterdir():
        if model.is_dir():
            logger.info(Fore.GREEN + f"  - {model.name}" + Style.RESET_ALL)
            models.append(model.name)
    return models

def organize_model(args):
    """
    Organize a model's files in the model_forge folder.
    
    Args:
        args: Parsed command-line arguments
    """
    logger.info(Fore.CYAN + f"🗂️  Organizing files for model: {args.model}" + Style.RESET_ALL)
    
    unsloth_model_dir = UNSLOTH_OUTPUT_DIR / args.model
    llamacpp_model_dir = LLAMACPP_OUTPUT_DIR / args.model
    
    unsloth_model_dir.mkdir(parents=True, exist_ok=True)
    llamacpp_model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(Fore.GREEN + f"✅ Created directories for {args.model} in model_forge" + Style.RESET_ALL)

def run_hf_command(args):
    """Run HuggingFace CLI command"""
    hf_cli_path = SCRIPT_DIR / "HF_CLI.py"
    command = [sys.executable, str(hf_cli_path), args.hf_command] + sys.argv[3:]
    subprocess.run(command)

def run_llamacpp(args):
    """
    Interface with the LlamaCpp CLI for GGUF conversion.
    
    Args:
        args: Parsed command-line arguments
    """
    if not args.model_name:
        raise ValueError("Model name is required")
    
    logger.info(Fore.CYAN + f"🦙 Running LlamaCpp CLI for conversion: {args.model_name}" + Style.RESET_ALL)
    
    llamacpp_script = SCRIPT_DIR / "llamacpp_tools_CLI.py"
    
    llamacpp_command = [
        sys.executable,
        str(llamacpp_script),
        "convert",
        "--input_dir", str(args.input_dir),
        "--model_name", args.model_name,
        "--output_dir", str(args.output_dir),
        "--quant_type", args.quant_type
    ]
    
    # Debug: Print the command being executed
    print("Debug: Executing command:", " ".join(llamacpp_command))
    
    with Halo(text='Converting to GGUF', spinner='dots') as spinner:
        try:
            result = subprocess.run(llamacpp_command, check=True, capture_output=True, text=True)
            print("Debug: Subprocess output:", result.stdout)
            print("Debug: Subprocess error:", result.stderr)
            spinner.succeed(Fore.GREEN + f"✅ LlamaCpp conversion completed for {args.model_name}" + Style.RESET_ALL)
        except subprocess.CalledProcessError as e:
            error_msg = Fore.RED + f"❌ LlamaCpp conversion failed for {args.model_name}: {e}" + Style.RESET_ALL
            print("Debug: Subprocess error output:", e.output)
            print("Debug: Subprocess error:", e.stderr)
            spinner.fail(error_msg)
            raise RuntimeError(error_msg)

def run_unsloth_train(args):
    """Train a model using Unsloth"""
    logger.info(f"Training model {args.model_name} with Unsloth")
    # Implement your Unsloth training logic here
    # You can use subprocess to call your existing Unsloth CLI script
    pass

def interactive_mode():
    """
    Run the CLI wizard in interactive mode.
    """
    print_fancy_header()
    
    while True:
        print(Fore.YELLOW + "\n" + "#" * 40 + Style.RESET_ALL)
        print(Fore.CYAN + "Available operations:" + Style.RESET_ALL)
        print(Fore.GREEN + "1. 🔄 Convert SafeTensors model to GGUF")
        print("2. 📋 List available models")
        print("3. 🚪 Exit" + Style.RESET_ALL)
        print(Fore.YELLOW + "#" * 40 + Style.RESET_ALL)
        
        choice = input(Fore.MAGENTA + "Enter your choice (1-3): " + Style.RESET_ALL).strip()
        
        if choice == '1':
            models = list_models()
            if not models:
                print(Fore.RED + "❌ No models found in the model_git directory." + Style.RESET_ALL)
                continue
            
            print(Fore.CYAN + "\nAvailable models:" + Style.RESET_ALL)
            for i, model in enumerate(models, 1):
                print(Fore.GREEN + f"{i}. {model}" + Style.RESET_ALL)
            
            model_choice = input(Fore.MAGENTA + "Enter the number of the model you want to convert: " + Style.RESET_ALL).strip()
            try:
                model_index = int(model_choice) - 1
                if 0 <= model_index < len(models):
                    selected_model = models[model_index]
                    quant_type = input(Fore.MAGENTA + "Enter quantization type (q8_0, f16, f32): " + Style.RESET_ALL).strip() or "q8_0"
                    
                    run_llamacpp(argparse.Namespace(model_name=selected_model, quant_type=quant_type))
                else:
                    print(Fore.RED + "❌ Invalid model number." + Style.RESET_ALL)
            except ValueError:
                print(Fore.RED + "❌ Please enter a valid number." + Style.RESET_ALL)
        
        elif choice == '2':
            list_models()
        
        elif choice == '3':
            print(Fore.YELLOW + "🧙‍♂️ Exiting the CLI Wizard. Farewell, brave adventurer! ✨" + Style.RESET_ALL)
            break
        
        else:
            print(Fore.RED + "❌ Invalid choice. Please enter a number between 1 and 3." + Style.RESET_ALL)

def run_llamacpp_gradio(model_name, quant_type):
    """
    Run LlamaCpp conversion for Gradio interface.
    """
    if not model_name:
        return "Error: No model selected. Please choose a model from the dropdown."
    
    args = argparse.Namespace(model_name=model_name, quant_type=quant_type)
    
    try:
        run_llamacpp(args)
        return f"Conversion completed for {model_name} with {quant_type} quantization."
    except Exception as e:
        return f"Error during conversion: {str(e)}"

def gradio_unsloth_interface():
    with gr.Blocks() as unsloth_interface:
        gr.Markdown("# Unsloth Training")
        model_name_input = gr.Textbox(label="Model Name")
        train_dataset_input = gr.File(label="Training Dataset")
        train_button = gr.Button("Train Model")
        train_output = gr.Textbox(label="Output")

        def train_model(model_name, train_dataset):
            args = argparse.Namespace(model_name=model_name, train_dataset=train_dataset.name)
            run_unsloth_train(args)
            return f"Trained model: {model_name} with dataset {train_dataset.name}"

        train_button.click(train_model, inputs=[model_name_input, train_dataset_input], outputs=train_output)

    return unsloth_interface

def gradio_llamacpp_interface():
    with gr.Blocks() as llamacpp_interface:
        gr.Markdown("# 🦙 LlamaCpp Operations")
        
        def get_available_models():
            return [model.name for model in MODEL_GIT_DIR.iterdir() if model.is_dir()]

        model_dropdown = gr.Dropdown(
            label="Select Model",
            choices=get_available_models(),
            type="value",
            interactive=True
        )
        
        refresh_button = gr.Button("🔄 Refresh Model List")
        
        quant_type_input = gr.Dropdown(
            ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
            label="Quantization Type",
            value="q8_0"
        )
        convert_button = gr.Button("🔄 Convert Model")
        convert_output = gr.Textbox(label="Conversion Output")

        def refresh_models():
            return gr.Dropdown.update(choices=get_available_models())

        def convert_model(model_name, quant_type):
            if not model_name:
                return "Error: No model selected. Please choose a model from the dropdown."
            
            args = argparse.Namespace(
                model_name=model_name,
                quant_type=quant_type,
                input_dir=str(MODEL_GIT_DIR),
                output_dir=str(LLAMACPP_OUTPUT_DIR)
            )
            
            try:
                run_llamacpp(args)
                return f"Conversion completed for {model_name} with {quant_type} quantization."
            except Exception as e:
                return f"Error during conversion: {str(e)}"

        refresh_button.click(refresh_models, outputs=[model_dropdown])
        convert_button.click(convert_model, inputs=[model_dropdown, quant_type_input], outputs=[convert_output])

    return llamacpp_interface

def gradio_hf_interface():
    with gr.Blocks() as hf_interface:
        gr.Markdown("# 🤗 HuggingFace Operations")
        
        with gr.Tab("🔑 Login"):
            token_input = gr.Textbox(label="HuggingFace Token", type="password")
            login_button = gr.Button("Login")
            login_output = gr.Textbox(label="Login Status")

            def hf_login(token):
                try:
                    login(token=token)
                    return "Successfully logged in to Hugging Face Hub"
                except Exception as e:
                    return f"Error logging in: {str(e)}"

            login_button.click(hf_login, inputs=token_input, outputs=login_output)

        with gr.Tab("⬇️ Pull Model"):
            model_name_input = gr.Textbox(label="Model Name")
            pull_button = gr.Button("Pull Model")
            pull_output = gr.Textbox(label="Output")

            def pull_model(model_name):
                result = subprocess.run([sys.executable, str(SCRIPT_DIR / "HF_CLI.py"), "pull", "--model_name", model_name], capture_output=True, text=True)
                return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"

            pull_button.click(pull_model, inputs=model_name_input, outputs=pull_output)

        with gr.Tab("⬆️ Push Model"):
            local_path_input = gr.Textbox(label="Local Model Path")
            repo_name_input = gr.Textbox(label="Repository Name")
            push_button = gr.Button("Push Model")
            push_output = gr.Textbox(label="Output")

            def push_model(local_path, repo_name):
                result = subprocess.run([sys.executable, str(SCRIPT_DIR / "HF_CLI.py"), "push", "--local_path", local_path, "--repo_name", repo_name], capture_output=True, text=True)
                return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"

            push_button.click(push_model, inputs=[local_path_input, repo_name_input], outputs=push_output)

    return hf_interface

def list_models_gradio():
    """
    List models for Gradio interface.
    """
    models = list_models()
    return "\n".join(models)

def launch_gradio():
    with gr.Blocks(title="🧙‍♂️ Model Management Wizard ✨", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🧙‍♂️ Enhanced CLI Wizard for Model Management ✨
            Welcome to the magical world of model management! Choose your adventure below.
            """
        )
        with gr.Tabs():
            with gr.TabItem("🤗 HuggingFace"):
                gradio_hf_interface()
            with gr.TabItem("🦙 LlamaCpp"):
                gradio_llamacpp_interface()
            with gr.TabItem("🦥 Unsloth"):
                gradio_unsloth_interface()

    demo.launch()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🧙‍♂️ Enhanced CLI Wizard for Model Management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    unsloth_parser = subparsers.add_parser("unsloth", help="Interface with Unsloth CLI")
    unsloth_parser.add_argument('--model_name', type=str, required=True, help="Name of the model to process")
    unsloth_parser.add_argument('--train_dataset', type=str, required=True, help="Path to the training dataset")
    unsloth_parser.add_argument('--validation_dataset', type=str, help="Path to the validation dataset")
    unsloth_parser.add_argument('--test_dataset', type=str, help="Path to the test dataset")

    llamacpp_parser = subparsers.add_parser("llamacpp", help="Interface with LlamaCpp CLI")
    llamacpp_parser.add_argument('--model_name', type=str, required=True, help="Name of the model to convert")
    llamacpp_parser.add_argument('--quant_type', type=str, default='q8_0', choices=['q8_0', 'f16', 'f32'], help="Quantization type")

    list_parser = subparsers.add_parser("list_models", help="List all models in model_git folder")

    organize_parser = subparsers.add_parser("organize", help="Organize a model's files in model_forge folder")
    organize_parser.add_argument('--model', type=str, required=True, help="Name of the model to organize")

    interactive_parser = subparsers.add_parser("interactive", help="Run in interactive mode")

    gradio_parser = subparsers.add_parser("gradio", help="Launch Gradio UI")

    args = parser.parse_args()

    if args.command == "unsloth":
        run_unsloth(args)
    elif args.command == "llamacpp":
        run_llamacpp(args)
    elif args.command == "list_models":
        list_models(args)
    elif args.command == "organize":
        organize_model(args)
    elif args.command == "interactive":
        interactive_mode()
    elif args.command == "gradio":
        launch_gradio()
    else:
        print_fancy_header()
        parser.print_help()