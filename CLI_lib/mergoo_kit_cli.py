"""
🧠 Advanced Script for Creating and Training Mixture of Experts (MoE) Models with Mergoo

This script provides a comprehensive interface for working with Mixture of Experts (MoE) models using the Mergoo library. It offers both a Command Line Interface (CLI) and a Gradio web interface for easy interaction.

Key Features:

1. MoE Checkpoint Creation:
   - Compose multiple expert models into a single MoE checkpoint
   - Flexible configuration options for expert selection and composition

2. MoE Model Training:
   - Fine-tune MoE models on custom datasets
   - Selectively train router (gating) layers while freezing other parameters

3. Dual Interface:
   - Command Line Interface (CLI) for scripting and automation
   - Gradio web interface for interactive use and experimentation

4. Advanced Model Handling:
   - Support for LLaMA model architecture
   - Integration with Hugging Face's Transformers library

5. Flexible Data Processing:
   - Support for various dataset formats through the Hugging Face datasets library

6. Optimized Training:
   - Utilizes SFTTrainer for efficient fine-tuning
   - Supports gradient accumulation and mixed precision training

Usage Examples:

1. Creating an MoE Checkpoint (CLI):
   python mergoo_kit_cli.py --mode create --config config.json --output moe_checkpoint

2. Training an MoE Model (CLI):
   python mergoo_kit_cli.py --mode train --model_path moe_checkpoint --train_dataset train_data --test_dataset test_data --output trained_model

3. Launch Gradio Interface:
   python mergoo_kit_cli.py

Configuration Options:

- For checkpoint creation:
  - model_type: Base model family (e.g., "llama", "mistral")
  - num_experts_per_tok: Number of active experts per token
  - experts: List of expert models to be composed
  - router_layers: Layers to be replaced with MoE layers

- For model training:
  - Learning rate, batch size, number of epochs, etc., can be customized in the TrainingArguments

Note on MoE Architecture:
Mixture of Experts (MoE) is an architecture that combines multiple "expert" models, each specialized in different aspects of the task. A routing mechanism determines which experts to use for each input, potentially improving model performance and efficiency.

To see a full list of configurable options, use:
    python mergoo_kit_cli.py --help

For more detailed information on Mergoo and MoE models, please refer to the Mergoo documentation.

Happy experimenting with Mixture of Experts! 🧠🚀
"""

import argparse
import gradio as gr
import torch
import json
from mergoo.compose_experts import ComposeExperts
from mergoo.models.modeling_llama import LlamaForCausalLM
from trl import SFTTrainer
from transformers import TrainingArguments
import datasets

# 🏗️ Function to create MOE checkpoint
def create_moe_checkpoint(config, output_path):
    # 🧠 Initialize the expert merger with the given config
    expertmerger = ComposeExperts(config, torch_dtype=torch.float16)
    # 🔨 Compose the experts
    expertmerger.compose()
    # 💾 Save the checkpoint
    expertmerger.save_checkpoint(output_path)
    print(f"MOE checkpoint saved at {output_path}")

# 🚂 Function to train MOE model
def train_moe_model(model_path, train_dataset, test_dataset, output_dir):
    # 📚 Load the pre-trained model
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # 🔒 Freeze all layers except the router (gating) layers
    for name, weight in model.named_parameters():
        if "gate" not in name:
            weight.requires_grad_(False)
    
    # ⚙️ Set up training arguments
    trainer_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-5,
        save_total_limit=1,
        num_train_epochs=1,
        eval_steps=5000,
        logging_strategy="steps",
        logging_steps=25,
        gradient_accumulation_steps=4,
        bf16=True
    )
    
    # 🏋️ Initialize the trainer
    trainer = SFTTrainer(
        model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="prompt",
    )
    
    # 🏃 Start training
    trainer.train()
    print(f"Training completed. Model saved at {output_dir}")

# 🖥️ Function to create Gradio interface
def gradio_interface():
    # 🏗️ Function to create MOE checkpoint (Gradio version)
    def create_moe_checkpoint_gradio(config_json):
        config = json.loads(config_json)
        output_path = "moe_checkpoint"
        create_moe_checkpoint(config, output_path)
        return f"MOE checkpoint created at {output_path}"
    
    # 🚂 Function to train MOE model (Gradio version)
    def train_moe_model_gradio(model_path, train_dataset_path, test_dataset_path):
        train_dataset = datasets.load_dataset(train_dataset_path)['train']
        test_dataset = datasets.load_dataset(test_dataset_path)['train']
        output_dir = "trained_moe_model"
        train_moe_model(model_path, train_dataset, test_dataset, output_dir)
        return f"Model trained and saved at {output_dir}"
    
    # 🎨 Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# Mergoo LoRA MoE Interface")
        
        # 🏗️ Tab for creating MOE checkpoint
        with gr.Tab("Create MOE Checkpoint"):
            config_input = gr.Textbox(label="Config JSON")
            create_button = gr.Button("Create MOE Checkpoint")
            create_output = gr.Textbox(label="Output")
            create_button.click(create_moe_checkpoint_gradio, inputs=[config_input], outputs=[create_output])
        
        # 🚂 Tab for training MOE model
        with gr.Tab("Train MOE Model"):
            model_path_input = gr.Textbox(label="MOE Checkpoint Path")
            train_dataset_input = gr.Textbox(label="Training Dataset Path")
            test_dataset_input = gr.Textbox(label="Test Dataset Path")
            train_button = gr.Button("Train MOE Model")
            train_output = gr.Textbox(label="Output")
            train_button.click(train_moe_model_gradio, inputs=[model_path_input, train_dataset_input, test_dataset_input], outputs=[train_output])
    
    return demo

# 🚀 Main execution block
if __name__ == "__main__":
    import sys
    
    # 🖥️ Check if running in CLI mode
    if len(sys.argv) > 1:
        # 📝 Set up argument parser for CLI
        parser = argparse.ArgumentParser(description="Mergoo LoRA MoE CLI")
        parser.add_argument("--mode", choices=["create", "train"], required=True, help="Mode of operation")
        parser.add_argument("--config", type=str, help="Path to the config file for creating MOE checkpoint")
        parser.add_argument("--output", type=str, required=True, help="Output path for checkpoint or trained model")
        parser.add_argument("--model_path", type=str, help="Path to the MOE checkpoint for training")
        parser.add_argument("--train_dataset", type=str, help="Path to the training dataset")
        parser.add_argument("--test_dataset", type=str, help="Path to the test dataset")
        
        args = parser.parse_args()
        
        # 🏗️ Mode: Create MOE checkpoint
        if args.mode == "create":
            if not args.config:
                parser.error("--config is required for create mode")
            config = json.load(open(args.config))
            create_moe_checkpoint(config, args.output)
        # 🚂 Mode: Train MOE model
        elif args.mode == "train":
            if not all([args.model_path, args.train_dataset, args.test_dataset]):
                parser.error("--model_path, --train_dataset, and --test_dataset are required for train mode")
            train_dataset = datasets.load_dataset(args.train_dataset)['train']
            test_dataset = datasets.load_dataset(args.test_dataset)['train']
            train_moe_model(args.model_path, train_dataset, test_dataset, args.output)
    # 🌐 If no CLI arguments, launch Gradio interface
    else:
        gradio_interface().launch()