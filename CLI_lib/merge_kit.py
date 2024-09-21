#!/usr/bin/env python3

"""
🔧 MergeKit CLI and Gradio Interface

This script provides a command-line interface (CLI) and a Gradio web interface for merging language models using the mergekit library.

Key Features:
1. Multiple merge methods: Linear, SLERP, Task Arithmetic, TIES, DARE, Passthrough, Model Breadcrumbs, Model Stock, DELLA
2. CLI with subcommands for merging and LoRA extraction
3. Gradio interface for web-based interaction
4. Flexible configuration options for each merge method

Usage examples:
    python mergekit_cli_gradio.py merge config.yml ./output-model-directory
    python mergekit_cli_gradio.py extract-lora finetuned_model base_model output_path --rank 8
    python mergekit_cli_gradio.py gradio

Customize this script by:
1. Adding or modifying merge methods
2. Updating the argument parsers for each method
3. Extending the Gradio interface to include new features
4. Implementing additional mergekit functionality
"""

import argparse
import logging
import yaml
import gradio as gr
from typing import List, Dict, Any
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_merge(config_path: str, output_path: str, cuda: bool, lazy_unpickle: bool, allow_crimes: bool) -> str:
    """Execute the merge operation using mergekit-yaml."""
    cmd = ["mergekit-yaml", config_path, output_path]
    if cuda:
        cmd.append("--cuda")
    if lazy_unpickle:
        cmd.append("--lazy-unpickle")
    if allow_crimes:
        cmd.append("--allow-crimes")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"

def run_extract_lora(finetuned_model: str, base_model: str, output_path: str, rank: int, no_lazy_unpickle: bool) -> str:
    """Execute the LoRA extraction operation using mergekit-extract-lora."""
    cmd = ["mergekit-extract-lora", finetuned_model, base_model, output_path, f"--rank={rank}"]
    if no_lazy_unpickle:
        cmd.append("--no-lazy-unpickle")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"

def create_cli_parser():
    parser = argparse.ArgumentParser(description="🔧 MergeKit CLI and Gradio Interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Merge parser
    merge_parser = subparsers.add_parser("merge", help="Execute model merge")
    merge_parser.add_argument('config', type=str, help="Path to YAML configuration file")
    merge_parser.add_argument('output', type=str, help="Output path for merged model")
    merge_parser.add_argument('--cuda', action='store_true', help="Use CUDA")
    merge_parser.add_argument('--lazy-unpickle', action='store_true', help="Use lazy unpickling")
    merge_parser.add_argument('--allow-crimes', action='store_true', help="Allow questionable operations")

    # Extract LoRA parser
    extract_lora_parser = subparsers.add_parser("extract-lora", help="Extract LoRA")
    extract_lora_parser.add_argument('finetuned_model', type=str, help="Finetuned model ID or path")
    extract_lora_parser.add_argument('base_model', type=str, help="Base model ID or path")
    extract_lora_parser.add_argument('output', type=str, help="Output path")
    extract_lora_parser.add_argument('--rank', type=int, required=True, help="Desired rank")
    extract_lora_parser.add_argument('--no-lazy-unpickle', action='store_true', help="Disable lazy unpickling")

    # Gradio parser
    parser.add_argument("--gradio", action="store_true", help="Run the Gradio interface")

    return parser

def create_gradio_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# MergeKit Model Merger and LoRA Extractor")
        
        with gr.Tab("Merge Models"):
            config_file = gr.File(label="Configuration YAML")
            output_path = gr.Textbox(label="Output Path")
            cuda_checkbox = gr.Checkbox(label="Use CUDA")
            lazy_unpickle_checkbox = gr.Checkbox(label="Use Lazy Unpickling")
            allow_crimes_checkbox = gr.Checkbox(label="Allow Questionable Operations")
            merge_button = gr.Button("Run Merge")
            merge_result = gr.Textbox(label="Merge Result")
            
            def run_gradio_merge(config_file, output_path, cuda, lazy_unpickle, allow_crimes):
                if config_file is None:
                    return "Error: Please upload a configuration file."
                with open(config_file.name, 'r') as f:
                    config_content = f.read()
                temp_config_path = "temp_config.yml"
                with open(temp_config_path, 'w') as f:
                    f.write(config_content)
                return run_merge(temp_config_path, output_path, cuda, lazy_unpickle, allow_crimes)
            
            merge_button.click(
                run_gradio_merge,
                inputs=[config_file, output_path, cuda_checkbox, lazy_unpickle_checkbox, allow_crimes_checkbox],
                outputs=merge_result
            )
        
        with gr.Tab("Extract LoRA"):
            finetuned_model = gr.Textbox(label="Finetuned Model ID or Path")
            base_model = gr.Textbox(label="Base Model ID or Path")
            lora_output_path = gr.Textbox(label="Output Path")
            rank = gr.Number(label="Desired Rank", precision=0)
            no_lazy_unpickle_checkbox = gr.Checkbox(label="Disable Lazy Unpickling")
            extract_lora_button = gr.Button("Extract LoRA")
            extract_lora_result = gr.Textbox(label="Extraction Result")
            
            extract_lora_button.click(
                run_extract_lora,
                inputs=[finetuned_model, base_model, lora_output_path, rank, no_lazy_unpickle_checkbox],
                outputs=extract_lora_result
            )

    return interface

if __name__ == "__main__":
    parser = create_cli_parser()
    args = parser.parse_args()

    if args.gradio:
        interface = create_gradio_interface()
        interface.launch()
    else:
        if args.command == "merge":
            result = run_merge(args.config, args.output, args.cuda, args.lazy_unpickle, args.allow_crimes)
        elif args.command == "extract-lora":
            result = run_extract_lora(args.finetuned_model, args.base_model, args.output, args.rank, args.no_lazy_unpickle)
        else:
            parser.print_help()
            exit(1)
        
        print(result)