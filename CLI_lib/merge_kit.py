#!/usr/bin/env python3

"""
🔧 MergeKit CLI and Gradio Interface

This script provides a command-line interface (CLI) and a Gradio web interface for merging language models using the mergekit library.

Key Features:
1. Multiple merge methods: SLERP, TIES, DARE, and Passthrough
2. CLI with subcommands for each merge method
3. Gradio interface for web-based interaction
4. Flexible configuration options for each merge method

Usage examples:
    python mergekit_cli_gradio.py slerp --model1 path/to/model1 --model2 path/to/model2 --t 0.5
    python mergekit_cli_gradio.py ties --models path/to/model1 path/to/model2 --densities 0.5 0.5 --weights 0.5 0.5
    python mergekit_cli_gradio.py dare --models path/to/model1 path/to/model2 --densities 0.5 0.5 --weights 0.4 0.6
    python mergekit_cli_gradio.py passthrough --model1 path/to/model1 --model2 path/to/model2 --layer_range1 0 32 --layer_range2 24 32
    python mergekit_cli_gradio.py --gradio

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_slerp(model1: str, model2: str, t: float, base_model: str, output_path: str) -> str:
    """Execute the SLERP merge method."""
    config = {
        "slices": [
            {
                "sources": [
                    {"model": model1, "layer_range": [0, 32]},
                    {"model": model2, "layer_range": [0, 32]}
                ]
            }
        ],
        "merge_method": "slerp",
        "base_model": base_model,
        "parameters": {"t": t},
        "dtype": "bfloat16"
    }
    
    # Here you would typically call the mergekit library to perform the merge
    # For demonstration, we'll just return the config as a string
    return f"SLERP merge config:\n{yaml.dump(config)}"

def run_ties(models: List[str], densities: List[float], weights: List[float], base_model: str, output_path: str) -> str:
    """Execute the TIES merge method."""
    config = {
        "models": [{"model": model, "parameters": {"density": d, "weight": w}} 
                   for model, d, w in zip(models, densities, weights)],
        "merge_method": "ties",
        "base_model": base_model,
        "parameters": {"normalize": True},
        "dtype": "float16"
    }
    
    return f"TIES merge config:\n{yaml.dump(config)}"

def run_dare(models: List[str], densities: List[float], weights: List[float], base_model: str, output_path: str, use_ties: bool) -> str:
    """Execute the DARE merge method."""
    config = {
        "models": [{"model": model, "parameters": {"density": d, "weight": w}} 
                   for model, d, w in zip(models, densities, weights)],
        "merge_method": "dare_ties" if use_ties else "dare_linear",
        "base_model": base_model,
        "parameters": {"int8_mask": True},
        "dtype": "bfloat16"
    }
    
    return f"DARE merge config:\n{yaml.dump(config)}"

def run_passthrough(model1: str, model2: str, layer_range1: List[int], layer_range2: List[int], output_path: str) -> str:
    """Execute the Passthrough merge method."""
    config = {
        "slices": [
            {"sources": [{"model": model1, "layer_range": layer_range1}]},
            {"sources": [{"model": model2, "layer_range": layer_range2}]}
        ],
        "merge_method": "passthrough",
        "dtype": "bfloat16"
    }
    
    return f"Passthrough merge config:\n{yaml.dump(config)}"

def create_cli_parser():
    parser = argparse.ArgumentParser(description="🔧 MergeKit CLI and Gradio Interface")
    parser.add_argument("--gradio", action="store_true", help="Run the Gradio interface")
    subparsers = parser.add_subparsers(dest="command", help="Available merge methods")
    
    # SLERP parser
    slerp_parser = subparsers.add_parser("slerp", help="Execute SLERP merge")
    slerp_parser.add_argument('--model1', type=str, required=True, help="Path to first model")
    slerp_parser.add_argument('--model2', type=str, required=True, help="Path to second model")
    slerp_parser.add_argument('--t', type=float, default=0.5, help="Interpolation factor")
    slerp_parser.add_argument('--base_model', type=str, required=True, help="Path to base model")
    slerp_parser.add_argument('--output', type=str, required=True, help="Output path for merged model")

    # TIES parser
    ties_parser = subparsers.add_parser("ties", help="Execute TIES merge")
    ties_parser.add_argument('--models', nargs='+', required=True, help="Paths to models to merge")
    ties_parser.add_argument('--densities', nargs='+', type=float, required=True, help="Density for each model")
    ties_parser.add_argument('--weights', nargs='+', type=float, required=True, help="Weight for each model")
    ties_parser.add_argument('--base_model', type=str, required=True, help="Path to base model")
    ties_parser.add_argument('--output', type=str, required=True, help="Output path for merged model")

    # DARE parser
    dare_parser = subparsers.add_parser("dare", help="Execute DARE merge")
    dare_parser.add_argument('--models', nargs='+', required=True, help="Paths to models to merge")
    dare_parser.add_argument('--densities', nargs='+', type=float, required=True, help="Density for each model")
    dare_parser.add_argument('--weights', nargs='+', type=float, required=True, help="Weight for each model")
    dare_parser.add_argument('--base_model', type=str, required=True, help="Path to base model")
    dare_parser.add_argument('--output', type=str, required=True, help="Output path for merged model")
    dare_parser.add_argument('--use_ties', action='store_true', help="Use TIES sign election")

    # Passthrough parser
    passthrough_parser = subparsers.add_parser("passthrough", help="Execute Passthrough merge")
    passthrough_parser.add_argument('--model1', type=str, required=True, help="Path to first model")
    passthrough_parser.add_argument('--model2', type=str, required=True, help="Path to second model")
    passthrough_parser.add_argument('--layer_range1', nargs=2, type=int, required=True, help="Layer range for first model")
    passthrough_parser.add_argument('--layer_range2', nargs=2, type=int, required=True, help="Layer range for second model")
    passthrough_parser.add_argument('--output', type=str, required=True, help="Output path for merged model")

    return parser

def create_gradio_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# MergeKit Model Merger")
        
        with gr.Tab("SLERP"):
            slerp_model1 = gr.Textbox(label="Model 1 Path")
            slerp_model2 = gr.Textbox(label="Model 2 Path")
            slerp_t = gr.Slider(minimum=0, maximum=1, value=0.5, label="Interpolation Factor (t)")
            slerp_base_model = gr.Textbox(label="Base Model Path")
            slerp_output = gr.Textbox(label="Output Path")
            slerp_button = gr.Button("Run SLERP Merge")
            slerp_result = gr.Textbox(label="Merge Config")
            
            slerp_button.click(
                run_slerp,
                inputs=[slerp_model1, slerp_model2, slerp_t, slerp_base_model, slerp_output],
                outputs=slerp_result
            )
        
        with gr.Tab("TIES"):
            ties_models = gr.Textbox(label="Model Paths (space-separated)")
            ties_densities = gr.Textbox(label="Densities (space-separated)")
            ties_weights = gr.Textbox(label="Weights (space-separated)")
            ties_base_model = gr.Textbox(label="Base Model Path")
            ties_output = gr.Textbox(label="Output Path")
            ties_button = gr.Button("Run TIES Merge")
            ties_result = gr.Textbox(label="Merge Config")
            
            ties_button.click(
                lambda *args: run_ties(
                    args[0].split(),
                    [float(d) for d in args[1].split()],
                    [float(w) for w in args[2].split()],
                    args[3],
                    args[4]
                ),
                inputs=[ties_models, ties_densities, ties_weights, ties_base_model, ties_output],
                outputs=ties_result
            )
        
        with gr.Tab("DARE"):
            dare_models = gr.Textbox(label="Model Paths (space-separated)")
            dare_densities = gr.Textbox(label="Densities (space-separated)")
            dare_weights = gr.Textbox(label="Weights (space-separated)")
            dare_base_model = gr.Textbox(label="Base Model Path")
            dare_output = gr.Textbox(label="Output Path")
            dare_use_ties = gr.Checkbox(label="Use TIES Sign Election")
            dare_button = gr.Button("Run DARE Merge")
            dare_result = gr.Textbox(label="Merge Config")
            
            dare_button.click(
                lambda *args: run_dare(
                    args[0].split(),
                    [float(d) for d in args[1].split()],
                    [float(w) for w in args[2].split()],
                    args[3],
                    args[4],
                    args[5]
                ),
                inputs=[dare_models, dare_densities, dare_weights, dare_base_model, dare_output, dare_use_ties],
                outputs=dare_result
            )
        
        with gr.Tab("Passthrough"):
            passthrough_model1 = gr.Textbox(label="Model 1 Path")
            passthrough_model2 = gr.Textbox(label="Model 2 Path")
            passthrough_layer_range1 = gr.Textbox(label="Layer Range for Model 1 (start end)")
            passthrough_layer_range2 = gr.Textbox(label="Layer Range for Model 2 (start end)")
            passthrough_output = gr.Textbox(label="Output Path")
            passthrough_button = gr.Button("Run Passthrough Merge")
            passthrough_result = gr.Textbox(label="Merge Config")
            
            passthrough_button.click(
                lambda *args: run_passthrough(
                    args[0],
                    args[1],
                    [int(lr) for lr in args[2].split()],
                    [int(lr) for lr in args[3].split()],
                    args[4]
                ),
                inputs=[passthrough_model1, passthrough_model2, passthrough_layer_range1, passthrough_layer_range2, passthrough_output],
                outputs=passthrough_result
            )

    return interface

if __name__ == "__main__":
    parser = create_cli_parser()
    args = parser.parse_args()

    if args.gradio:
        interface = create_gradio_interface()
        interface.launch()
    else:
        if args.command == "slerp":
            result = run_slerp(args.model1, args.model2, args.t, args.base_model, args.output)
        elif args.command == "ties":
            result = run_ties(args.models, args.densities, args.weights, args.base_model, args.output)
        elif args.command == "dare":
            result = run_dare(args.models, args.densities, args.weights, args.base_model, args.output, args.use_ties)
        elif args.command == "passthrough":
            result = run_passthrough(args.model1, args.model2, args.layer_range1, args.layer_range2, args.output)
        else:
            parser.print_help()
            exit(1)
        
        print(result)