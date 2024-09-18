#!/usr/bin/env python3

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default path to llama.cpp directory
MODEL_GIT_DIR = PROJECT_ROOT.parent
DEFAULT_LLAMACPP_PATH = f"{MODEL_GIT_DIR}"

def convert_safetensor_to_gguf(input_dir, model_name, output_dir, quant_type='q8_0', llamacpp_path=DEFAULT_LLAMACPP_PATH):
    """
    Convert a SafeTensor model to GGUF format using llama.cpp's convert-hf-to-gguf command.
    """
    input_path = Path(input_dir) / model_name
    output_file = Path(output_dir) / f"{model_name}-{quant_type}.gguf"
    converter_path = Path(llamacpp_path) / "convert_hf_to_gguf.py"
    
    print(f"Debug: Input path: {input_path}")
    print(f"Debug: Output file: {output_file}")
    print(f"Debug: Converter path: {converter_path}")
    
    command = [
        sys.executable,
        str(converter_path),
        "--outtype", quant_type,
        "--outfile", str(output_file),
        str(input_path)
    ]
    
    print(f"Debug: Executing command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Debug: Subprocess output: {result.stdout}")
        print(f"Debug: Subprocess error: {result.stderr}")
        logger.info(f"Conversion successful: {model_name} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Debug: Subprocess error output: {e.output}")
        print(f"Debug: Subprocess error: {e.stderr}")
        logger.error(f"Conversion failed for {model_name}: {e}")
    except FileNotFoundError:
        logger.error(f"Converter not found at {converter_path}. Please check your llama.cpp path.")

def process_model(input_dir, model_name, output_dir, quant_types, llamacpp_path):
    """
    Process a model with multiple quantization types.
    """
    for quant_type in quant_types:
        convert_safetensor_to_gguf(input_dir, model_name, output_dir, quant_type, llamacpp_path)

def run_convert(args):
    """
    Execute the conversion of a single model to GGUF format.
    """
    logger.info(f"Converting model: {args.model_name}")
    process_model(args.input_dir, args.model_name, args.output_dir, [args.quant_type], args.llamacpp_path)

def run_process_dir(args):
    """
    Execute the processing of a directory containing multiple models.
    """
    logger.info(f"Processing directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    for model_name in os.listdir(args.input_dir):
        model_path = os.path.join(args.input_dir, model_name)
        if os.path.isdir(model_path):
            process_model(args.input_dir, model_name, args.output_dir, args.quant_types, args.llamacpp_path)

def run_interactive(args):
    """
    Run the converter in interactive mode, prompting the user for input.
    """
    logger.info("Running in interactive mode")
    input_dir = input("Enter the directory containing model folders: ").strip()
    output_dir = input("Enter the directory to save GGUF files: ").strip()
    model_name = input("Enter the name of the model to convert (or 'all' for all models): ").strip()
    quant_types = input("Enter quantization types (comma-separated, e.g., q8_0,f16,f32): ").strip().split(',')
    llamacpp_path = input(f"Enter the path to llama.cpp directory (press Enter for default: {DEFAULT_LLAMACPP_PATH}): ").strip()
    llamacpp_path = llamacpp_path if llamacpp_path else DEFAULT_LLAMACPP_PATH
    
    if model_name.lower() == 'all':
        run_process_dir(argparse.Namespace(input_dir=input_dir, output_dir=output_dir, quant_types=quant_types, llamacpp_path=llamacpp_path))
    else:
        process_model(input_dir, model_name, output_dir, quant_types, llamacpp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🔧 GGUF Converter CLI")
    parser.add_argument('--llamacpp_path', type=str, default=DEFAULT_LLAMACPP_PATH, help="Path to llama.cpp directory")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    convert_parser = subparsers.add_parser("convert", help="Convert a single model to GGUF")
    convert_parser.add_argument('--input_dir', type=str, required=True, help="Input directory containing model files")
    convert_parser.add_argument('--model_name', type=str, required=True, help="Name of the model to convert")
    convert_parser.add_argument('--output_dir', type=str, required=True, help="Output directory for GGUF files")
    convert_parser.add_argument('--quant_type', type=str, default='q8_0', choices=['q8_0', 'f16', 'f32'], help="Quantization type")

    process_dir_parser = subparsers.add_parser("process_dir", help="Process a directory of models")
    process_dir_parser.add_argument('--input_dir', type=str, required=True, help="Input directory containing model folders")
    process_dir_parser.add_argument('--output_dir', type=str, required=True, help="Output directory for GGUF files")
    process_dir_parser.add_argument('--quant_types', nargs='+', default=['q8_0', 'f16', 'f32'], help="Quantization types to process")

    interactive_parser = subparsers.add_parser("interactive", help="Run in interactive mode")

    args = parser.parse_args()

    if args.command == "convert":
        run_convert(args)
    elif args.command == "process_dir":
        run_process_dir(args)
    elif args.command == "interactive":
        run_interactive(args)
    else:
        parser.print_help()