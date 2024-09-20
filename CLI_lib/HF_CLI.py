#!/usr/bin/env python3

"""
🤗 Hugging Face CLI

This script provides a command-line interface for interacting with the Hugging Face Hub,
allowing users to pull models from and push models to the Hub.

Key Features:
1. Authenticate with Hugging Face Hub
2. Pull models from Hugging Face Hub
3. Push local models to Hugging Face Hub

Usage examples:
    python HF_CLI.py login --token "your_huggingface_token"
    python HF_CLI.py pull --model_name "bert-base-uncased"
    python HF_CLI.py push --local_path "./my_model" --repo_name "my-username/my-model"

Make sure to install required packages:
    pip install huggingface_hub
"""

import argparse
import logging
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, login

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Define paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
#TODO MAKE PARENT DIR
MODEL_GIT_DIR = PROJECT_ROOT.parent

def hf_login(args):
    """Login to Hugging Face Hub"""
    logger.info("Logging in to Hugging Face Hub")
    try:
        login(token=args.token)
        logger.info("Successfully logged in to Hugging Face Hub")
    except Exception as e:
        logger.error(f"Error logging in: {e}")

def pull_model(args):
    """Pull a model from HuggingFace Hub"""
    logger.info(f"Pulling model {args.model_name} from HuggingFace Hub")
    try:
        modeldir = args.model_name.split("/")[2]
        local_dir = MODEL_GIT_DIR / args.modelname
        snapshot_download(repo_id=args.model_name, local_dir=str(local_dir))
        logger.info(f"Model successfully downloaded to {local_dir}")
    except Exception as e:
        logger.error(f"Error pulling model: {e}")

def push_model(args):
    """Push a model to HuggingFace Hub"""
    logger.info(f"Pushing model from {args.local_path} to HuggingFace Hub as {args.repo_name}")
    try:
        api = HfApi()
        api.upload_folder(
            folder_path=args.local_path,
            repo_id=args.repo_name,
            repo_type="model"
        )
        logger.info(f"Model successfully pushed to HuggingFace Hub")
    except Exception as e:
        logger.error(f"Error pushing model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face CLI for model management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Login parser
    login_parser = subparsers.add_parser("login", help="Login to Hugging Face Hub")
    login_parser.add_argument('--token', type=str, required=True, help="Your Hugging Face API token")

    # Pull parser
    pull_parser = subparsers.add_parser("pull", help="Pull a model from HuggingFace Hub")
    pull_parser.add_argument('--model_name', type=str, required=True, help="Name of the model to pull")

    # Push parser
    push_parser = subparsers.add_parser("push", help="Push a model to HuggingFace Hub")
    push_parser.add_argument('--local_path', type=str, required=True, help="Local path of the model to push")
    push_parser.add_argument('--repo_name', type=str, required=True, help="Name of the repository on HuggingFace Hub")

    args = parser.parse_args()

    if args.command == "login":
        hf_login(args)
    elif args.command == "pull":
        pull_model(args)
    elif args.command == "push":
        push_model(args)
    else:
        parser.print_help()