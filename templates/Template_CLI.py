#!/usr/bin/env python3

"""
🔧 Template CLI Structure with Gradio Integration

This script provides a template for creating command-line interfaces (CLIs) with multiple subcommands
and organized argument groups. It also includes a Gradio interface option for web-based interaction.

Key Features:
1. Multiple subcommands (e.g., command1, command2, command3)
2. Organized argument groups for each subcommand
3. Flexible argument types (e.g., positional, optional, flags)
4. Command-specific help messages
5. Main execution logic for each command
6. Gradio interface for web-based interaction

Usage examples:
    python template_cli.py command1 --arg1 value1 --arg2 value2
    python template_cli.py command2 positional_arg --flag
    python template_cli.py command3 --help
    python template_cli.py --gradio

Customize this template by:
1. Renaming commands and argument groups
2. Adding or removing arguments as needed
3. Implementing the logic for each command in the corresponding run_* function
4. Updating the main execution block to handle your specific commands
5. Modifying the Gradio interface to match your CLI functionality
"""

import argparse
import logging
import gradio as gr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command1(arg1, arg2, flag1, option1):
    """Execute the logic for command1 based on the provided arguments."""
    logger.info("Executing command1")
    return f"Command1 executed with arg1={arg1}, arg2={arg2}, flag1={flag1}, option1={option1}"

def run_command2(positional_arg, flag):
    """Execute the logic for command2 based on the provided arguments."""
    logger.info("Executing command2")
    return f"Command2 executed with positional_arg={positional_arg}, flag={flag}"

def run_command3(input_value, output_value):
    """Execute the logic for command3 based on the provided arguments."""
    logger.info("Executing command3")
    return f"Command3 executed with input={input_value}, output={output_value}"

def create_cli_parser():
    parser = argparse.ArgumentParser(description="🔧 Template CLI Structure")
    parser.add_argument("--gradio", action="store_true", help="Run the Gradio interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Command1 parser
    command1_parser = subparsers.add_parser("command1", help="Execute command1")
    group1 = command1_parser.add_argument_group("Group 1")
    group1.add_argument('--arg1', type=str, required=True, help="Description for arg1")
    group1.add_argument('--arg2', type=int, default=0, help="Description for arg2")
    group2 = command1_parser.add_argument_group("Group 2")
    group2.add_argument('--flag1', action='store_true', help="Flag 1 for command1")
    group2.add_argument('--option1', choices=['a', 'b', 'c'], default='a', help="Option 1 for command1")

    # Command2 parser
    command2_parser = subparsers.add_parser("command2", help="Execute command2")
    command2_parser.add_argument('positional_arg', type=str, help="Positional argument for command2")
    command2_parser.add_argument('--flag', action='store_true', help="Flag for command2")

    # Command3 parser
    command3_parser = subparsers.add_parser("command3", help="Execute command3")
    group3 = command3_parser.add_argument_group("Group 3")
    group3.add_argument('--input', type=str, required=True, help="Input for command3")
    group3.add_argument('--output', type=str, required=True, help="Output for command3")

    return parser

def create_gradio_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# CLI Template with Gradio Interface")
        
        with gr.Tab("Command 1"):
            arg1_input = gr.Textbox(label="Arg1")
            arg2_input = gr.Number(label="Arg2", value=0)
            flag1_input = gr.Checkbox(label="Flag1")
            option1_input = gr.Radio(["a", "b", "c"], label="Option1", value="a")
            command1_button = gr.Button("Run Command 1")
            command1_output = gr.Textbox(label="Output")
            
            command1_button.click(
                run_command1,
                inputs=[arg1_input, arg2_input, flag1_input, option1_input],
                outputs=command1_output
            )
        
        with gr.Tab("Command 2"):
            positional_arg_input = gr.Textbox(label="Positional Argument")
            flag_input = gr.Checkbox(label="Flag")
            command2_button = gr.Button("Run Command 2")
            command2_output = gr.Textbox(label="Output")
            
            command2_button.click(
                run_command2,
                inputs=[positional_arg_input, flag_input],
                outputs=command2_output
            )
        
        with gr.Tab("Command 3"):
            input_value = gr.Textbox(label="Input")
            output_value = gr.Textbox(label="Output")
            command3_button = gr.Button("Run Command 3")
            command3_output = gr.Textbox(label="Result")
            
            command3_button.click(
                run_command3,
                inputs=[input_value, output_value],
                outputs=command3_output
            )

    return interface

if __name__ == "__main__":
    parser = create_cli_parser()
    args = parser.parse_args()

    if args.gradio:
        interface = create_gradio_interface()
        interface.launch()
    else:
        if args.command == "command1":
            result = run_command1(args.arg1, args.arg2, args.flag1, args.option1)
        elif args.command == "command2":
            result = run_command2(args.positional_arg, args.flag)
        elif args.command == "command3":
            result = run_command3(args.input, args.output)
        else:
            parser.print_help()
            exit(1)
        
        print(result)