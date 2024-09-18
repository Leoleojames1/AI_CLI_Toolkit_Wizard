#!/usr/bin/env python3

"""
🔧 Template CLI Structure

This script provides a template for creating command-line interfaces (CLIs) with multiple subcommands
and organized argument groups. Use this structure as a starting point for designing your own CLIs.

Key Features:
1. Multiple subcommands (e.g., command1, command2, command3)
2. Organized argument groups for each subcommand
3. Flexible argument types (e.g., positional, optional, flags)
4. Command-specific help messages
5. Main execution logic for each command

Usage example:
    python template_cli.py command1 --arg1 value1 --arg2 value2
    python template_cli.py command2 positional_arg --flag
    python template_cli.py command3 --help

Customize this template by:
1. Renaming commands and argument groups
2. Adding or removing arguments as needed
3. Implementing the logic for each command in the corresponding run_* function
4. Updating the main execution block to handle your specific commands
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command1(args):
    """
    Execute the logic for command1 based on the provided arguments.
    
    Args:
        args: Parsed command-line arguments
    """
    logger.info("Executing command1")
    # Implement command1 logic here
    pass

def run_command2(args):
    """
    Execute the logic for command2 based on the provided arguments.
    
    Args:
        args: Parsed command-line arguments
    """
    logger.info("Executing command2")
    # Implement command2 logic here
    pass

def run_command3(args):
    """
    Execute the logic for command3 based on the provided arguments.
    
    Args:
        args: Parsed command-line arguments
    """
    logger.info("Executing command3")
    # Implement command3 logic here
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🔧 Template CLI Structure")
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

    args = parser.parse_args()

    if args.command == "command1":
        run_command1(args)
    elif args.command == "command2":
        run_command2(args)
    elif args.command == "command3":
        run_command3(args)
    else:
        parser.print_help()