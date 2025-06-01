"""
Prompt Engineering Workshop - Comprehensive Guide
Based on materials from IHK Presentation: "Prompt Engineering: The Art of Communicating with AI"

This workshop provides practical examples and hands-on exercises for mastering prompt engineering.
"""

import os
import json
import time
import asyncio
import random
import requests
import re
from typing import List, Dict, Any, Optional
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import importlib.util
import sys
from dotenv import load_dotenv

# Check if OpenAI is installed, if not guide the user
try:
    import openai
except ImportError:
    print("OpenAI package not found. Please install using:")
    print("pip install openai dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

# OpenAI API configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    print("\n⚠️  WARNING: OpenAI API Key not found!")
    print("Please set your OPENAI_API_KEY in a .env file or as an environment variable")
    print("If you want to continue without setting the API key, example responses will be shown\n")


class PromptEngineering:
    """
    Main class for demonstrating prompt engineering concepts
    """
    def __init__(self, model="gpt-3.5-turbo", temperature=0.7, max_tokens=256):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.conversation_history = []
    
    def set_parameters(self, model=None, temperature=None, max_tokens=None, 
                      top_p=1, frequency_penalty=0, presence_penalty=0):
        """Configure OpenAI parameters for different demonstrations"""
        params = {
            'model': model or self.model,
            'temperature': temperature if temperature is not None else self.temperature,
            'max_tokens': max_tokens or self.max_tokens,
            'top_p': top_p,
            'frequency_penalty': frequency_penalty,
            'presence_penalty': presence_penalty
        }
        return params
    
    def get_completion(self, messages, **kwargs):
        """Get completion from OpenAI API with error handling"""
        params = self.set_parameters(**kwargs)
        
        if not openai.api_key:
            # Mock response when no API key is available
            return f"[Example response for: {messages[-1]['content'][:50]}...]"
        
        try:
            response = openai.chat.completions.create(
                model=params['model'],
                messages=messages,
                temperature=params['temperature'],
                max_tokens=params['max_tokens'],
                top_p=params['top_p'],
                frequency_penalty=params['frequency_penalty'],
                presence_penalty=params['presence_penalty']
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def display_result(self, prompt, response, concept="Basic Prompting"):
        """Display results in a formatted way for workshop demonstrations"""
        print(f"\n{'='*60}")
        print(f"CONCEPT: {concept}")
        print(f"{'='*60}")
        print(f"PROMPT:\n{prompt}")
        print(f"\nRESPONSE:\n{response}")
        print(f"{'='*60}\n")


# Import modules for each workshop section
from modules.basic_prompting import demonstrate_basic_prompting
from modules.instruction_prompting import demonstrate_instruction_prompting, demonstrate_good_prompt_structure
from modules.shot_prompting import demonstrate_shot_based_prompting, demonstrate_few_shot_learning_tasks
from modules.chain_of_thought import demonstrate_chain_of_thought, demonstrate_few_shot_cot
from modules.self_consistency import demonstrate_advanced_self_consistency
from modules.tree_of_thoughts import demonstrate_tree_of_thoughts
from modules.react_framework import demonstrate_react_framework
from modules.applications import PromptEngineeringWorkshop, RealWorldApplication, final_workshop_demonstration


def run_workshop():
    """
    Execute complete prompt engineering workshop
    """
    print("\n🚀 PROMPT ENGINEERING WORKSHOP 🚀")
    print("=" * 60)
    print("Welcome to this comprehensive hands-on workshop on prompt engineering!")
    print("We'll cover fundamental to advanced techniques with practical examples.")
    print("=" * 60)
    
    # Initialize main prompt engineering instance
    pe_demo = PromptEngineering()
    
    # Workshop menu
    while True:
        print("\nWORKSHOP SECTIONS:")
        print("1. Basic Prompting")
        print("2. Instruction-based Prompting") 
        print("3. Zero/One/Few-Shot Prompting")
        print("4. Chain-of-Thought Reasoning")
        print("5. Self-Consistency Techniques")
        print("6. Tree of Thoughts")
        print("7. ReAct Framework (Reasoning + Action)")
        print("8. Real-world Applications & Integration")
        print("9. Run Complete Workshop")
        print("0. Exit Workshop")
        
        choice = input("\nSelect section (0-9): ")
        
        if choice == '1':
            demonstrate_basic_prompting(pe_demo)
        elif choice == '2':
            demonstrate_instruction_prompting(pe_demo)
            demonstrate_good_prompt_structure(pe_demo)
        elif choice == '3':
            demonstrate_shot_based_prompting(pe_demo)
            demonstrate_few_shot_learning_tasks(pe_demo)
        elif choice == '4':
            demonstrate_chain_of_thought(pe_demo)
            demonstrate_few_shot_cot(pe_demo)
        elif choice == '5':
            demonstrate_advanced_self_consistency(pe_demo)
        elif choice == '6':
            demonstrate_tree_of_thoughts(pe_demo)
        elif choice == '7':
            demonstrate_react_framework(pe_demo)
        elif choice == '8':
            workshop = PromptEngineeringWorkshop(pe_demo)
            real_world = RealWorldApplication(pe_demo)
            final_workshop_demonstration(pe_demo, workshop, real_world)
        elif choice == '9':
            # Run complete workshop
            demonstrate_basic_prompting(pe_demo)
            demonstrate_instruction_prompting(pe_demo)
            demonstrate_good_prompt_structure(pe_demo)
            demonstrate_shot_based_prompting(pe_demo)
            demonstrate_few_shot_learning_tasks(pe_demo)
            demonstrate_chain_of_thought(pe_demo)
            demonstrate_few_shot_cot(pe_demo)
            demonstrate_advanced_self_consistency(pe_demo)
            demonstrate_tree_of_thoughts(pe_demo)
            demonstrate_react_framework(pe_demo)
            
            workshop = PromptEngineeringWorkshop(pe_demo)
            real_world = RealWorldApplication(pe_demo)
            final_workshop_demonstration(pe_demo, workshop, real_world)
            
            print("\n🎉 WORKSHOP COMPLETE! 🎉")
            print("You've successfully completed the prompt engineering workshop!")
        elif choice == '0':
            print("\nThank you for participating in the workshop!")
            break
        else:
            print("\nInvalid selection. Please try again.")


if __name__ == "__main__":
    # Create modules directory if it doesn't exist
    if not os.path.exists("modules"):
        os.makedirs("modules")
        # Create __init__.py to make it a proper package
        with open(os.path.join("modules", "__init__.py"), "w") as f:
            f.write("# Prompt Engineering Workshop Modules")
    
    # Check if modules are created, if not create them
    module_files = [
        "basic_prompting.py",
        "instruction_prompting.py",
        "shot_prompting.py", 
        "chain_of_thought.py",
        "self_consistency.py",
        "tree_of_thoughts.py",
        "react_framework.py",
        "applications.py"
    ]
    
    for module in module_files:
        if not os.path.exists(os.path.join("modules", module)):
            print(f"Creating module: {module}")
            # This will be implemented in the next steps
    
    # Run the workshop
    run_workshop()