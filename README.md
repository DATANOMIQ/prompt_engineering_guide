# Prompt Engineering Workshop

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alexanderlammers/prompt_engineering_guide/blob/master/LICENSE)
[![Workshop](https://img.shields.io/badge/workshop-AI%20Convention%202025-green.svg)](https://ihk-schwaben.de)

A comprehensive interactive workshop for learning prompt engineering techniques with AI language models. This package provides practical examples, hands-on exercises, and real-world applications of advanced prompting strategies.

<p align="center">
  <img src="https://img.shields.io/badge/techniques-8+-brightgreen.svg" alt="8+ Techniques" />
  <img src="https://img.shields.io/badge/examples-50+-orange.svg" alt="50+ Examples" />
  <img src="https://img.shields.io/badge/interactive-yes-blue.svg" alt="Interactive" />
</p>

This workshop covers essential prompt engineering techniques from basic prompting to advanced frameworks like ReAct and Tree of Thoughts. An interactive Streamlit application provides hands-on experience with real AI models, while comprehensive Jupyter notebooks offer detailed explanations and examples.

Following details the functionality provided by the workshop:

- **8 Core Techniques**: From basic prompting to advanced reasoning frameworks
  - Basic Prompting - Simple text completions and natural language interactions
  - Instruction-based Prompting - Explicit task directions and structured guidance
  - Zero/One/Few-Shot Learning - Learning from examples with varying amounts of context
  - Chain-of-Thought Reasoning - Step-by-step logical problem solving
  - Self-Consistency - Multiple solution paths with consensus voting
  - Tree of Thoughts - Systematic exploration of solution spaces
  - ReAct Framework - Combining reasoning with external tool interaction
  - Real-world Applications - Practical implementations and use cases

- **Interactive Learning Environment**: Web-based interface for hands-on experimentation
- **Comprehensive Examples**: Over 50 practical examples across different domains
- **Model Flexibility**: Support for OpenAI GPT and Anthropic Claude models
- **Educational Materials**: Detailed notebooks with theory and implementation

The workshop is compatible with Python 3.8+ and runs on Linux, MacOS X and Windows.
It is distributed under the Apache 2.0 license.

## 📖 Contents

- Installation
- Quick Start
- Workshop Techniques
- Interactive Application
- Examples
- Configuration
- Contributing
- License

## ⚙️ Installation

There are two ways to set up the Prompt Engineering Workshop:

- **Clone from GitHub** (recommended):

```bash
git clone https://github.com/alexanderlammers/prompt_engineering_guide.git
cd prompt_engineering_guide
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- **Direct download**:

```bash
# Download and extract the workshop materials
# Install dependencies
pip install streamlit openai anthropic python-dotenv pandas numpy
```

### API Keys Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## 🚀 Quick Start

### Interactive Web Application

Launch the interactive Streamlit workshop:

```bash
streamlit run prompt_engineering_workshop_app.py
```

This opens a web interface where you can:

- Experiment with different prompting techniques
- Adjust model parameters in real-time
- See immediate results from AI models
- Follow guided tutorials for each technique

### Jupyter Notebooks

For detailed learning, explore the notebooks:

```bash
jupyter notebook notebooks/
```

Start with [`0_Environmental_Setup.ipynb`](notebooks/0_Environmental_Setup.ipynb ) to configure your environment.

### Basic Example

Here's a simple example of chain-of-thought reasoning:

```python
from modules.prompt_engineering_workshop import PromptEngineering

# Initialize the prompt engineering framework
pe = PromptEngineering()

# Chain-of-thought prompting example
problem = """
A library has 1,200 books on the first floor, 40% more books on the second floor, 
and 25% fewer books on the third floor than the second floor. How many books total?
"""

prompt = f"Problem: {problem}\n\nLet's solve this step-by-step:"
messages = [{"role": "user", "content": prompt}]

response = pe.get_completion(messages, temperature=0.1)
print(response)
```

### Advanced Framework Example

Using the ReAct framework with tools:

```python
from modules.react_framework import ReActAgent, CalculatorTool, WikipediaTool

# Initialize tools and agent
tools = [CalculatorTool(), WikipediaTool()]
agent = ReActAgent(pe, tools)

# Solve a complex problem requiring multiple tools
question = """
What's the area of a circle with radius 15 meters, and how does this 
compare to the area of Tokyo Disneyland?
"""

result = agent.solve_with_react(question)
print(result)
```

## 🎯 Workshop Techniques

### 1. Basic Prompting

Simple, natural language inputs that leverage the model's pretrained knowledge.

```python
prompt = "Explain quantum computing in simple terms"
```

### 2. Instruction-based Prompting

Explicit task directions with format specifications.

```python
prompt = """
Task: Classify the sentiment of the following text as positive, negative, or neutral.
Text: "This workshop is incredibly helpful for learning AI!"
Format: Output only the classification.
"""
```

### 3. Few-Shot Learning

Learning from examples with varying amounts of context.

```python
prompt = """
Examples:
Text: "Amazing product!" → Positive
Text: "Terrible service." → Negative
Text: "It's okay." → Neutral

Text: "This workshop exceeded my expectations!" → 
"""
```

### 4. Chain-of-Thought Reasoning

Step-by-step problem solving with explicit reasoning.

```python
prompt = """
Problem: A company's revenue increased by 15% in Q1, then decreased by 8% in Q2. 
If they started with $1M, what's their Q2 revenue?

Let's solve this step-by-step:
"""
```

### 5. Self-Consistency

Multiple solution paths with consensus voting.

```python
# Generate multiple solutions and find consensus
solutions = []
for i in range(5):
    response = pe.get_completion(messages, temperature=0.7)
    solutions.append(response)

consensus = find_consensus(solutions)
```

### 6. Tree of Thoughts

Systematic exploration of multiple solution branches.

```python
from modules.tree_of_thoughts import TreeOfThoughts

tot = TreeOfThoughts(pe, max_depth=3, breadth=3)
result = tot.solve_with_tree_of_thoughts(complex_problem)
```

### 7. ReAct Framework

Reasoning combined with external tool interaction.

```python
# Combines thinking, acting, and observing in iterative cycles
# Thought → Action → Observation → Thought → ...
```

## 🖥️ Interactive Application

The Streamlit application provides:

- **Real-time Experimentation**: Adjust parameters and see immediate results
- **Model Comparison**: Test different AI models (GPT-3.5, GPT-4, Claude)
- **Parameter Tuning**: Control temperature, top-p, max tokens
- **Guided Examples**: Step-by-step tutorials for each technique
- **Performance Insights**: Compare effectiveness across techniques

### Key Features

- **8 Interactive Sections**: One for each major technique
- **Model Flexibility**: Switch between OpenAI and Anthropic models
- **Parameter Control**: Fine-tune model behavior
- **Real-time Results**: See outputs as you experiment
- **Educational Content**: Built-in explanations and best practices

## 📚 Examples

The workshop includes comprehensive examples:

### Notebooks Structure

```
notebooks/
├── 0_Environmental_Setup.ipynb
├── 1_basic_prompting.ipynb
├── 2_instruction_prompting.ipynb
├── 3_zero_one_fewshot_prompting.ipynb
├── 4_chain_of_thought_reasoning_prompting.ipynb
├── 5_self_consistency_advanced_prompting.ipynb
├── 6_tree_of_thought_prompting.ipynb
├── 7_reason_and_act(ReAct).ipynb
└── 8_advanced_application_integration.ipynb
```

### Code Modules

```
modules/
├── basic_prompting.py
├── instruction_prompting.py
├── shot_prompting.py
├── chain_of_thought.py
├── self_consistency.py
├── tree_of_thoughts.py
├── react_framework.py
└── applications.py
```

## ⚙️ Configuration

### Model Parameters

Control AI model behavior through various parameters:

```python
# Temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
temperature = 0.7

# Top P: Nucleus sampling (0.5 = conservative, 1.0 = full vocabulary)
top_p = 0.9

# Max Tokens: Response length limit
max_tokens = 500

# Model Selection
model = "gpt-4"  # or "gpt-3.5-turbo", "claude-3-sonnet", etc.
```

### Environment Variables

Set up your API keys in `.env`:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key-here

# Anthropic Configuration  
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

## 🎓 Educational Approach

This workshop follows a progressive learning structure:

1. **Foundational Concepts**: Basic prompting and instruction-following
2. **Example-Based Learning**: Zero, one, and few-shot techniques
3. **Advanced Reasoning**: Chain-of-thought and self-consistency
4. **Complex Problem Solving**: Tree of thoughts and ReAct frameworks
5. **Real-World Applications**: Practical implementations and use cases

Each technique builds upon previous concepts, providing a comprehensive understanding of prompt engineering principles.

## 🤝 Contribute

We welcome contributions to improve the workshop:

- **Add Examples**: Contribute new prompting examples
- **Improve Documentation**: Enhance explanations and tutorials
- **Fix Issues**: Report and fix bugs
- **New Techniques**: Add emerging prompting strategies

See the contribution guidelines for more details.

## 📝 Workshop Materials

This workshop is based on the IHK Presentation: "Prompt Engineering: The Art of Communicating with AI" and includes:

- **Theoretical Foundation**: Academic research and best practices
- **Practical Implementation**: Hands-on coding examples
- **Interactive Learning**: Web-based experimentation platform
- **Real-World Applications**: Industry-relevant use cases

## 🏗 Maintainers

- Alexander Lammers, DATANOMIQ GmbH - [GitHub](https://github.com/alexanderlammers)

## © Copyright

Copyright 2025 DATANOMIQ GmbH. All rights reserved.

See [`LICENSE`](LICENSE ) for details.

---

*Built for AI Convention 2025 (IHK Schwaben) - Empowering developers with advanced prompt engineering techniques.*
