"""
Prompt Engineering Guide - Hands-On Tutorial
Based on IHK Presentation: "Prompt Engineering: The Art of Communicating with AI"

This guide provides practical examples for each concept covered in the presentation.
"""

from typing import List, Dict, Any
import os
import sys
import openai
from dotenv import load_dotenv

# Check if necessary directories exist
if not os.path.exists("modules"):
    print("Creating modules directory...")
    os.makedirs("modules")
    with open(os.path.join("modules", "__init__.py"), "w") as f:
        f.write("# Prompt Engineering Workshop Modules")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class PromptEngineeringTutorial:
    """
    Interactive tutorial for learning prompt engineering concepts
    """

    def __init__(self, api_key: str = None):
        """Initialize with OpenAI API key (optional for demonstration)"""
        # Load API key from environment if not provided
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
        self.examples = []

    def demonstrate_concept(self, concept_name: str, prompt: str, explanation: str):
        """Template for demonstrating each concept"""
        print(f"\n{'=' * 60}")
        print(f"CONCEPT: {concept_name.upper()}")
        print(f"{'=' * 60}")
        print(f"\nExplanation: {explanation}")
        print(f"\nPrompt Example:")
        print(f"'{prompt}'")
        print(f"\n{'-' * 40}")

        # Store example for later reference
        self.examples.append(
            {"concept": concept_name, "prompt": prompt, "explanation": explanation}
        )

    def run_comprehensive_workshop(self):
        """Run the complete workshop from the main file"""
        try:
            # Import the workshop runner
            from modules.prompt_engineering_workshop import run_workshop

            run_workshop()
        except ImportError:
            print("\n⚠️ Workshop module not found. Please run setup first.")
            print("You can run: python setup_workshop.py")


def setup_workshop():
    """
    Set up the workshop by creating all necessary files
    """
    # Create module files if they don't exist yet
    module_files = [
        "basic_prompting.py",
        "instruction_prompting.py",
        "shot_prompting.py",
        "chain_of_thought.py",
        "self_consistency.py",
        "tree_of_thoughts.py",
        "react_framework.py",
        "applications.py",
    ]

    # Import template content
    try:
        from setup_templates import get_module_content

        if not os.path.exists("modules"):
            os.makedirs("modules")
            with open(os.path.join("modules", "__init__.py"), "w") as f:
                f.write("# Prompt Engineering Workshop Modules")

        for module_file in module_files:
            file_path = os.path.join("modules", module_file)
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    content = get_module_content(module_file)
                    f.write(content)
                print(f"Created module: {module_file}")

        # Create main workshop file
        if not os.path.exists("prompt_engineering_workshop.py"):
            with open("prompt_engineering_workshop.py", "w") as f:
                f.write(
                    '"""Prompt Engineering Workshop - Main Runner"""\n\n'
                    "from modules.basic_prompting import run_basic_prompting\n"
                    "from modules.instruction_prompting import run_instruction_prompting\n"
                    "from modules.shot_prompting import run_shot_prompting\n"
                    "from modules.chain_of_thought import run_chain_of_thought\n"
                    "from modules.self_consistency import run_self_consistency\n"
                    "from modules.tree_of_thoughts import run_tree_of_thoughts\n"
                    "from modules.react_framework import run_react_framework\n"
                    "from modules.applications import run_applications\n\n"
                    "def run_workshop():\n"
                    '    print("🚀 Running Prompt Engineering Workshop...")\n\n'
                    "    # Run all modules\n"
                    "    run_basic_prompting()\n"
                    "    run_instruction_prompting()\n"
                    "    run_shot_prompting()\n"
                    "    run_chain_of_thought()\n"
                    "    run_self_consistency()\n"
                    "    run_tree_of_thoughts()\n"
                    "    run_react_framework()\n"
                    "    run_applications()\n\n"
                    '    print("🎉 Workshop complete! Explore the modules directory for details.")\n'
                )
                print("Created main workshop file: prompt_engineering_workshop.py")

    except ImportError:
        print(
            "Error: setup_templates module not found. Ensure it's in the Python path."
        )
    except Exception as e:
        print(f"Error during setup: {e}")


# =============================================================================
# PART 1: FOUNDATION CONCEPTS (Slides 1-5)
# =============================================================================


def part1_foundations():
    """Cover the foundational concepts of prompt engineering"""

    tutorial = PromptEngineeringTutorial()

    print("PART 1: FOUNDATIONS OF PROMPT ENGINEERING")
    print("=" * 60)

    # 1.1 What is Prompt Engineering?
    tutorial.demonstrate_concept(
        "Basic Prompt Structure",
        "You are a helpful assistant. Explain quantum computing in simple terms for a 10-year-old.",
        "A prompt has three parts: role definition, task specification, and context/constraints",
    )

    # 1.2 Poor vs Good Prompts
    print(f"\n{'=' * 60}")
    print("COMPARISON: POOR vs GOOD PROMPTS")
    print(f"{'=' * 60}")

    poor_prompt = "Write about AI"
    good_prompt = """You are an AI expert writing for business executives. 
    Write a 200-word executive summary about how AI can improve customer service, 
    including 3 specific examples and potential ROI."""

    print("❌ POOR PROMPT:")
    print(f"'{poor_prompt}'")
    print("\nProblems: Vague, no context, no specific requirements")

    print("\n✅ GOOD PROMPT:")
    print(f"'{good_prompt}'")
    print(
        "\nStrengths: Clear role, specific task, defined length, concrete examples requested"
    )


# =============================================================================
# PART 2: CORE TECHNIQUES (Slides 6-12)
# =============================================================================


def part2_core_techniques():
    """Demonstrate core prompting techniques"""

    tutorial = PromptEngineeringTutorial()

    print(f"\n\nPART 2: CORE PROMPTING TECHNIQUES")
    print("=" * 60)

    # 2.1 Zero-Shot Prompting
    tutorial.demonstrate_concept(
        "Zero-Shot Prompting",
        "Classify the sentiment of this text as positive, negative, or neutral: 'The weather today is absolutely perfect for a picnic!'",
        "Give the model a task without any examples. Works well for simple, well-defined tasks.",
    )

    # 2.2 Few-Shot Prompting
    few_shot_prompt = """Classify these product reviews as positive, negative, or neutral:

Example 1: "This phone is amazing, great battery life!" → Positive
Example 2: "Terrible customer service, very disappointed." → Negative  
Example 3: "The product arrived on time, standard quality." → Neutral

Now classify: "Absolutely love this product, exceeded expectations!"
Classification:"""

    tutorial.demonstrate_concept(
        "Few-Shot Prompting",
        few_shot_prompt,
        "Provide 2-5 examples to teach the model the pattern. More effective for complex or nuanced tasks.",
    )

    # 2.3 Chain-of-Thought Prompting
    cot_prompt = """Solve this step by step:

Problem: A store has 150 items. On Monday, they sell 30% of their items. On Tuesday, they sell 20 more items. How many items are left?

Let me think through this step by step:
1) First, calculate 30% of 150 items sold on Monday
2) Then subtract Monday's sales from the original amount
3) Then subtract Tuesday's sales
4) The result is the remaining items

Step 1:"""

    tutorial.demonstrate_concept(
        "Chain-of-Thought Prompting",
        cot_prompt,
        "Guide the model to break down complex problems into logical steps. Improves accuracy for reasoning tasks.",
    )

    # 2.4 Role-Based Prompting
    tutorial.demonstrate_concept(
        "Role-Based Prompting",
        "You are a senior software architect with 15 years of experience. A junior developer asks: 'Should I use microservices for a simple blog application?' Provide a thoughtful, mentoring response.",
        "Assign a specific role/persona to get responses with appropriate expertise and tone.",
    )


# =============================================================================
# PART 3: ADVANCED TECHNIQUES (Slides 13-18)
# =============================================================================


def part3_advanced_techniques():
    """Demonstrate advanced prompting strategies"""

    tutorial = PromptEngineeringTutorial()

    print(f"\n\nPART 3: ADVANCED TECHNIQUES")
    print("=" * 60)

    # 3.1 Prompt Chaining
    print(f"\n{'=' * 60}")
    print("PROMPT CHAINING EXAMPLE")
    print(f"{'=' * 60}")

    chain_prompts = [
        "Step 1 - Research: List 5 current trends in renewable energy technology.",
        "Step 2 - Analysis: Take the trends from Step 1 and analyze which has the most commercial potential.",
        "Step 3 - Strategy: Based on the analysis from Step 2, create a 3-point business strategy for a startup.",
    ]

    for i, prompt in enumerate(chain_prompts, 1):
        print(f"\n{prompt}")
        if i < len(chain_prompts):
            print("↓ (Output becomes input for next prompt)")

    print(
        "\nBenefits: Breaks complex tasks into manageable steps, each optimized for specific sub-goals"
    )

    # 3.2 Constitutional AI / Self-Correction
    constitutional_prompt = """First, provide your initial response to this question: "How do I get revenge on someone who wronged me?"

Then, review your response and consider:
1) Does this promote harmful behavior?
2) Are there more constructive alternatives?
3) What would be the ethical approach?

Provide a revised, more helpful response:"""

    tutorial.demonstrate_concept(
        "Constitutional AI / Self-Correction",
        constitutional_prompt,
        "Have the model critique and improve its own responses based on ethical guidelines or quality criteria.",
    )

    # 3.3 Tree of Thoughts
    tree_of_thoughts_prompt = """Problem: Design a mobile app for busy parents to manage family schedules.

Generate 3 different approaches:

Approach A: [Focus on calendar integration]
- Pros:
- Cons:
- Key features:

Approach B: [Focus on task delegation]  
- Pros:
- Cons:
- Key features:

Approach C: [Focus on communication hub]
- Pros:
- Cons: 
- Key features:

Now evaluate each approach and recommend the best one with reasoning:"""

    tutorial.demonstrate_concept(
        "Tree of Thoughts",
        tree_of_thoughts_prompt,
        "Explore multiple solution paths simultaneously, then evaluate and choose the best approach.",
    )


# =============================================================================
# PART 4: DOMAIN-SPECIFIC APPLICATIONS (Slides 19-23)
# =============================================================================


def part4_domain_applications():
    """Show prompting techniques for specific domains"""

    tutorial = PromptEngineeringTutorial()

    print(f"\n\nPART 4: DOMAIN-SPECIFIC APPLICATIONS")
    print("=" * 60)

    # 4.1 Code Generation
    code_prompt = """You are an expert Python developer. Create a function that:

Requirements:
- Takes a list of dictionaries representing students with 'name' and 'grades' keys
- Calculates the average grade for each student  
- Returns a sorted list of students by average grade (highest first)
- Include proper error handling and type hints
- Add docstring with examples

Provide clean, production-ready code:"""

    tutorial.demonstrate_concept(
        "Code Generation",
        code_prompt,
        "Specify requirements clearly, request best practices like error handling and documentation.",
    )

    # 4.2 Data Analysis
    data_analysis_prompt = """You are a data analyst. I have sales data with these columns: date, product_id, quantity, price, customer_segment.

I want to understand: "Which customer segment generates the most revenue and shows the highest growth?"

Provide:
1) SQL queries to extract the relevant data
2) Python code using pandas to analyze the results  
3) Suggested visualizations
4) Key insights to look for

Analysis approach:"""

    tutorial.demonstrate_concept(
        "Data Analysis",
        data_analysis_prompt,
        "Break down analytical tasks into data extraction, processing, visualization, and interpretation steps.",
    )

    # 4.3 Creative Writing
    creative_prompt = """You are a creative writing coach. Help me write a compelling opening paragraph for a sci-fi short story.

Constraints:
- Setting: Mars colony in 2157
- Protagonist: Young engineer discovering something unexpected
- Tone: Mysterious but hopeful
- Length: 3-4 sentences
- Hook: Start with an intriguing sensory detail

First, brainstorm 3 different opening approaches, then write the best one:"""

    tutorial.demonstrate_concept(
        "Creative Writing",
        creative_prompt,
        "Provide specific constraints for setting, character, tone, and structure. Request brainstorming first.",
    )

    # 4.4 Business Analysis
    business_prompt = """You are a senior business consultant. A mid-size manufacturing company (500 employees, $50M revenue) wants to implement AI to improve operations.

Analyze this scenario:
- Current challenges: Manual quality control, inventory forecasting issues, customer service bottlenecks
- Budget: $2M over 18 months  
- Technical capability: Limited AI expertise in-house

Provide:
1) Prioritized list of AI implementation opportunities
2) ROI estimates for top 3 opportunities
3) Implementation roadmap with phases
4) Risk assessment and mitigation strategies

Analysis:"""

    tutorial.demonstrate_concept(
        "Business Analysis",
        business_prompt,
        "Provide detailed context about company size, challenges, constraints, and request structured analysis.",
    )


# =============================================================================
# PART 5: OPTIMIZATION STRATEGIES (Slides 24-28)
# =============================================================================


def part5_optimization():
    """Demonstrate prompt optimization techniques"""

    print(f"\n\nPART 5: PROMPT OPTIMIZATION STRATEGIES")
    print("=" * 60)

    # 5.1 Iterative Refinement
    print(f"\n{'=' * 60}")
    print("ITERATIVE REFINEMENT EXAMPLE")
    print(f"{'=' * 60}")

    iterations = [
        ("Version 1", "Write about customer service"),
        ("Version 2", "Write tips for good customer service"),
        (
            "Version 3",
            "Write 5 actionable tips for improving customer service in retail",
        ),
        (
            "Version 4",
            "You are a retail operations expert. Write 5 specific, actionable tips for improving customer service in retail stores. For each tip, include: the specific action, why it works, and a real example. Target audience: store managers.",
        ),
    ]

    for version, prompt in iterations:
        print(f"\n{version}: '{prompt}'")
        if version != "Version 4":
            print("↓ (Adding more specificity)")

    print(
        f"\nKey improvements: Added expertise, specific format, clear audience, actionable focus"
    )

    # 5.2 A/B Testing Framework
    print(f"\n{'=' * 60}")
    print("A/B TESTING FRAMEWORK")
    print(f"{'=' * 60}")

    ab_test_code = '''
def test_prompt_variants(base_task, variants, test_inputs, evaluation_criteria):
    """
    Framework for testing different prompt approaches
    """
    results = {}
    
    for variant_name, prompt_template in variants.items():
        variant_results = []
        
        for test_input in test_inputs:
            # Format prompt with test input
            full_prompt = prompt_template.format(input=test_input)
            
            # Get AI response (placeholder)
            response = get_ai_response(full_prompt)
            
            # Evaluate response
            score = evaluate_response(response, evaluation_criteria)
            
            variant_results.append({
                'input': test_input,
                'response': response, 
                'score': score
            })
        
        results[variant_name] = variant_results
    
    return analyze_results(results)

# Example usage:
variants = {
    "direct": "Summarize this text: {input}",
    "detailed": "You are a professional editor. Create a concise summary of the following text, highlighting the 3 most important points: {input}",
    "structured": "Summarize this text in exactly 3 bullet points, each under 20 words: {input}"
}

evaluation_criteria = ['accuracy', 'conciseness', 'clarity', 'completeness']
'''

    print(ab_test_code)

    # 5.3 Evaluation Metrics
    print(f"\n{'=' * 60}")
    print("EVALUATION METRICS FOR DIFFERENT TASKS")
    print(f"{'=' * 60}")

    metrics = {
        "Code Generation": ["Correctness", "Efficiency", "Readability", "Security"],
        "Creative Writing": [
            "Creativity",
            "Coherence",
            "Tone consistency",
            "Engagement",
        ],
        "Data Analysis": ["Accuracy", "Insight quality", "Methodology", "Clarity"],
        "Customer Service": [
            "Helpfulness",
            "Politeness",
            "Problem resolution",
            "Brevity",
        ],
        "Technical Documentation": ["Completeness", "Accuracy", "Clarity", "Structure"],
    }

    for task, task_metrics in metrics.items():
        print(f"\n{task}:")
        for metric in task_metrics:
            print(f"  • {metric}")


# =============================================================================
# PART 6: BEST PRACTICES & COMMON PITFALLS (Slides 29-33)
# =============================================================================


def part6_best_practices():
    """Cover best practices and common mistakes"""

    print(f"\n\nPART 6: BEST PRACTICES & COMMON PITFALLS")
    print("=" * 60)

    # 6.1 Common Pitfalls
    print(f"\n{'=' * 60}")
    print("COMMON PITFALLS TO AVOID")
    print(f"{'=' * 60}")

    pitfalls = [
        {
            "pitfall": "Ambiguous Instructions",
            "bad": "Make this better",
            "good": "Improve this email's tone to be more professional while keeping it friendly. Focus on clarity and conciseness.",
            "why": "Specific criteria for improvement",
        },
        {
            "pitfall": "Too Many Tasks at Once",
            "bad": "Analyze this data, create visualizations, write a report, and suggest next steps",
            "good": "First, analyze this sales data and identify the top 3 trends. I'll ask for visualizations in a follow-up.",
            "why": "Single focus leads to better results",
        },
        {
            "pitfall": "No Context Provided",
            "bad": "Is this a good strategy?",
            "good": "You are a marketing consultant. For a B2B SaaS startup with $1M ARR, is account-based marketing a good strategy? Consider our limited budget and 2-person marketing team.",
            "why": "Context enables appropriate advice",
        },
    ]

    for item in pitfalls:
        print(f"\n❌ {item['pitfall']}:")
        print(f"Bad: '{item['bad']}'")
        print(f"✅ Good: '{item['good']}'")
        print(f"Why: {item['why']}")

    # 6.2 Best Practices Checklist
    print(f"\n{'=' * 60}")
    print("BEST PRACTICES CHECKLIST")
    print(f"{'=' * 60}")

    checklist = [
        "✅ Define clear role/persona for the AI",
        "✅ Specify the exact task and desired outcome",
        "✅ Provide relevant context and constraints",
        "✅ Include examples when helpful (few-shot)",
        "✅ Request specific format/structure",
        "✅ Set appropriate length/scope",
        "✅ Use iterative refinement",
        "✅ Test with multiple inputs",
        "✅ Specify tone and style",
        "✅ Include evaluation criteria",
    ]

    for item in checklist:
        print(item)

    # 6.3 Ethical Considerations
    print(f"\n{'=' * 60}")
    print("ETHICAL CONSIDERATIONS")
    print(f"{'=' * 60}")

    ethical_prompt = """Before using any AI-generated content, consider:

1. Bias Check: "Could this response reflect harmful biases?"
2. Accuracy Verification: "Do I need to fact-check this information?"  
3. Attribution: "Should I disclose that AI was used?"
4. Impact Assessment: "What are the potential consequences of using this?"
5. Human Oversight: "Does this need human review before implementation?"

Example ethical prompt addition:
"Please note any assumptions you're making and suggest where human expertise would be valuable."
"""

    print(ethical_prompt)


# =============================================================================
# PART 7: PRACTICAL EXERCISES
# =============================================================================


def part7_exercises():
    """Hands-on exercises for practice"""

    print(f"\n\nPART 7: HANDS-ON EXERCISES")
    print("=" * 60)

    exercises = [
        {
            "title": "Exercise 1: Email Improvement",
            "task": "Transform this email to be more professional",
            "input": "hey, need the report asap. thanks.",
            "requirements": [
                "Professional tone",
                "Specific deadline",
                "Context for urgency",
                "Polite but clear",
            ],
        },
        {
            "title": "Exercise 2: Code Review",
            "task": "Create a prompt to review Python code for best practices",
            "input": "A function that processes user data",
            "requirements": [
                "Check for security issues",
                "Suggest performance improvements",
                "Verify error handling",
                "Assess readability",
            ],
        },
        {
            "title": "Exercise 3: Business Strategy",
            "task": "Analyze market entry strategy for a new product",
            "input": "AI-powered fitness app for seniors",
            "requirements": [
                "Market size assessment",
                "Competition analysis",
                "Go-to-market strategy",
                "Risk evaluation",
            ],
        },
    ]

    for i, exercise in enumerate(exercises, 1):
        print(f"\n{exercise['title']}")
        print("-" * 40)
        print(f"Task: {exercise['task']}")
        print(f"Input: {exercise['input']}")
        print("Requirements:")
        for req in exercise["requirements"]:
            print(f"  • {req}")
        print(f"\nYour turn: Write a prompt for this exercise")
        print("(Think about role, context, specific instructions, format)")


# =============================================================================
# MAIN TUTORIAL RUNNER
# =============================================================================


def run_complete_tutorial():
    """Run the complete hands-on tutorial"""

    print("🤖 PROMPT ENGINEERING: HANDS-ON TUTORIAL")
    print("Based on IHK Presentation")
    print("=" * 60)

    # Run all parts
    part1_foundations()
    part2_core_techniques()
    part3_advanced_techniques()
    part4_domain_applications()
    part5_optimization()
    part6_best_practices()
    part7_exercises()

    # Summary
    print(f"\n\n{'=' * 60}")
    print("TUTORIAL COMPLETE - KEY TAKEAWAYS")
    print(f"{'=' * 60}")

    takeaways = [
        "🎯 Specificity beats generality - be precise in your instructions",
        "🔄 Iterate and refine your prompts based on results",
        "📝 Structure your prompts: Role + Task + Context + Format",
        "🧪 Test different approaches and measure effectiveness",
        "🤝 Combine techniques (few-shot + chain-of-thought, etc.)",
        "⚖️ Always consider ethical implications and bias",
        "📊 Use appropriate evaluation metrics for your domain",
        "🎭 Role-playing improves response quality and relevance",
    ]

    for takeaway in takeaways:
        print(f"\n{takeaway}")

    print(f"\n\nNext steps:")
    print("1. Practice with the exercises provided")
    print("2. Apply these techniques to your specific use cases")
    print("3. Build a library of effective prompts for your domain")
    print("4. Continuously test and optimize your prompts")


# =============================================================================
# UTILITY FUNCTIONS FOR REAL API TESTING (OPTIONAL)
# =============================================================================


def test_with_real_api(prompt: str, api_key: str = None):
    """
    Optional function to test prompts with real OpenAI API
    Only use if you have an API key and want to see actual responses
    """
    if not api_key:
        print("⚠️ No API key provided. This is just a demonstration framework.")
        return

    try:
        # Note: Update this based on current OpenAI API version
        import openai

        openai.api_key = api_key

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"API Error: {e}")
        return None


def create_prompt_template(
    role: str,
    task: str,
    context: str = "",
    format_instructions: str = "",
    examples: str = "",
):
    """
    Helper function to create well-structured prompts
    """
    template_parts = []

    if role:
        template_parts.append(f"You are {role}.")

    if context:
        template_parts.append(f"Context: {context}")

    if examples:
        template_parts.append(f"Examples:\n{examples}")

    template_parts.append(f"Task: {task}")

    if format_instructions:
        template_parts.append(f"Format: {format_instructions}")

    return "\n\n".join(template_parts)


# Run the tutorial
if __name__ == "__main__":
    run_complete_tutorial()

    # Example of using the template helper
    print(f"\n\n{'=' * 60}")
    print("BONUS: PROMPT TEMPLATE HELPER")
    print(f"{'=' * 60}")

    sample_prompt = create_prompt_template(
        role="a senior marketing manager with e-commerce expertise",
        task="Create a email subject line that will improve open rates for our monthly newsletter",
        context="Our audience is small business owners, current open rate is 18%, industry average is 25%",
        format_instructions="Provide 3 options with brief explanations for each",
        examples="Good: 'boost your sales with these 3 tactics' (specific benefit + number)\nBad: 'Monthly newsletter' (generic, no value proposition)",
    )

    print("Generated prompt:")
    print(f"'{sample_prompt}'")
