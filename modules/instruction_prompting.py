def demonstrate_instruction_prompting(pe_demo):
    """
    Demonstrate instruction-based prompting with various task types
    """
    print("\nWORKSHOP SECTION 2: INSTRUCTION-BASED PROMPTING")
    print("=" * 60)

    # Example 1: Text Classification with Clear Instructions
    classification_prompt = """
    Classify the following text into one of these categories: positive, negative, or neutral.
    
    Text: "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
    
    Classification:"""

    messages = [{"role": "user", "content": classification_prompt}]
    response = pe_demo.get_completion(messages, temperature=0.1)
    pe_demo.display_result(classification_prompt, response, "Text Classification")

    # Example 2: Structured Output with Output Indicators
    structured_prompt = """
    Analyze the following customer review and provide a structured analysis.
    
    Review: "The product arrived late and the packaging was damaged. However, the product quality itself is excellent and works as expected."
    
    Analysis:
    Sentiment: 
    Key Issues:
    Positive Aspects:
    Overall Rating (1-5):"""

    messages = [{"role": "user", "content": structured_prompt}]
    response = pe_demo.get_completion(messages, temperature=0.2)
    pe_demo.display_result(structured_prompt, response, "Structured Analysis")

    # Example 3: Code Generation with Specific Requirements
    code_prompt = """
    Write a Python function that:
    1. Takes a list of numbers as input
    2. Returns the median value
    3. Handles edge cases (empty list, single element)
    4. Includes proper documentation
    
    Function:"""

    messages = [{"role": "user", "content": code_prompt}]
    response = pe_demo.get_completion(messages, temperature=0.1)
    pe_demo.display_result(code_prompt, response, "Code Generation")

    # Example 4: Multi-step Instructions
    complex_prompt = """
    Follow these steps to create a brief business proposal:
    
    Step 1: Identify the problem (waste management in urban areas)
    Step 2: Propose a solution using AI technology
    Step 3: Outline the target market
    Step 4: Estimate initial investment needed
    Step 5: Project potential ROI in 2 years
    
    Business Proposal:"""

    messages = [{"role": "user", "content": complex_prompt}]
    response = pe_demo.get_completion(messages, temperature=0.4, max_tokens=400)
    pe_demo.display_result(complex_prompt, response, "Multi-step Instructions")


def demonstrate_good_prompt_structure(pe_demo):
    """
    Demonstrate the components of well-structured prompts
    """
    print("\n🎯 DEMONSTRATION: Components of Effective Prompts")
    print("=" * 60)

    # Example of a well-structured prompt with all components
    comprehensive_prompt = """
    ROLE: You are an expert data scientist with 10 years of experience in machine learning.
    
    TASK: Explain the concept of overfitting in machine learning models.
    
    CONTEXT: This explanation is for undergraduate computer science students who have basic programming knowledge but are new to machine learning concepts.
    
    FORMAT: Provide your explanation in the following structure:
    1. Definition (2-3 sentences)
    2. Why it happens (key causes)
    3. How to detect it (practical indicators)
    4. Prevention strategies (3-4 methods)
    
    AUDIENCE: Undergraduate students
    
    TONE: Educational, clear, and encouraging
    
    ADDITIONAL REQUIREMENTS: Use simple examples and avoid overly technical jargon.
    """

    messages = [{"role": "user", "content": comprehensive_prompt}]
    response = pe_demo.get_completion(messages, temperature=0.3, max_tokens=500)
    pe_demo.display_result(comprehensive_prompt, response, "Well-Structured Prompt")

    # Compare with a poorly structured prompt
    poor_prompt = "Explain overfitting."
    messages = [{"role": "user", "content": poor_prompt}]
    poor_response = pe_demo.get_completion(messages, temperature=0.3)
    pe_demo.display_result(poor_prompt, poor_response, "Poorly Structured Prompt")
