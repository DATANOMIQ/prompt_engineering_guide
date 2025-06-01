def demonstrate_basic_prompting(pe_demo):
    """
    Demonstrate the most basic form of prompting - text completion
    """
    print("WORKSHOP SECTION 1: BASIC PROMPTING")
    print("=" * 50)
    
    # Example 1: Simple text completion
    basic_prompt = "The sky is"
    messages = [{"role": "user", "content": basic_prompt}]
    
    response = pe_demo.get_completion(messages, temperature=0.1)
    pe_demo.display_result(basic_prompt, response, "Basic Text Completion")
    
    # Example 2: Story beginning
    story_prompt = "Once upon a time, in a distant galaxy"
    messages = [{"role": "user", "content": story_prompt}]
    
    response = pe_demo.get_completion(messages, temperature=0.8, max_tokens=150)
    pe_demo.display_result(story_prompt, response, "Creative Text Generation")
    
    # Example 3: Technical topic starter
    tech_prompt = "Machine learning algorithms can be categorized into"
    messages = [{"role": "user", "content": tech_prompt}]
    
    response = pe_demo.get_completion(messages, temperature=0.3)
    pe_demo.display_result(tech_prompt, response, "Technical Text Completion")
    
    # Interactive demonstration
    print("\n🔬 INTERACTIVE DEMO: Try your own basic prompts!")
    print("Enter a prompt fragment (or 'skip' to continue):")
    
    user_input = input("> ")
    if user_input.lower() != 'skip':
        messages = [{"role": "user", "content": user_input}]
        response = pe_demo.get_completion(messages)
        pe_demo.display_result(user_input, response, "Your Basic Prompt")