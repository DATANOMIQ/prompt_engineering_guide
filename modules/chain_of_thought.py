def demonstrate_chain_of_thought(pe_demo):
    """
    Demonstrate Chain-of-Thought prompting for complex reasoning
    """
    print("\nWORKSHOP SECTION 4: CHAIN-OF-THOUGHT REASONING")
    print("=" * 60)

    # Basic CoT example with math problem
    print("\n🧠 BASIC CHAIN-OF-THOUGHT")
    basic_cot_prompt = """
    Q: A store has 23 apples. They sell 8 apples in the morning and 5 apples in the afternoon. 
    Then they receive a delivery of 12 more apples. How many apples do they have now?
    
    Let's work through this step-by-step:
    A:"""

    messages = [{"role": "user", "content": basic_cot_prompt}]
    response = pe_demo.get_completion(messages, temperature=0.1)
    pe_demo.display_result(basic_cot_prompt, response, "Basic Chain-of-Thought")

    # Zero-shot CoT with "Let's think step by step"
    print("\n🧠 ZERO-SHOT CHAIN-OF-THOUGHT")
    zero_shot_cot_prompt = """
    Q: If a train travels 120 miles in 2 hours, and then travels 180 miles in 3 hours, 
    what is the average speed for the entire journey?
    
    Let's think step by step.
    A:"""

    messages = [{"role": "user", "content": zero_shot_cot_prompt}]
    response = pe_demo.get_completion(messages, temperature=0.1)
    pe_demo.display_result(zero_shot_cot_prompt, response, "Zero-Shot Chain-of-Thought")


def demonstrate_few_shot_cot(pe_demo):
    """
    Demonstrate few-shot Chain-of-Thought with examples
    """
    print("\n🧠 FEW-SHOT CHAIN-OF-THOUGHT")

    few_shot_cot_prompt = """
    Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
    A: Roger started with 5 tennis balls. He bought 2 cans of tennis balls. Each can has 3 tennis balls, so 2 cans have 2 × 3 = 6 tennis balls. In total, he has 5 + 6 = 11 tennis balls.
    
    Q: A cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
    A: The cafeteria started with 23 apples. They used 20 apples to make lunch, so they had 23 - 20 = 3 apples left. Then they bought 6 more apples, so they have 3 + 6 = 9 apples now.
    
    Q: Sarah has $50. She buys 3 books that cost $12 each. Then she finds $8 on the ground. How much money does she have now?
    A:"""

    messages = [{"role": "user", "content": few_shot_cot_prompt}]
    response = pe_demo.get_completion(messages, temperature=0.1)
    pe_demo.display_result(few_shot_cot_prompt, response, "Few-Shot Chain-of-Thought")

    # Complex problem-solving with CoT
    complex_problem_prompt = """
    Scenario: A company is planning to launch a new product. They need to decide between two marketing strategies:
    
    Strategy A: 
    - Initial cost: $100,000
    - Expected monthly revenue: $25,000
    - Monthly operational costs: $8,000
    
    Strategy B:
    - Initial cost: $150,000  
    - Expected monthly revenue: $35,000
    - Monthly operational costs: $12,000
    
    Question: Which strategy will be more profitable after 18 months, and by how much?
    
    Let's analyze this step-by-step:
    """

    messages = [{"role": "user", "content": complex_problem_prompt}]
    response = pe_demo.get_completion(messages, temperature=0.1)
    pe_demo.display_result(
        complex_problem_prompt, response, "Complex Business Reasoning"
    )
