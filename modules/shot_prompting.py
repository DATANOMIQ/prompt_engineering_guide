def demonstrate_shot_based_prompting(pe_demo):
    """
    Demonstrate zero-shot, one-shot, and few-shot prompting techniques
    """
    print("\nWORKSHOP SECTION 3: SHOT-BASED PROMPTING TECHNIQUES")
    print("=" * 70)

    # Zero-shot prompting example
    print("\n🎯 ZERO-SHOT PROMPTING")
    zero_shot_prompt = """
    Classify the sentiment of the following text as positive, negative, or neutral.
    
    Text: "The weather today is quite unpredictable, but I'm managing to get things done."
    
    Sentiment:"""

    messages = [{"role": "user", "content": zero_shot_prompt}]
    response = pe_demo.get_completion(messages, temperature=0.1)
    pe_demo.display_result(zero_shot_prompt, response, "Zero-Shot Classification")

    # One-shot prompting example
    print("\n🎯 ONE-SHOT PROMPTING")
    one_shot_prompt = """
    Classify the sentiment of the following text as positive, negative, or neutral.
    
    Example:
    Text: "I love spending time in nature, it's so peaceful."
    Sentiment: Positive
    
    Now classify this:
    Text: "The weather today is quite unpredictable, but I'm managing to get things done."
    Sentiment:"""

    messages = [{"role": "user", "content": one_shot_prompt}]
    response = pe_demo.get_completion(messages, temperature=0.1)
    pe_demo.display_result(one_shot_prompt, response, "One-Shot Classification")

    # Few-shot prompting example
    print("\n🎯 FEW-SHOT PROMPTING")
    few_shot_prompt = """
    Classify the sentiment of the following text as positive, negative, or neutral.
    
    Examples:
    Text: "I love spending time in nature, it's so peaceful."
    Sentiment: Positive
    
    Text: "This traffic jam is really frustrating and making me late."
    Sentiment: Negative
    
    Text: "The meeting was okay, nothing particularly exciting happened."
    Sentiment: Neutral
    
    Text: "I can't believe how rude the customer service was today."
    Sentiment: Negative
    
    Now classify this:
    Text: "The weather today is quite unpredictable, but I'm managing to get things done."
    Sentiment:"""

    messages = [{"role": "user", "content": few_shot_prompt}]
    response = pe_demo.get_completion(messages, temperature=0.1)
    pe_demo.display_result(few_shot_prompt, response, "Few-Shot Classification")


def demonstrate_few_shot_learning_tasks(pe_demo):
    """
    Demonstrate few-shot learning across different task types
    """
    print("\n🔬 ADVANCED FEW-SHOT DEMONSTRATIONS")
    print("=" * 50)

    # Few-shot for entity extraction
    entity_extraction_prompt = """
    Extract person names, organizations, and locations from the text.
    
    Examples:
    Text: "John Smith works at Microsoft in Seattle."
    Entities: Person: John Smith, Organization: Microsoft, Location: Seattle
    
    Text: "Apple Inc. was founded by Steve Jobs in Cupertino."
    Entities: Person: Steve Jobs, Organization: Apple Inc., Location: Cupertino
    
    Text: "The conference at Stanford University featured Dr. Sarah Johnson from IBM."
    Entities:"""

    messages = [{"role": "user", "content": entity_extraction_prompt}]
    response = pe_demo.get_completion(messages, temperature=0.1)
    pe_demo.display_result(
        entity_extraction_prompt, response, "Few-Shot Entity Extraction"
    )

    # Few-shot for code pattern recognition
    code_pattern_prompt = """
    Convert the description into a Python function signature.
    
    Examples:
    Description: "A function that calculates the area of a circle given its radius"
    Function: def calculate_circle_area(radius: float) -> float:
    
    Description: "A function that finds the maximum value in a list of integers"
    Function: def find_maximum(numbers: List[int]) -> int:
    
    Description: "A function that checks if a string is a valid email address"
    Function:"""

    messages = [{"role": "user", "content": code_pattern_prompt}]
    response = pe_demo.get_completion(messages, temperature=0.1)
    pe_demo.display_result(code_pattern_prompt, response, "Few-Shot Code Generation")
