class PromptEngineeringWorkshop:
    """
    Comprehensive workshop framework combining all techniques
    """

    def __init__(self, pe_instance):
        self.pe = pe_instance
        self.techniques = {
            "basic": self.basic_prompting,
            "instruction": self.instruction_prompting,
            "few_shot": self.few_shot_prompting,
            "cot": self.chain_of_thought,
            "self_consistency": self.self_consistency,
            "tree_of_thoughts": self.tree_of_thoughts,
            "react": self.react_framework,
        }
        self.results_log = []

    def basic_prompting(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        return self.pe.get_completion(messages, temperature=0.3)

    def instruction_prompting(self, task, format_instructions=None):
        prompt = f"Task: {task}\n"
        if format_instructions:
            prompt += f"Format: {format_instructions}\n"
        messages = [{"role": "user", "content": prompt}]
        return self.pe.get_completion(messages, temperature=0.3)

    def few_shot_prompting(self, examples, new_input):
        prompt = "Examples:\n"
        for ex_input, ex_output in examples:
            prompt += f"Input: {ex_input}\nOutput: {ex_output}\n\n"
        prompt += f"Input: {new_input}\nOutput:"
        messages = [{"role": "user", "content": prompt}]
        return self.pe.get_completion(messages, temperature=0.3)

    def chain_of_thought(self, problem):
        prompt = f"Problem: {problem}\n\nLet's solve this step-by-step:"
        messages = [{"role": "user", "content": prompt}]
        return self.pe.get_completion(messages, temperature=0.3)

    def self_consistency(self, problem, num_samples=3):
        # Simplified implementation
        results = []
        prompt = f"Problem: {problem}\n\nLet's solve this step-by-step:"
        for _ in range(num_samples):
            messages = [{"role": "user", "content": prompt}]
            results.append(self.pe.get_completion(messages, temperature=0.7))
        return results

    def tree_of_thoughts(self, problem):
        # Simplified implementation
        prompt = f"""
        Problem: {problem}
        
        Generate three different approaches to solve this problem:
        
        Approach 1:
        Approach 2:
        Approach 3:
        
        Now evaluate each approach and select the best one:
        """
        messages = [{"role": "user", "content": prompt}]
        return self.pe.get_completion(messages, temperature=0.4)

    def react_framework(self, question, tools_description="No tools available"):
        prompt = f"""
        Question: {question}
        
        Available tools: {tools_description}
        
        Think step by step:
        """
        messages = [{"role": "user", "content": prompt}]
        return self.pe.get_completion(messages, temperature=0.4)

    def technique_comparison(self, problem: str):
        """
        Compare how different techniques handle the same problem
        """
        print(f"\n🔬 TECHNIQUE COMPARISON")
        print(f"Problem: {problem}")
        print("=" * 60)

        results = {}

        # Basic prompting
        basic_prompt = f"Problem: {problem}\nSolution:"
        basic_result = self.basic_prompting(basic_prompt)
        results["basic"] = basic_result
        print(f"\n📝 Basic Prompting:\n{basic_result[:150]}...")

        # Instruction-based
        instruction_result = self.instruction_prompting(
            f"Analyze and solve: {problem}",
            "Provide a structured solution with steps and reasoning.",
        )
        results["instruction"] = instruction_result
        print(f"\n📋 Instruction-based:\n{instruction_result[:150]}...")

        # Chain-of-Thought
        cot_result = self.chain_of_thought(problem)
        results["cot"] = cot_result
        print(f"\n🧠 Chain-of-Thought:\n{cot_result[:150]}...")

        return results


class RealWorldApplication:
    """
    Demonstrate real-world applications combining multiple techniques
    """

    def __init__(self, pe_instance):
        self.pe = pe_instance

    def content_analysis_pipeline(self, content: str):
        """
        Multi-step content analysis using various prompting techniques
        """
        print(f"\n📊 CONTENT ANALYSIS PIPELINE")
        print("=" * 50)

        # Step 1: Basic categorization (zero-shot)
        category_prompt = f"""
        Categorize this content into one of: News, Opinion, Educational, Entertainment, Commercial
        
        Content: {content[:200]}...
        
        Category:"""

        messages = [{"role": "user", "content": category_prompt}]
        category = self.pe.get_completion(messages, temperature=0.1)
        print(f"📂 Category: {category.strip()}")

        # Step 2: Sentiment analysis (few-shot)
        sentiment_prompt = f"""
        Examples:
        "This product is amazing!" → Positive
        "Terrible customer service" → Negative
        "The weather is okay today" → Neutral
        
        Analyze sentiment: "{content[:100]}..."
        Sentiment:"""

        messages = [{"role": "user", "content": sentiment_prompt}]
        sentiment = self.pe.get_completion(messages, temperature=0.1)
        print(f"😊 Sentiment: {sentiment.strip()}")

        # Step 3: Key insights (Chain-of-Thought)
        insights_prompt = f"""
        Content: {content}
        
        Let's analyze this content step-by-step to extract key insights:
        
        1. Main topics discussed:
        2. Target audience:
        3. Key messages:
        4. Overall quality assessment:
        """

        messages = [{"role": "user", "content": insights_prompt}]
        insights = self.pe.get_completion(messages, temperature=0.3)
        print(f"🔍 Key Insights:\n{insights}")

        return {
            "category": category.strip(),
            "sentiment": sentiment.strip(),
            "insights": insights,
        }

    def problem_solving_assistant(self, problem_description: str):
        """
        Multi-technique problem solving assistant
        """
        print(f"\n🔧 PROBLEM SOLVING ASSISTANT")
        print("=" * 50)

        # Step 1: Problem classification and approach selection
        classification_prompt = f"""
        Problem: {problem_description}
        
        Classify this problem type and recommend the best approach:
        
        Problem Type: (Technical, Business, Personal, Academic, Creative)
        Complexity Level: (Simple, Medium, Complex)
        Recommended Approach: (Direct solution, Step-by-step analysis, Creative brainstorming, Research required)
        Time Sensitivity: (Urgent, Normal, Long-term)
        
        Classification:"""

        messages = [{"role": "user", "content": classification_prompt}]
        classification = self.pe.get_completion(messages, temperature=0.2)
        print(f"📋 Problem Classification:\n{classification}")

        # Step 2: Solution generation using appropriate technique
        if "Complex" in classification:
            # Use Tree of Thoughts approach for complex problems
            solution_prompt = f"""
            Complex Problem: {problem_description}
            
            Let's explore multiple solution approaches:
            
            Approach 1: [Conservative/Safe solution]
            Approach 2: [Innovative/Creative solution]  
            Approach 3: [Resource-efficient solution]
            
            For each approach, consider:
            - Feasibility
            - Resources required
            - Timeline
            - Risk factors
            - Expected outcomes
            """
        else:
            # Use Chain-of-Thought for simpler problems
            solution_prompt = f"""
            Problem: {problem_description}
            
            Let's solve this step-by-step:
            
            Step 1: Understand the problem
            Step 2: Identify constraints and requirements
            Step 3: Generate potential solutions
            Step 4: Evaluate and select best solution
            Step 5: Create implementation plan
            """

        messages = [{"role": "user", "content": solution_prompt}]
        solution = self.pe.get_completion(messages, temperature=0.4)
        print(f"💡 Solution Analysis:\n{solution}")

        return {
            "classification": classification,
            "solution": solution,
        }


def final_workshop_demonstration(pe_demo, workshop, real_world):
    """
    Final comprehensive demonstration bringing all techniques together
    """
    print("\n🎓 FINAL WORKSHOP DEMONSTRATION")
    print("=" * 60)

    # Demonstrate technique comparison
    comparison_problem = (
        "How can a small business improve customer retention in a competitive market?"
    )
    comparison_results = workshop.technique_comparison(comparison_problem)

    # Real-world application examples
    sample_content = """
    The latest smartphone release promises revolutionary battery life with their new 
    graphene-enhanced cells lasting up to 3 days on a single charge. Early reviews 
    suggest impressive performance, though the $1200 price point may limit adoption. 
    Industry experts predict this technology will become standard within 2-3 years.
    """

    content_analysis = real_world.content_analysis_pipeline(sample_content)

    # Problem-solving demonstration
    business_problem = """
    Our software development team is consistently missing sprint deadlines. 
    Team morale is low, client satisfaction is decreasing, and we're losing 
    competitive advantage.
    """

    problem_solution = real_world.problem_solving_assistant(business_problem)

    print(f"\n🎯 WORKSHOP SUMMARY")
    print("=" * 40)
    print("✅ Techniques covered:")
    print("  • Basic Prompting")
    print("  • Instruction-based Prompting")
    print("  • Zero/One/Few-shot Learning")
    print("  • Chain-of-Thought Reasoning")
    print("  • Self-Consistency")
    print("  • Tree of Thoughts")
    print("  • ReAct Framework")
    print("  • Real-world Applications")
