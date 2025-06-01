from collections import Counter


class SelfConsistencyFramework:
    """
    Framework for implementing self-consistency with various problems
    """

    def __init__(self, pe_instance, num_samples=5):
        self.pe = pe_instance
        self.num_samples = num_samples

    async def generate_multiple_solutions(self, problem, temperature=0.7):
        """Generate multiple solutions to a problem"""
        solutions = []

        for i in range(self.num_samples):
            messages = [{"role": "user", "content": problem}]
            response = self.pe.get_completion(messages, temperature=temperature)
            solutions.append(response)

        return solutions

    def extract_numerical_answer(self, text):
        """Extract numerical answers from text"""
        import re

        # Look for various numerical patterns
        patterns = [
            r"\$[\d,]+\.?\d*",  # Dollar amounts
            r"\b\d+\.?\d*\b",  # Simple numbers
            r"\b\d+/\d+\b",  # Fractions
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1]  # Return the last match
        return None

    def find_consensus(self, solutions, extract_func=None):
        """Find consensus among multiple solutions"""
        if extract_func is None:
            extract_func = self.extract_numerical_answer

        answers = []
        for solution in solutions:
            answer = extract_func(solution)
            if answer:
                answers.append(answer)

        answer_counts = Counter(answers)

        if answer_counts:
            consensus = answer_counts.most_common(1)[0]
            confidence = consensus[1] / len(answers) if answers else 0
            return {
                "answer": consensus[0],
                "confidence": confidence,
                "vote_count": consensus[1],
                "total_responses": len(answers),
                "all_answers": answer_counts,
            }

        return None


def demonstrate_advanced_self_consistency(pe_demo):
    """
    Demonstrate self-consistency with different types of problems
    """
    print("\n🧠 SELF-CONSISTENCY DEMONSTRATIONS")
    print("=" * 60)

    framework = SelfConsistencyFramework(pe_demo, num_samples=4)

    # Math word problem
    math_problem = """
    Q: A school library has 3 floors. The first floor has 1,200 books, 
    the second floor has 40% more books than the first floor, and the 
    third floor has 25% fewer books than the second floor. How many 
    books does the library have in total?
    
    Let's calculate step-by-step:
    A:"""

    print("Problem: Library book calculation")
    print("-" * 40)

    # Generate solutions synchronously for demonstration
    solutions = []
    for i in range(4):
        messages = [{"role": "user", "content": math_problem}]
        response = pe_demo.get_completion(messages, temperature=0.6)
        solutions.append(response)
        print(f"\nSolution {i + 1}:")
        print(response[:200] + "..." if len(response) > 200 else response)

    # Find consensus
    consensus = framework.find_consensus(solutions)
    if consensus:
        print(f"\n✅ CONSENSUS RESULT:")
        print(f"Answer: {consensus['answer']}")
        print(f"Confidence: {consensus['confidence']:.2%}")
        print(f"Votes: {consensus['vote_count']}/{consensus['total_responses']}")

    # Logical reasoning problem
    logic_problem = """
    Q: In a class of 30 students:
    - 18 students like mathematics
    - 15 students like science  
    - 8 students like both mathematics and science
    - How many students like neither mathematics nor science?
    
    Let's solve this using set theory principles:
    A:"""

    print(f"\n{'=' * 60}")
    print("Problem: Set theory logical reasoning")
    print("-" * 40)

    logic_solutions = []
    for i in range(4):
        messages = [{"role": "user", "content": logic_problem}]
        response = pe_demo.get_completion(messages, temperature=0.5)
        logic_solutions.append(response)
        print(f"\nLogical reasoning path {i + 1}:")
        print(response[:150] + "..." if len(response) > 150 else response)

    logic_consensus = framework.find_consensus(logic_solutions)
    if logic_consensus:
        print(f"\n✅ LOGICAL CONSENSUS:")
        print(f"Answer: {logic_consensus['answer']}")
        print(f"Agreement level: {logic_consensus['confidence']:.2%}")
