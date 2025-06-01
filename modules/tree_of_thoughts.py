from dataclasses import dataclass
from typing import List, Dict, Optional
import re


@dataclass
class Thought:
    """Represents a single thought in the tree"""

    content: str
    score: float = 0.0
    depth: int = 0
    parent_id: Optional[str] = None
    thought_id: str = ""

    def __post_init__(self):
        if not self.thought_id:
            import uuid

            self.thought_id = str(uuid.uuid4())[:8]


class TreeOfThoughts:
    """
    Implementation of Tree of Thoughts for systematic problem exploration
    """

    def __init__(self, pe_instance, max_depth=3, breadth=3):
        self.pe = pe_instance
        self.max_depth = max_depth
        self.breadth = breadth
        self.thoughts = {}
        self.current_problem = ""

    def generate_initial_thoughts(self, problem: str) -> List[Thought]:
        """Generate initial thoughts for the given problem"""
        self.current_problem = problem

        prompt = f"""
        Problem: {problem}
        
        Generate {self.breadth} different initial approaches or perspectives to solve this problem.
        Each approach should be distinct and creative.
        
        Format your response as:
        Approach 1: [brief description]
        Approach 2: [brief description]  
        Approach 3: [brief description]
        """

        messages = [{"role": "user", "content": prompt}]
        response = self.pe.get_completion(messages, temperature=0.8)

        # Parse the response to extract individual thoughts
        thoughts = []
        lines = response.strip().split("\n")
        for i, line in enumerate(lines):
            if line.strip() and ("Approach" in line or f"{i + 1}:" in line):
                content = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                thought = Thought(content=content, depth=0)
                thoughts.append(thought)
                self.thoughts[thought.thought_id] = thought

        return thoughts[: self.breadth]

    def expand_thought(self, thought: Thought) -> List[Thought]:
        """Expand a thought into multiple sub-thoughts"""
        if thought.depth >= self.max_depth:
            return []

        prompt = f"""
        Original Problem: {self.current_problem}
        Current Approach: {thought.content}
        
        Based on this approach, generate {self.breadth} specific next steps or refinements.
        Each should build upon the current approach and move closer to a solution.
        
        Format as:
        Step 1: [specific action or refinement]
        Step 2: [specific action or refinement]
        Step 3: [specific action or refinement]
        """

        messages = [{"role": "user", "content": prompt}]
        response = self.pe.get_completion(messages, temperature=0.7)

        expanded_thoughts = []
        lines = response.strip().split("\n")
        for i, line in enumerate(lines):
            if line.strip() and ("Step" in line or f"{i + 1}:" in line):
                content = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                new_thought = Thought(
                    content=content,
                    depth=thought.depth + 1,
                    parent_id=thought.thought_id,
                )
                expanded_thoughts.append(new_thought)
                self.thoughts[new_thought.thought_id] = new_thought

        return expanded_thoughts[: self.breadth]

    def evaluate_thought(self, thought: Thought) -> float:
        """Evaluate the quality/promise of a thought"""
        evaluation_prompt = f"""
        Problem: {self.current_problem}
        Proposed approach/step: {thought.content}
        
        Evaluate this approach on a scale of 1-10 considering:
        - Feasibility (how realistic is this approach?)
        - Effectiveness (how likely is it to solve the problem?)
        - Creativity (how innovative or unique is this approach?)
        
        Provide just a numerical score (1-10):
        """

        messages = [{"role": "user", "content": evaluation_prompt}]
        response = self.pe.get_completion(messages, temperature=0.1)

        # Extract numerical score
        score_match = re.search(r"\b([1-9]|10)\b", response)
        score = float(score_match.group(1)) if score_match else 5.0

        thought.score = score
        return score

    def select_best_thoughts(
        self, thoughts: List[Thought], k: int = 2
    ) -> List[Thought]:
        """Select the k best thoughts based on their scores"""
        # Evaluate all thoughts if not already scored
        for thought in thoughts:
            if thought.score == 0.0:
                self.evaluate_thought(thought)

        # Sort by score and return top k
        sorted_thoughts = sorted(thoughts, key=lambda t: t.score, reverse=True)
        return sorted_thoughts[:k]

    def solve_with_tree_of_thoughts(self, problem: str) -> Dict:
        """
        Main method to solve a problem using Tree of Thoughts
        """
        print(f"\n🌳 TREE OF THOUGHTS: {problem}")
        print("=" * 60)

        # Step 1: Generate initial thoughts
        print("🌱 Generating initial approaches...")
        initial_thoughts = self.generate_initial_thoughts(problem)

        for i, thought in enumerate(initial_thoughts, 1):
            print(f"   Approach {i}: {thought.content}")

        # Step 2: Iteratively expand and prune
        current_level = initial_thoughts

        for depth in range(self.max_depth):
            print(f"\n🔄 Depth {depth + 1}: Expanding and evaluating...")

            next_level = []
            for thought in current_level:
                expanded = self.expand_thought(thought)
                next_level.extend(expanded)

                # Show expanded thoughts
                if expanded:
                    print(f"   From '{thought.content[:50]}...':")
                    for exp_thought in expanded:
                        score = self.evaluate_thought(exp_thought)
                        print(f"     → {exp_thought.content} (Score: {score})")

            if not next_level:
                break

            # Select best thoughts for next iteration
            current_level = self.select_best_thoughts(
                next_level, k=min(3, len(next_level))
            )

            print(f"\n✅ Selected top thoughts for depth {depth + 2}:")
            for thought in current_level:
                print(f"   • {thought.content} (Score: {thought.score})")

        # Step 3: Generate final solution
        best_thought = (
            max(current_level, key=lambda t: t.score)
            if current_level
            else initial_thoughts[0]
        )

        final_solution_prompt = f"""
        Problem: {problem}
        Best approach identified: {best_thought.content}
        
        Based on this approach, provide a detailed, concrete solution to the original problem.
        Include specific steps, considerations, and expected outcomes.
        """

        messages = [{"role": "user", "content": final_solution_prompt}]
        final_solution = self.pe.get_completion(messages, temperature=0.3)

        return {
            "problem": problem,
            "best_approach": best_thought.content,
            "best_score": best_thought.score,
            "final_solution": final_solution,
            "total_thoughts_explored": len(self.thoughts),
            "search_depth": best_thought.depth,
        }

    def visualize_tree(self):
        """Create a simple text visualization of the thought tree"""
        print("\n🌳 THOUGHT TREE VISUALIZATION")
        print("=" * 40)

        # Group thoughts by depth
        by_depth = {}
        for thought in self.thoughts.values():
            if thought.depth not in by_depth:
                by_depth[thought.depth] = []
            by_depth[thought.depth].append(thought)

        for depth in sorted(by_depth.keys()):
            print(f"\nDepth {depth}:")
            for thought in by_depth[depth]:
                indent = "  " * depth
                score_str = f"(Score: {thought.score:.1f})" if thought.score > 0 else ""
                print(f"{indent}• {thought.content[:60]}... {score_str}")


def demonstrate_tree_of_thoughts(pe_demo):
    """
    Demonstrate Tree of Thoughts with practical problems
    """
    print("\nWORKSHOP SECTION 6: TREE OF THOUGHTS")
    print("=" * 50)

    tot = TreeOfThoughts(pe_demo, max_depth=2, breadth=3)

    # Problem 1: Business strategy
    business_problem = """
    A small local bookstore is losing customers to online retailers and e-books. 
    The owner wants to innovate and find new ways to attract customers while 
    maintaining the community feel of the bookstore. What strategies should they implement?
    """

    result1 = tot.solve_with_tree_of_thoughts(business_problem)

    print(f"\n🎯 FINAL SOLUTION:")
    print(f"Best approach: {result1['best_approach']}")
    print(f"Confidence score: {result1['best_score']}/10")
    print(f"Detailed solution:\n{result1['final_solution']}")

    tot.visualize_tree()
