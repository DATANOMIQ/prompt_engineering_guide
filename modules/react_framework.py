from abc import ABC, abstractmethod
import requests
import json
from typing import List, Tuple


class Tool(ABC):
    """Abstract base class for ReAct tools"""

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def execute(self, query: str) -> str:
        pass


class WikipediaTool(Tool):
    """Tool for searching Wikipedia"""

    def name(self) -> str:
        return "wikipedia"

    def description(self) -> str:
        return "Search Wikipedia for information about a topic. Input should be a search query."

    def execute(self, query: str) -> str:
        try:
            # Simplified Wikipedia API call
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(
                " ", "_"
            )
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                return (
                    f"Wikipedia Summary: {data.get('extract', 'No summary available')}"
                )
            else:
                return f"No Wikipedia page found for '{query}'"
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"


class CalculatorTool(Tool):
    """Tool for mathematical calculations"""

    def name(self) -> str:
        return "calculator"

    def description(self) -> str:
        return "Perform mathematical calculations. Input should be a mathematical expression."

    def execute(self, query: str) -> str:
        try:
            # Safe evaluation of mathematical expressions
            import ast
            import operator

            # Supported operations
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }

            def eval_expr(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](
                        eval_expr(node.left), eval_expr(node.right)
                    )
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](eval_expr(node.operand))
                else:
                    raise TypeError(node)

            # Parse and evaluate
            tree = ast.parse(query, mode="eval")
            result = eval_expr(tree.body)
            return f"Calculation result: {result}"

        except Exception as e:
            return f"Error in calculation: {str(e)}"


class WeatherTool(Tool):
    """Mock weather tool for demonstration"""

    def name(self) -> str:
        return "weather"

    def description(self) -> str:
        return "Get current weather information for a location. Input should be a city name."

    def execute(self, query: str) -> str:
        # Mock weather data for demonstration
        mock_weather = {
            "new york": "Temperature: 22°C, Condition: Partly cloudy, Humidity: 65%",
            "london": "Temperature: 15°C, Condition: Rainy, Humidity: 80%",
            "tokyo": "Temperature: 28°C, Condition: Sunny, Humidity: 55%",
            "paris": "Temperature: 18°C, Condition: Overcast, Humidity: 70%",
        }

        location = query.lower().strip()
        if location in mock_weather:
            return f"Weather in {query}: {mock_weather[location]}"
        else:
            return f"Weather data not available for {query} (using mock data)"


class ReActAgent:
    """
    ReAct Agent implementation that combines reasoning and acting
    """

    def __init__(self, pe_instance, tools: List[Tool], max_iterations=5):
        self.pe = pe_instance
        self.tools = {tool.name(): tool for tool in tools}
        self.max_iterations = max_iterations
        self.conversation_history = []

    def create_tool_description(self) -> str:
        """Create a description of available tools"""
        tool_descriptions = []
        for tool_name, tool in self.tools.items():
            tool_descriptions.append(f"- {tool_name}: {tool.description()}")

        return "\n".join(tool_descriptions)

    def parse_action(self, response: str) -> Tuple[str, str]:
        """Parse action from LLM response"""
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("Action:"):
                action_part = line[7:].strip()
                if "[" in action_part and "]" in action_part:
                    tool_name = action_part.split("[")[0].strip()
                    query = action_part.split("[")[1].split("]")[0].strip()
                    return tool_name, query

        return None, None

    def solve_with_react(self, question: str) -> str:
        """
        Solve a question using the ReAct framework
        """
        print(f"\n🤖 ReAct AGENT: {question}")
        print("=" * 60)

        tools_desc = self.create_tool_description()

        system_prompt = f"""
You are a helpful assistant that can use tools to answer questions.

Available tools:
{tools_desc}

Use the following format:
Thought: (your reasoning about what to do next)
Action: tool_name[query]
Observation: (result of the action)

Continue this process until you can provide a final answer.
When you have enough information, end with:
Final Answer: (your complete answer to the question)

Question: {question}
"""

        conversation = [{"role": "system", "content": system_prompt}]
        full_response = ""

        for iteration in range(self.max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}")
            print("-" * 30)

            # Get LLM response
            response = self.pe.get_completion(conversation, temperature=0.1)
            full_response += response + "\n"

            print(f"LLM Response:\n{response}")

            # Check if we have a final answer
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                print(f"\n✅ FINAL ANSWER: {final_answer}")
                return final_answer

            # Parse and execute action
            tool_name, query = self.parse_action(response)

            if tool_name and tool_name in self.tools:
                print(f"\n🔧 Executing: {tool_name}[{query}]")
                observation = self.tools[tool_name].execute(query)
                print(f"📊 Observation: {observation}")

                # Add observation to conversation
                observation_text = f"\nObservation: {observation}\n"
                full_response += observation_text

                # Update conversation with the full context
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_response},
                ]
            else:
                print(f"❌ Invalid or unknown action: {tool_name}")
                break

        return "Could not solve the question within the maximum iterations."


def demonstrate_react_framework(pe_demo):
    """
    Demonstrate the ReAct framework with various questions
    """
    print("\nWORKSHOP SECTION 7: ReAct (REASON + ACT) FRAMEWORK")
    print("=" * 70)

    # Initialize tools
    tools = [WikipediaTool(), CalculatorTool(), WeatherTool()]

    # Create ReAct agent
    react_agent = ReActAgent(
        pe_demo, tools, max_iterations=3
    )  # Reduced for demo purposes

    # Example: Mathematical reasoning with external calculation
    math_question = """
    If a circle has a radius of 15 meters, what is its area? 
    """

    result = react_agent.solve_with_react(math_question)
    print(f"\nResult: {result}")
