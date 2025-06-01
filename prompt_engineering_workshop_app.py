import streamlit as st
import os
from dotenv import load_dotenv
import sys
import openai

# Add project root to path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# No unused imports

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class StreamlitPromptEngineering:
    """Frontend version of PromptEngineering class adapted for Streamlit"""

    def __init__(self, model="gpt-3.5-turbo", temperature=0.7, max_tokens=256):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def set_parameters(
        self,
        model=None,
        temperature=None,
        max_tokens=None,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        # Same as original
        params = {
            "model": model or self.model,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        return params

    def get_completion(self, messages, **kwargs):
        """Get completion with Streamlit progress indicator"""
        params = self.set_parameters(**kwargs)

        if not openai.api_key:
            return f"[Example response for: {messages[-1]['content'][:50]}...]"

        try:
            with st.spinner("Getting AI response..."):
                response = openai.chat.completions.create(
                    model=params["model"],
                    messages=messages,
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    top_p=params["top_p"],
                    frequency_penalty=params["frequency_penalty"],
                    presence_penalty=params["presence_penalty"],
                )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def display_result(self, prompt, response, concept="Basic Prompting"):
        """Display results in Streamlit UI"""
        st.subheader(f"CONCEPT: {concept}")

        col1, col2 = st.columns(2)
        with col1:
            st.text_area("PROMPT", prompt, height=200)
        with col2:
            st.text_area("RESPONSE", response, height=200)
        st.divider()


# Streamlit app layout
def main():
    st.set_page_config(
        page_title="Prompt Engineering Workshop",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🚀 Prompt Engineering Workshop")
    st.subheader("Interactive Guide to Mastering Prompt Engineering")

    # Initialize frontend prompt engineering instance
    pe_demo = StreamlitPromptEngineering()

    # Sidebar for navigation
    st.sidebar.title("Workshop Sections")
    section = st.sidebar.radio(
        "Choose a section:",
        [
            "Introduction",
            "1. Basic Prompting",
            "2. Instruction-based Prompting",
            "3. Zero/One/Few-Shot Prompting",
            "4. Chain-of-Thought Reasoning",
            "5. Self-Consistency Techniques",
            "6. Tree of Thoughts",
            "7. ReAct Framework",
            "8. Real-world Applications",
        ],
    )

    # API Key input in sidebar
    with st.sidebar.expander("OpenAI API Configuration"):
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            openai.api_key = api_key

    # Display content based on selected section
    if section == "Introduction":
        st.write("""
        ## Welcome to the Prompt Engineering Workshop! 
        
        This interactive workshop will guide you through various techniques for effectively communicating with AI models.
        
        ### What is Prompt Engineering?
        
        Prompt engineering is the process of designing and optimizing inputs (prompts) to AI language models to get the most useful, relevant, and accurate outputs.
        
        ### How to use this workshop:
        
        1. Select a section from the sidebar to explore different prompt engineering techniques
        2. Experiment with the interactive examples
        3. Try modifying the prompts to see how small changes affect the output
        
        Let's get started!
        """)

    elif section == "1. Basic Prompting":
        st.write("## Basic Prompting Techniques")
        st.write(
            "Simple text completions that demonstrate how models respond to basic prompts."
        )

        with st.expander("About Basic Prompting", expanded=True):
            st.write("""
            Basic prompting involves providing a simple text input to the AI and allowing it to complete or respond to that text.
            These prompts have minimal structure and rely on the model's pretrained capabilities.
            """)

        st.subheader("Try it yourself:")
        user_prompt = st.text_area("Enter your prompt:", "The sky is")
        if st.button("Generate Response", key="basic"):
            messages = [{"role": "user", "content": user_prompt}]
            response = pe_demo.get_completion(messages, temperature=0.7)
            pe_demo.display_result(user_prompt, response, "Basic Prompting")

    elif section == "2. Instruction-based Prompting":
        st.write("## Instruction-based Prompting")
        st.write("Provide clear instructions to guide the AI's response.")

        with st.expander("About Instruction Prompting", expanded=True):
            st.write("""
            Instruction-based prompting involves giving the AI explicit directions about what you want it to do.
            This approach provides more control over the format and content of the response.
            """)

        st.subheader("Try it yourself:")
        task = st.text_area(
            "Enter task:",
            "Classify the following text as positive, negative, or neutral.",
        )
        text = st.text_area(
            "Enter text to analyze:",
            "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
        )
        format_instructions = st.text_area(
            "Format instructions (optional):",
            "Output only the classification without explanation.",
        )

        if st.button("Generate Response", key="instruction"):
            prompt = f"Task: {task}\nText: {text}\n"
            if format_instructions:
                prompt += f"Format: {format_instructions}\n"

            messages = [{"role": "user", "content": prompt}]
            response = pe_demo.get_completion(messages, temperature=0.1)
            pe_demo.display_result(prompt, response, "Instruction-based Prompting")

    elif section == "3. Zero/One/Few-Shot Prompting":
        st.write("## Shot-based Prompting Techniques")
        st.write("Provide examples to guide the model's understanding of the task.")

        shot_type = st.radio("Select shot type:", ["Zero-shot", "One-shot", "Few-shot"])

        with st.expander("About Shot-based Prompting", expanded=True):
            st.write("""
            - **Zero-shot**: No examples provided, asking the model to perform a task directly
            - **One-shot**: One example provided before asking the model to perform a similar task
            - **Few-shot**: Multiple examples provided to establish a pattern
            """)

        st.subheader("Sentiment Classification Example")

        if shot_type == "Zero-shot":
            prompt = st.text_area(
                "Zero-shot prompt:",
                'Classify the sentiment of the following text as positive, negative, or neutral.\n\nText: "The weather today is quite unpredictable, but I\'m managing to get things done."\n\nSentiment:',
            )

        elif shot_type == "One-shot":
            prompt = st.text_area(
                "One-shot prompt:",
                'Classify the sentiment of the following text as positive, negative, or neutral.\n\nExample:\nText: "I love spending time in nature, it\'s so peaceful."\nSentiment: Positive\n\nNow classify this:\nText: "The weather today is quite unpredictable, but I\'m managing to get things done."\nSentiment:',
            )

        else:  # Few-shot
            prompt = st.text_area(
                "Few-shot prompt:",
                'Classify the sentiment of the following text as positive, negative, or neutral.\n\nExamples:\nText: "I love spending time in nature, it\'s so peaceful."\nSentiment: Positive\n\nText: "This traffic jam is really frustrating and making me late."\nSentiment: Negative\n\nText: "The meeting was okay, nothing particularly exciting happened."\nSentiment: Neutral\n\nNow classify this:\nText: "The weather today is quite unpredictable, but I\'m managing to get things done."\nSentiment:',
            )

        if st.button("Generate Response", key="shot"):
            messages = [{"role": "user", "content": prompt}]
            response = pe_demo.get_completion(messages, temperature=0.1)
            pe_demo.display_result(prompt, response, f"{shot_type} Prompting")

    elif section == "4. Chain-of-Thought Reasoning":
        st.write("## Chain-of-Thought Reasoning")
        st.write("Guide the model to break down complex problems into logical steps.")

        with st.expander("About Chain-of-Thought Reasoning", expanded=True):
            st.write("""
            Chain-of-thought (CoT) prompting encourages the model to generate a series of intermediate reasoning steps 
            that lead to a final answer. This technique significantly improves performance on complex tasks that require 
            multi-step reasoning, such as math word problems, logical reasoning, and complex analyses.
            """)

        st.subheader("Try it yourself:")

        cot_type = st.radio(
            "Select reasoning type:",
            ["Basic", "Complex Business Scenario", "Custom Problem"],
        )

        if cot_type == "Basic":
            problem = st.text_area(
                "Problem to solve:",
                "A store has 23 apples. They sell 8 apples in the morning and 5 apples in the afternoon. "
                "Then they receive a delivery of 12 more apples. How many apples do they have now?",
            )
        elif cot_type == "Complex Business Scenario":
            problem = st.text_area(
                "Business scenario:",
                "A company is planning to launch a new product. They need to decide between two marketing strategies:\n\n"
                "Strategy A:\n- Initial cost: $100,000\n- Expected monthly revenue: $25,000\n"
                "- Monthly operational costs: $8,000\n\n"
                "Strategy B:\n- Initial cost: $150,000\n- Expected monthly revenue: $35,000\n"
                "- Monthly operational costs: $12,000\n\n"
                "Which strategy will be more profitable after 18 months, and by how much?",
            )
        else:  # Custom Problem
            problem = st.text_area(
                "Enter your own problem:",
                "Write your problem here. Make sure it requires multi-step reasoning.",
            )

        if st.button("Generate Step-by-Step Solution", key="cot"):
            prompt = f"Problem: {problem}\n\nLet's solve this step-by-step:"
            messages = [{"role": "user", "content": prompt}]
            response = pe_demo.get_completion(messages, temperature=0.3)
            pe_demo.display_result(prompt, response, "Chain-of-Thought Reasoning")

    elif section == "5. Self-Consistency Techniques":
        st.write("## Self-Consistency Techniques")
        st.write("Generate multiple solution paths and find consensus among them.")

        with st.expander("About Self-Consistency", expanded=True):
            st.write("""
            Self-consistency involves generating multiple independent solutions to a problem and then selecting 
            the most consistent answer. This technique increases reliability for complex reasoning tasks by 
            mitigating the effects of randomness in the model's responses.
            
            The technique works by:
            1. Solving the same problem multiple times with different sampling temperatures
            2. Extracting the final answer from each solution
            3. Taking the most frequent answer as the final result
            """)

        st.subheader("Try Self-Consistency:")

        problem_type = st.selectbox(
            "Select problem type:",
            ["Math Word Problem", "Logical Reasoning", "Custom Problem"],
        )

        if problem_type == "Math Word Problem":
            problem = st.text_area(
                "Math problem:",
                "A school library has 3 floors. The first floor has 1,200 books, "
                "the second floor has 40% more books than the first floor, and the "
                "third floor has 25% fewer books than the second floor. How many "
                "books does the library have in total?",
            )
        elif problem_type == "Logical Reasoning":
            problem = st.text_area(
                "Logical reasoning problem:",
                "In a class of 30 students:\n"
                "- 18 students like mathematics\n"
                "- 15 students like science\n"
                "- 8 students like both mathematics and science\n"
                "- How many students like neither mathematics nor science?",
            )
        else:
            problem = st.text_area(
                "Enter your own problem:",
                "Write a problem that would benefit from multiple solution attempts.",
            )

        num_samples = st.slider("Number of solutions to generate:", 2, 5, 3)

        if st.button("Generate Multiple Solutions", key="self_consistency"):
            st.write("### Generated Solutions:")
            all_solutions = []

            progress_bar = st.progress(0)
            for i in range(num_samples):
                prompt = f"Problem: {problem}\n\nLet's solve this step-by-step:"
                messages = [{"role": "user", "content": prompt}]
                response = pe_demo.get_completion(messages, temperature=0.6)
                all_solutions.append(response)

                with st.expander(f"Solution {i + 1}"):
                    st.write(response)

                progress_bar.progress((i + 1) / num_samples)

            st.success("All solutions generated!")

            st.write("### Consensus Analysis:")
            st.info(
                "In a real self-consistency implementation, an algorithm would extract the final answers from each solution and determine the consensus. Since this is a demonstration, please review the solutions manually to identify common answers."
            )

    elif section == "6. Tree of Thoughts":
        st.write("## Tree of Thoughts")
        st.write(
            "Explore multiple solution paths systematically by creating a tree of possibilities."
        )

        with st.expander("About Tree of Thoughts", expanded=True):
            st.write("""
            The Tree of Thoughts technique extends chain-of-thought prompting by exploring multiple reasoning paths 
            in parallel. It creates a tree structure where each node represents a thought, and branches represent different 
            approaches to solving a problem.
            
            This method is particularly useful for:
            - Complex, creative problems with multiple viable solutions
            - Problems requiring exploration of different perspectives
            - Situations where the first approach might not yield optimal results
            """)

        st.subheader("Try Tree of Thoughts:")

        problem_options = {
            "Business Strategy": "A small local bookstore is losing customers to online retailers and e-books. The owner wants to innovate and find new ways to attract customers while maintaining the community feel of the bookstore. What strategies should they implement?",
            "Product Design": "Design a smart home device that helps elderly people maintain independence while ensuring their safety and well-being.",
            "Education Challenge": "How might we redesign the classroom experience to better engage students who have different learning styles?",
            "Custom Problem": "Enter your own problem that would benefit from exploring multiple solution paths.",
        }

        selected_problem = st.selectbox(
            "Select problem type:", list(problem_options.keys())
        )

        if selected_problem == "Custom Problem":
            problem = st.text_area("Enter your problem:", "")
        else:
            problem = st.text_area("Problem:", problem_options[selected_problem])

        max_depth = st.slider("Maximum exploration depth:", 1, 3, 2)
        breadth = st.slider("Number of branches at each point:", 2, 4, 3)

        if st.button("Generate Tree of Thoughts", key="tot"):
            st.write("### Generating Initial Approaches...")

            # Step 1: Generate initial approaches
            initial_prompt = f"""
        # Use max_depth in the logic or remove if not needed
        breadth = st.slider("Number of branches at each point:", 2, 4, 3)
            Generate {breadth} different initial approaches or perspectives to solve this problem.
            Each approach should be distinct and creative.
            
            Format your response as:
            Approach 1: [brief description]
            Approach 2: [brief description]  
            Approach 3: [brief description]
            """

            messages = [{"role": "user", "content": initial_prompt}]
            initial_response = pe_demo.get_completion(messages, temperature=0.8)

            # Display initial approaches
            st.write("### Initial Approaches:")
            st.write(initial_response)

            # Extract approaches (simplified parsing)
            approaches = []
            for line in initial_response.split("\n"):
                if line.strip().startswith("Approach"):
                    approach = line.split(":", 1)[1].strip() if ":" in line else line
                    approaches.append(approach)

            # Step 2: Expand best approach
            if approaches:
                st.write("### Developing Best Approach:")

                # For demonstration, let's expand the first approach
                best_approach = approaches[0]

                expand_prompt = f"""
                Original Problem: {problem}
                Current Approach: {best_approach}
                
                Based on this approach, generate {breadth} specific next steps or refinements.
                Each should build upon the current approach and move closer to a solution.
                
                Format as:
                Step 1: [specific action or refinement]
                Step 2: [specific action or refinement]
                Step 3: [specific action or refinement]
                """

                messages = [{"role": "user", "content": expand_prompt}]
                expanded_response = pe_demo.get_completion(messages, temperature=0.7)

                st.write("#### Next Steps:")
                st.write(expanded_response)

                # Step 3: Generate final solution
                st.write("### Final Solution:")

                final_prompt = f"""
                Problem: {problem}
                Best approach identified: {best_approach}
                
                Based on this approach, provide a detailed, concrete solution to the original problem.
                Include specific steps, considerations, and expected outcomes.
                """

                messages = [{"role": "user", "content": final_prompt}]
                final_solution = pe_demo.get_completion(messages, temperature=0.3)

                st.write(final_solution)

    elif section == "7. ReAct Framework":
        st.write("## ReAct Framework")
        st.write(
            "Combining reasoning and acting to solve problems by interacting with tools."
        )

        with st.expander("About ReAct", expanded=True):
            st.write("""
            ReAct (Reasoning + Acting) is a framework that combines chain-of-thought reasoning with the 
            ability to interact with external tools. This allows models to break down complex problems,
            use tools when needed, and integrate information from those tools into their reasoning process.
            
            This approach is especially useful for tasks that require:
            - Gathering information not available in the model's knowledge
            - Performing calculations or specialized operations
            - Multi-step plans where each step may depend on external data
            """)

        st.subheader("Try ReAct Framework:")

        available_tools = st.multiselect(
            "Select available tools:",
            ["Wikipedia Search", "Calculator", "Weather API"],
            default=["Calculator"],
        )

        tool_description = ""
        if "Wikipedia Search" in available_tools:
            tool_description += "wikipedia: Search Wikipedia for information about a topic. Input should be a search query.\n"
        if "Calculator" in available_tools:
            tool_description += "calculator: Perform mathematical calculations. Input should be a mathematical expression.\n"
        if "Weather API" in available_tools:
            tool_description += "weather: Get current weather for a location. Input should be a city name.\n"

        question_options = {
            "Mathematical": "If a circle has a radius of 15 meters, what is its area?",
            "Research": "When was the first computer invented and by whom?",
            "Weather": "What's the current weather in Tokyo and is it a good day for tourism?",
            "Custom": "Enter your own question that might require using tools.",
        }

        selected_question = st.selectbox(
            "Select question type:", list(question_options.keys())
        )

        if selected_question == "Custom":
            question = st.text_area("Enter your question:", "")
        else:
            question = st.text_area("Question:", question_options[selected_question])

        if st.button("Solve with ReAct", key="react"):
            st.write("### ReAct Process:")

            react_prompt = f"""
            You are a helpful assistant that can use tools to answer questions.

            Available tools:
            {tool_description}

            Use the following format:
            Thought: (your reasoning about what to do next)
            Action: tool_name[query]
            Observation: (result of the action)

            Continue this process until you can provide a final answer.
            When you have enough information, end with:
            Final Answer: (your complete answer to the question)

            Question: {question}
            """

            messages = [{"role": "system", "content": react_prompt}]

            # This is a simplified version - in a real implementation, you would
            # actually parse the model's responses and execute real tool calls
            response = pe_demo.get_completion(
                messages, temperature=0.4, max_tokens=1000
            )

            # Format the response with highlighting
            formatted_response = (
                response.replace("Thought:", "**Thought:**")
                .replace("Action:", "**Action:**")
                .replace("Observation:", "**Observation:**")
                .replace("Final Answer:", "**Final Answer:**")
            )

            st.write(formatted_response)

            st.info(
                "Note: This is a simulation of the ReAct framework. In a real implementation, the system would actually execute the tool actions and feed real observations back to the model."
            )

    elif section == "8. Real-world Applications":
        st.write("## Real-world Applications")
        st.write("Practical applications that combine multiple prompting techniques.")

        application_type = st.selectbox(
            "Select application type:",
            ["Content Analysis", "Problem Solving Assistant", "Technique Comparison"],
        )

        if application_type == "Content Analysis":
            st.write("### Content Analysis Pipeline")
            st.write("Analyze content through multiple steps using various techniques.")

            content = st.text_area(
                "Enter content to analyze:",
                "The latest smartphone release promises revolutionary battery life with their new "
                "graphene-enhanced cells lasting up to 3 days on a single charge. Early reviews "
                "suggest impressive performance, though the $1200 price point may limit adoption. "
                "Industry experts predict this technology will become standard within 2-3 years.",
            )

            if st.button("Run Content Analysis", key="content_analysis"):
                st.write("#### Step 1: Basic Categorization")

                category_prompt = f"""
                Categorize this content into one of: News, Opinion, Educational, Entertainment, Commercial
                
                Content: {content[:200]}...
                
                Category:"""

                messages = [{"role": "user", "content": category_prompt}]
                category = pe_demo.get_completion(messages, temperature=0.1)

                st.write(f"**Category:** {category.strip()}")

                st.write("#### Step 2: Sentiment Analysis")

                sentiment_prompt = f"""
                Examples:
                "This product is amazing!" → Positive
                "Terrible customer service" → Negative
                "The weather is okay today" → Neutral
                
                Analyze sentiment: "{content[:100]}..."
                Sentiment:"""

                messages = [{"role": "user", "content": sentiment_prompt}]
                sentiment = pe_demo.get_completion(messages, temperature=0.1)

                st.write(f"**Sentiment:** {sentiment.strip()}")

                st.write("#### Step 3: Key Insights")

                insights_prompt = f"""
                Content: {content}
                
                Let's analyze this content step-by-step to extract key insights:
                
                1. Main topics discussed:
                2. Target audience:
                3. Key messages:
                4. Overall quality assessment:
                """

                messages = [{"role": "user", "content": insights_prompt}]
                insights = pe_demo.get_completion(messages, temperature=0.3)

                st.write(insights)

        elif application_type == "Problem Solving Assistant":
            st.write("### Problem Solving Assistant")
            st.write(
                "Combined approach using multiple techniques to solve complex problems."
            )

            problem_description = st.text_area(
                "Enter problem description:",
                "Our software development team is consistently missing sprint deadlines. "
                "Team morale is low, client satisfaction is decreasing, and we're losing "
                "competitive advantage.",
            )

            if st.button("Solve Problem", key="problem_solving"):
                st.write("#### Step 1: Problem Classification")

                classification_prompt = f"""
                Problem: {problem_description}
                
                Classify this problem type and recommend the best approach:
                
                Problem Type: (Technical, Business, Personal, Academic, Creative)
                Complexity Level: (Simple, Medium, Complex)
                Recommended Approach: (Direct solution, Step-by-step analysis, Creative brainstorming, Research required)
                Time Sensitivity: (Urgent, Normal, Long-term)
                
                Classification:"""

                messages = [{"role": "user", "content": classification_prompt}]
                classification = pe_demo.get_completion(messages, temperature=0.2)

                st.write("**Problem Classification:**")
                st.write(classification)

                # Determine which approach to use based on the classification
                is_complex = "Complex" in classification

                st.write("#### Step 2: Solution Generation")

                if is_complex:
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
                solution = pe_demo.get_completion(messages, temperature=0.4)

                st.write("**Solution Analysis:**")
                st.write(solution)

        else:  # Technique Comparison
            st.write("### Technique Comparison")
            st.write(
                "Compare how different prompting techniques handle the same problem."
            )

            comparison_problem = st.text_area(
                "Problem to compare techniques:",
                "How can a small business improve customer retention in a competitive market?",
            )

            if st.button("Compare Techniques", key="technique_comparison"):
                # Basic prompting
                st.write("#### Basic Prompting")

                basic_prompt = f"Problem: {comparison_problem}\nSolution:"
                messages = [{"role": "user", "content": basic_prompt}]
                basic_result = pe_demo.get_completion(messages, temperature=0.3)

                with st.expander("Basic Prompting Result", expanded=False):
                    st.write(basic_result)

                # Instruction-based
                st.write("#### Instruction-based Prompting")

                instruction_prompt = f"Task: Analyze and solve: {comparison_problem}\nFormat: Provide a structured solution with steps and reasoning."
                messages = [{"role": "user", "content": instruction_prompt}]
                instruction_result = pe_demo.get_completion(messages, temperature=0.3)

                with st.expander("Instruction-based Result", expanded=False):
                    st.write(instruction_result)

                # Chain-of-Thought
                st.write("#### Chain-of-Thought")

                cot_prompt = (
                    f"Problem: {comparison_problem}\n\nLet's solve this step-by-step:"
                )
                messages = [{"role": "user", "content": cot_prompt}]
                cot_result = pe_demo.get_completion(messages, temperature=0.3)

                with st.expander("Chain-of-Thought Result", expanded=False):
                    st.write(cot_result)

                # Comparison summary
                st.write("#### Technique Comparison Summary")

                comparison_prompt = f"""
                Compare the following three solutions to the problem: "{comparison_problem}"
                
                Solution 1 (Basic):
                {basic_result[:300]}...
                
                Solution 2 (Instruction-based):
                {instruction_result[:300]}...
                
                Solution 3 (Chain-of-Thought):
                {cot_result[:300]}...
                
                Provide a brief analysis of the strengths and weaknesses of each approach:
                """

                messages = [{"role": "user", "content": comparison_prompt}]
                comparison_result = pe_demo.get_completion(messages, temperature=0.3)

                st.write(comparison_result)

                # Call the main function to run the app


if __name__ == "__main__":
    main()
