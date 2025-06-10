import streamlit as st
import os
from dotenv import load_dotenv
import sys
import openai
import anthropic

# Add project root to path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = None

# Initialize Anthropic client if API key is available
if anthropic_api_key:
    try:
        anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
    except Exception as e:
        print(f"Error initializing Anthropic client: {str(e)}")


class StreamlitPromptEngineering:
    """Frontend version of PromptEngineering class adapted for Streamlit"""

    def __init__(self, model="gpt-3.5-turbo", temperature=0.7, max_tokens=256):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Determine model provider (OpenAI or Anthropic)
        self.provider = "openai" if "gpt" in model else "anthropic"

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
        # Update provider if model changed
        if model and model != self.model:
            self.provider = "openai" if "gpt" in model else "anthropic"
        return params

    def get_completion(self, messages, **kwargs):
        """Get completion with Streamlit progress indicator"""
        params = self.set_parameters(**kwargs)
        provider = "anthropic" if "claude" in params["model"] else "openai"

        # Validate required API keys
        if provider == "openai" and not openai.api_key:
            return f"[OpenAI API key not set. Example response for: {messages[-1]['content'][:50]}...]"
        if provider == "anthropic" and not anthropic_client:
            return f"[Anthropic API key not set. Example response for: {messages[-1]['content'][:50]}...]"

        try:
            with st.spinner("Getting AI response..."):
                if provider == "openai":
                    # Use OpenAI API
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
                else:
                    # Use Anthropic API
                    system_message = ""
                    user_messages = []

                    # Convert ChatML format to Anthropic format
                    for msg in messages:
                        if msg["role"] == "system":
                            system_message = msg["content"]
                        else:
                            user_messages.append(msg["content"])

                    # Join user messages if multiple (simplification for demo)
                    user_content = "\n".join(user_messages) if user_messages else ""

                    response = anthropic_client.messages.create(
                        model=params["model"],
                        max_tokens=params["max_tokens"],
                        temperature=params["temperature"],
                        top_p=params["top_p"],
                        system=system_message,
                        messages=[{"role": "user", "content": user_content}],
                    )
                    return response.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"

    def display_result(self, prompt, response, concept="Basic Prompting"):
        """Display results in Streamlit UI"""
        st.subheader(f"CONCEPT: {concept}")

        col1, col2 = st.columns(2)
        with col1:
            st.text_area("PROMPT", prompt, height=400)
        with col2:
            st.text_area("RESPONSE", response, height=400)
        st.divider()


# Streamlit app layout
def main():
    st.set_page_config(
        page_title="Prompt Engineering Workshop",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Add custom CSS to increase font size and sidebar width
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Space Mono', monospace;
        font-size: 22px;
    }
    h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2.5rem !important;
    }
    h2 {
        font-family: 'Space Mono', monospace;
        font-size: 2rem !important;
    }
    h3 {
        font-family: 'Space Mono', monospace;
        font-size: 1.5rem !important;
    }
    .stTextArea textarea {
        font-family: 'Space Mono', monospace;
        font-size: 20px !important;
        min-height: 400px !important;
    }
    .stButton button {
        font-family: 'Space Mono', monospace;
        font-size: 22px !important;
    }
    .stRadio label {
        font-family: 'Space Mono', monospace;
        font-size: 22px !important;
    }
    .stSelectbox label {
        font-family: 'Space Mono', monospace;
        font-size: 22px !important;
    }
    .sidebar .sidebar-content {
        font-family: 'Space Mono', monospace;
        font-size: 22px !important;
    }
    /* Make sliders wider */
    .stSlider {
        width: 100% !important;
    }
    .stSlider > div > div {
        width: 100% !important;
    }
    /* Make sidebar wider */
    [data-testid="stSidebar"] {
        font-family: 'Space Mono', monospace;
        min-width: 400px !important;
        max-width: 400px !important;
    }
    /* Make sidebar collapsible */
    [data-testid="stSidebar"][aria-expanded="false"] {
        margin-left: -400px !important;
    }
    /* Ensure main content adjusts accordingly */
    .main .block-container {
        max-width: calc(100% - 450px) !important;
        padding-left: 7rem !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create a layout with columns for the title and logo
    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("Prompt Engineering Workshop")
        st.subheader("@AI Convention 2025 (IHK Schwaben)")

    with col2:
        # Display the IHK Schwaben logo flush right
        try:
            # Apply CSS for right alignment
            st.markdown(
                """
                <style>
                [data-testid="column"] > div:has(img) {
                    display: flex;
                    justify-content: flex-end;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.image("assets/DATANOMIQ.png", width=300)
        except Exception as e:
            st.error(f"Could not load logo: {e}")

    # Initialize frontend prompt engineering instance
    pe_demo = StreamlitPromptEngineering()

    # Initialize session state to store settings
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "top_p" not in st.session_state:
        st.session_state.top_p = 1.0
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 256
    if "model" not in st.session_state:
        st.session_state.model = "gpt-3.5-turbo"
    if "provider" not in st.session_state:
        st.session_state.provider = "openai"

    # Sidebar for navigation
    st.sidebar.title("Workshop Sections")
    section = st.sidebar.radio(
        "Choose a section:",
        [
            "Introduction",
            "Settings",  # New settings section
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

    # Add model indicator to the sidebar
    st.sidebar.divider()
    st.sidebar.subheader("Current Model")
    model_name = st.session_state.model
    provider = "OpenAI" if "gpt" in model_name else "Anthropic"
    model_display = model_name.replace("-", " ").title().replace("Gpt", "GPT")
    st.sidebar.markdown(f"**Provider**: {provider}")
    st.sidebar.markdown(f"**Model**: {model_display}")
    st.sidebar.markdown(f"**Temperature**: {st.session_state.temperature}")

    # add Sources
    st.sidebar.divider()
    # add Sources section
    st.sidebar.subheader("Sources")
    st.sidebar.markdown(
        """
        - [Prompt Engineering Paper](https://github.com/thunlp/PromptPapers#papers)
        - [Survey Paper](https://arxiv.org/abs/2402.07927)
        """
    )

    st.sidebar.divider()
    # copyright
    st.sidebar.markdown(
        """
        &copy; 2025 Alexander Lammers (DATANOMIQ GmbH)
        \nAll rights reserved.
        """
    )

    # Display content based on selected section
    if section == "Introduction":
        st.write("""
        ## Introduction
        Welcome to the Prompt Engineering Workshop! This workshop will help you learn effective techniques for working with AI language models.
        
        ### What is Prompt Engineering?
        
        Prompt engineering is the practice of crafting effective inputs (prompts) for AI models to get desired outputs.
        Good prompts can significantly improve the quality, accuracy, and relevance of AI responses.
        
        ### Techniques Overview
        
        **1. Basic Prompting**
        Simple text inputs that rely on the model's pretrained capabilities with minimal structure.
        
        **2. Instruction-based Prompting**
        Explicit directions about what you want the AI to do, providing more control over format and content.
        
        **3. Zero/One/Few-Shot Prompting**
        - Zero-shot: No examples provided
        - One-shot: One example provided before asking for a similar task
        - Few-shot: Multiple examples to establish a pattern
        
        **4. Chain-of-Thought Reasoning**
        Guiding the model to break down complex problems into logical steps, improving performance on multi-step reasoning.
        
        **5. Self-Consistency Techniques**
        Generating multiple independent solutions and finding consensus to increase reliability.
        
        **6. Tree of Thoughts**
        Exploring multiple reasoning paths in parallel to approach complex problems from different angles.
        
        **7. ReAct Framework**
        Combining reasoning with actions to solve problems by interacting with external tools.
        
        **8. Real-world Applications**
        Practical examples combining multiple techniques for effective solutions to complex problems.
        """)

    elif section == "Settings":
        st.write("## Model Configuration Settings")
        st.write("""
        Configure how the AI model generates responses by adjusting these parameters.
        Changes made here will affect all examples throughout the workshop.
        """)

        with st.expander("Understanding Model Parameters", expanded=False):
            st.write("""
            ### Temperature
            
            **Temperature** controls the randomness of the model's output. Higher values (closer to 1.0) make the 
            output more random and creative, while lower values (closer to 0.0) make it more focused and deterministic.
            
            - **Low temperature (0.1-0.3)**: Good for factual responses, classification tasks, or when you need 
              consistent, predictable outputs.
            - **Medium temperature (0.4-0.7)**: Balanced between creativity and coherence, suitable for most general-purpose tasks.
            - **High temperature (0.8-1.0)**: Produces more diverse and creative responses, good for brainstorming, 
              creative writing, or generating multiple alternatives.
            
            ### Top P (Nucleus Sampling)
            
            **Top P** determines how the model selects words when generating text. It filters the output by keeping only 
            the tokens whose cumulative probability exceeds the Top P value.
            
            - Lower values (e.g., 0.5) restrict the model to higher-probability words, making output more conservative
            - Higher values (e.g., 0.95) allow more diversity in word choice
            - Setting Top P = 1.0 means no filtering is applied
            
            ### Maximum Tokens
            
            **Maximum tokens** limits how long the model's response can be. One token is roughly 4 characters or 3/4 of a word.
            
            Setting this properly helps to:
            - Prevent overly lengthy responses
            - Reduce API costs for production applications
            - Control response time
            
            For this workshop, we recommend values between 200-1000 tokens.
            
            ### Model Selection
            
            You can choose from different AI models:
            
            **OpenAI Models**:
            - **GPT-3.5 Turbo**: Balances capability and speed, suitable for most general tasks.
            - **GPT-4**: Higher capability especially for complex reasoning, coding, and creative tasks.
            
            **Anthropic Models**:
            - **Claude 3 Sonnet**: Balanced performance, suitable for most tasks.
            - **Claude 3 Haiku**: Faster and more cost-effective for simpler tasks.
            - **Claude 3 Opus**: Maximum capability for the most complex tasks.
            
            Each model has different strengths, weaknesses, and pricing. Try different models for the same prompt to see how they compare!
            """)

        col1, col2 = st.columns(2)

        with col1:
            # Temperature slider
            temp = st.slider(
                "Temperature:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.05,
                help="Controls randomness: Lower values are more deterministic, higher values more creative",
            )
            st.session_state.temperature = temp

            # Top P slider
            top_p = st.slider(
                "Top P (Nucleus Sampling):",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.top_p,
                step=0.05,
                help="Controls diversity: 1.0 = consider all options, 0.5 = consider only the most likely options",
            )
            st.session_state.top_p = top_p

        with col2:
            # Max tokens slider
            max_tokens = st.slider(
                "Maximum Tokens:",
                min_value=50,
                max_value=1000,
                value=st.session_state.max_tokens,
                step=50,
                help="Maximum length of the response",
            )
            st.session_state.max_tokens = max_tokens

            # Updated model selection with provider groups
            provider = st.selectbox(
                "Provider:",
                ["OpenAI", "Anthropic"],
                index=0 if "gpt" in st.session_state.model else 1,
                help="Select the AI provider",
            )

            if provider == "OpenAI":
                model = st.selectbox(
                    "OpenAI Model:",
                    ["gpt-3.5-turbo", "gpt-4"],
                    index=0 if st.session_state.model == "gpt-3.5-turbo" else 1,
                    help="Select which OpenAI model to use",
                )
            else:
                model = st.selectbox(
                    "Claude Model:",
                    [
                        "claude-3-5-sonnet-latest",
                        "claude-3-haiku-20240307",
                        "claude-3-opus-latest",
                    ],
                    index=0,
                    help="Select which Anthropic Claude model to use",
                )

            st.session_state.model = model
            st.session_state.provider = provider.lower()

            # Add API key settings
            with st.expander("API Keys", expanded=False):
                openai_key = st.text_input(
                    "OpenAI API Key:",
                    type="password",
                    value=os.getenv("OPENAI_API_KEY", ""),
                    help="Your OpenAI API key (stored in session only)",
                )

                if openai_key:
                    openai.api_key = openai_key
                    os.environ["OPENAI_API_KEY"] = openai_key

                anthropic_key = st.text_input(
                    "Anthropic API Key:",
                    type="password",
                    value=os.getenv("ANTHROPIC_API_KEY", ""),
                    help="Your Anthropic API key (stored in session only)",
                )

                if anthropic_key:
                    os.environ["ANTHROPIC_API_KEY"] = anthropic_key
                    # Reinitialize Anthropic client
                    try:
                        global anthropic_client
                        anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                    except Exception as e:
                        st.error(f"Error initializing Anthropic client: {str(e)}")

        st.divider()

        st.subheader("Try Different Settings")
        test_prompt = st.text_area(
            "Test prompt:", "Write a short paragraph about artificial intelligence."
        )

        if st.button("Test Settings"):
            messages = [{"role": "user", "content": test_prompt}]

            # Use the session state values
            response = pe_demo.get_completion(
                messages,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                top_p=st.session_state.top_p,
                model=st.session_state.model,
            )

            st.write("### Response:")
            st.write(response)

            st.info(f"""
            This response was generated using:
            - Provider: {st.session_state.provider.title()}
            - Model: {st.session_state.model}
            - Temperature: {st.session_state.temperature}
            - Top P: {st.session_state.top_p}
            - Max Tokens: {st.session_state.max_tokens}
            """)

        # Settings usage reminder
        st.success("""
        ✅ Your settings have been saved and will be used for all examples in the workshop.
        Feel free to return to this page anytime to adjust the parameters.
        """)

    elif section == "1. Basic Prompting":
        st.write("## Basic Prompting Techniques")
        st.write(
            "Simple text completions that demonstrate how models respond to basic prompts."
        )

        with st.expander("About Basic Prompting", expanded=False):
            st.write("""
            ### Basic Prompting
            
            Basic prompting involves providing a simple text input to the AI and allowing it to complete or respond to that text.
            These prompts have minimal structure and rely on the model's pretrained capabilities.
            
            #### Key characteristics:
            - **Simple structure**: Uses natural language without specialized formatting
            - **Open-ended**: Often allows the model to determine the appropriate response format
            - **Leverages pretrained knowledge**: Relies on the model's existing training rather than explicit instructions
            
            #### Common use cases:
            - Text completion exercises
            - Simple question answering
            - Generating creative content
            - Exploratory interactions to test model capabilities
            
            #### Strengths:
            - Quick and easy to implement
            - Works well for straightforward tasks
            - Requires minimal prompt engineering effort
            
            #### Limitations:
            - Less control over response format and content
            - May produce inconsistent or unpredictable outputs
            - Not ideal for complex or structured tasks
            
            #### Tips for effective basic prompts:
            - Be clear and concise
            - Avoid ambiguous phrasing
            - Use proper spelling and grammar
            - Start with clear context if needed
            """)

        # Initialize chat history in session state if it doesn't exist
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Add a key for managing input field resets
        if "input_key" not in st.session_state:
            st.session_state.input_key = 0

        st.subheader("Try it yourself - Conversational Mode:")

        # Display previous chat messages
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**AI:** {message['content']}")

                # Add a small divider between messages except the last one
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("---")

        # Input for new message - using a unique key that changes when we want to clear the field
        current_input_key = f"user_message_{st.session_state.input_key}"
        user_message = st.text_area("Your message:", key=current_input_key, height=100)

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Send", key="send_message"):
                if user_message:
                    # Add user message to chat history
                    st.session_state.chat_history.append(
                        {"role": "user", "content": user_message}
                    )

                    # Get AI response
                    messages = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.chat_history
                    ]

                    response = pe_demo.get_completion(
                        messages,
                        temperature=st.session_state.temperature,
                        max_tokens=st.session_state.max_tokens,
                        top_p=st.session_state.top_p,
                        model=st.session_state.model,
                    )

                    # Add AI response to chat history
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )

                    # Increment the input key to create a fresh text area (clearing the previous input)
                    st.session_state.input_key += 1

                    # Rerun to update the display with the new messages
                    st.rerun()

        with col2:
            if st.button("Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

    elif section == "2. Instruction-based Prompting":
        st.write("## Instruction-based Prompting")
        st.write("Provide clear instructions to guide the AI's response.")

        with st.expander("About Instruction Prompting", expanded=False):
            st.write("""
            ### Instruction-based Prompting
            
            Instruction-based prompting involves giving the AI explicit directions about what you want it to do.
            This approach provides more control over the format and content of the response.
            
            #### Key characteristics:
            - **Explicit directions**: Clear instructions on task, format, and expected output
            - **Task-oriented**: Focuses on specific actions for the model to perform
            - **Structured approach**: Often includes formatting guidelines or output constraints
            
            #### Common use cases:
            - Data transformation tasks
            - Content classification
            - Structured information extraction
            - Format conversion (e.g., text to JSON)
            - Specific analytical tasks
            
            #### Best practices:
            - **Be specific**: Clearly state what you want the model to do
            - **Define scope**: Set boundaries for the response
            - **Format instructions**: Specify how the answer should be structured
            - **Use action verbs**: Start with "Classify," "Summarize," "List," etc.
            - **Provide examples**: Show the expected output format when needed
            
            #### Advanced techniques:
            - **Multi-step instructions**: Break down complex tasks into sequential steps
            - **Conditional instructions**: "If X applies, do Y; otherwise, do Z"
            - **Reference documents**: "Based on the text below, answer the following questions..."
            
            #### Example structure:
            ```
            Task: [Clear description of what to do]
            Input: [Content to analyze/transform]
            Format: [Instructions about how to structure the output]
            Additional constraints: [Any other requirements]
            ```
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
            response = pe_demo.get_completion(
                messages,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                top_p=st.session_state.top_p,
                model=st.session_state.model,
            )
            pe_demo.display_result(prompt, response, "Instruction-based Prompting")

    elif section == "3. Zero/One/Few-Shot Prompting":
        st.write("## Shot-based Prompting Techniques")
        st.write("Provide examples to guide the model's understanding of the task.")

        shot_type = st.radio("Select shot type:", ["Zero-shot", "One-shot", "Few-shot"])

        with st.expander("About Shot-based Prompting", expanded=False):
            st.write("""
            ### Shot-based Prompting Techniques
            
            Shot-based prompting refers to providing a varying number of examples before asking the model to perform a task.
            
            #### Zero-shot learning
            No examples are provided; the model must perform a task based solely on instructions.
            
            - **Characteristics**: Relies entirely on pretrained knowledge
            - **Best for**: Simple tasks within the model's training domain
            - **Example**: "Classify this movie review as positive or negative: [review]"
            - **Limitations**: Less reliable for complex, ambiguous, or domain-specific tasks
            
            #### One-shot learning
            One example is provided before asking the model to perform a similar task.
            
            - **Characteristics**: Uses a single demonstration to establish the pattern
            - **Best for**: When you need to clarify task format but have limited context space
            - **Example**: 
            ```
            Input: "I loved this movie!" 
            Classification: Positive
            
            Input: "Worst experience ever."
            Classification:
            ```
            - **Benefits**: Significantly improves performance over zero-shot for many tasks
            
            #### Few-shot learning
            Multiple examples are provided to establish a clear pattern.
            
            - **Characteristics**: Uses 2+ examples to demonstrate the desired behavior
            - **Best for**: Complex tasks, unusual formats, or domain-specific knowledge
            - **Example**:
            ```
            Input: "Great product, fast shipping."
            Sentiment: Positive
            Category: Customer Service
            
            Input: "Item arrived damaged and customer service was unhelpful."
            Sentiment: Negative
            Category: Product Quality, Customer Service
            
            Input: "The price was reasonable but delivery took longer than expected."
            Sentiment:
            Category:
            ```
            - **Benefits**: Most reliable approach for consistent, formatted responses
            
            #### Implementation strategies:
            - **Diverse examples**: Include edge cases and various formats
            - **Ordered complexity**: Arrange examples from simple to complex
            - **Balance**: For classification tasks, include examples from all classes
            - **Format consistency**: Maintain the same structure across examples
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
            response = pe_demo.get_completion(
                messages,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                top_p=st.session_state.top_p,
                model=st.session_state.model,
            )
            pe_demo.display_result(prompt, response, f"{shot_type} Prompting")

    elif section == "4. Chain-of-Thought Reasoning":
        st.write("## Chain-of-Thought Reasoning")
        st.write("Guide the model to break down complex problems into logical steps.")

        with st.expander("About Chain-of-Thought Reasoning", expanded=False):
            st.write("""
            ### Chain-of-Thought (CoT) Reasoning
            
            Chain-of-thought (CoT) prompting encourages the model to generate a series of intermediate reasoning steps 
            that lead to a final answer. This technique significantly improves performance on complex tasks that require 
            multi-step reasoning, such as math word problems, logical reasoning, and complex analyses.
            
            #### Key principles:
            - **Externalized reasoning**: Makes the model's thinking process visible
            - **Step-by-step approach**: Breaks complex problems into manageable parts
            - **Reduced error rates**: Helps prevent logical mistakes and oversights
            - **Self-verification**: Allows the model to check its work during the process
            
            #### Implementation methods:
            
            1. **Prompt-based CoT**
            - Add phrases like "Let's think about this step-by-step" or "Let's solve this systematically"
            - Example: "Problem: [problem description]. Let's solve this step-by-step:"
            
            2. **Few-shot CoT**
            - Provide examples of step-by-step reasoning for similar problems
            - Demonstrate the reasoning process you want the model to follow
            
            3. **Generated CoT**
            - First ask the model to generate reasoning steps
            - Then ask it to use those steps to provide a final answer
            
            #### Research findings:
            - CoT improves performance on mathematical problems by 20-40% on standard benchmarks
            - Particularly effective for problems requiring multi-step logical reasoning
            - Works best with more capable models (e.g., GPT-4, Claude, etc.)
            - Can be combined with techniques like Self-Consistency for even better results
            
            #### Best practices:
            - Explicitly request intermediate reasoning steps
            - Break down complex problems into clear stages
            - For math problems, encourage calculation details
            - For logical reasoning, promote consideration of different cases
            - Allow sufficient token space for detailed reasoning
            - Instruct the model to verify its calculations when appropriate
            
            #### Example structure:
            ```
            Problem: [Complex problem]
            Let's solve this step-by-step:
            Step 1: [Understanding the problem]
            Step 2: [Identifying relevant information]
            Step 3: [Applying relevant formulas/concepts]
            Step 4: [Performing calculations/reasoning]
            Step 5: [Verifying the solution]
            Final answer: [Conclusion]
            ```
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
            response = pe_demo.get_completion(
                messages,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                top_p=st.session_state.top_p,
                model=st.session_state.model,
            )
            pe_demo.display_result(prompt, response, "Chain-of-Thought Reasoning")

    elif section == "5. Self-Consistency Techniques":
        st.write("## Self-Consistency Techniques")
        st.write("Generate multiple solution paths and find consensus among them.")

        with st.expander("About Self-Consistency", expanded=False):
            st.write("""
            ### Self-Consistency Techniques
            
            Self-consistency involves generating multiple independent solutions to a problem and then selecting 
            the most consistent answer. This technique increases reliability for complex reasoning tasks by 
            mitigating the effects of randomness in the model's responses.
            
            #### How it works:
            1. **Multiple generations**: Solve the same problem multiple times with different sampling temperatures
            2. **Path diversity**: Each solution may take a different reasoning approach
            3. **Answer extraction**: Extract the final answer from each solution path
            4. **Majority voting**: Take the most frequent answer as the final result
            
            #### Technical implementation:
            - Generate N different responses (typically 5-20) using chain-of-thought prompting
            - Vary temperature/sampling parameters across generations to encourage diversity
            - Parse the final answers programmatically
            - Apply a consensus mechanism (typically majority voting)
            
            #### When to use self-consistency:
            - Complex mathematical or logical problems
            - Tasks where accuracy is critical
            - Problems with potential for different valid solution methods
            - When resources allow for multiple API calls
            - For high-stakes applications requiring confidence in results
            
            #### Research findings:
            - Improves accuracy on GSM8K math problems by 10-15% over standard CoT
            - Reduces variance in model performance
            - Particularly effective when combined with chain-of-thought reasoning
            - More effective with larger, more capable models
            
            #### Limitations:
            - Higher computational cost and API usage
            - May reach incorrect consensus if the model has systematic biases
            - Requires parsing/extraction of answers from free-text responses
            - Not always suitable for open-ended or creative tasks
            
            #### Advanced variations:
            - **Weighted voting**: Consider confidence scores for each solution path
            - **Reasoning verification**: Cross-check intermediate reasoning steps
            - **Iterative refinement**: Use consensus answers as starting points for deeper analysis
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
                response = pe_demo.get_completion(
                    messages,
                    temperature=0.6,  # Use slightly higher temperature for diversity
                    max_tokens=st.session_state.max_tokens,
                    top_p=st.session_state.top_p,
                    model=st.session_state.model,
                )
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

        with st.expander("About Tree of Thoughts", expanded=False):
            st.write("""
            ### Tree of Thoughts (ToT)
            
            The Tree of Thoughts technique extends chain-of-thought prompting by exploring multiple reasoning paths 
            in parallel. It creates a tree structure where each node represents a thought, and branches represent different 
            approaches to solving a problem.
            
            #### Key concepts:
            
            - **Thought as state**: Each reasoning step is a "state" in the search space
            - **Branching**: Multiple possible next steps from each thought
            - **Deliberate exploration**: Systematically consider alternative approaches
            - **Evaluation**: Assess the promise of different reasoning paths
            - **Backtracking**: Ability to abandon unproductive paths and try alternatives
            
            #### Structure of Tree of Thoughts:
            1. **Root**: Problem statement
            2. **First-level branches**: Different initial approaches
            3. **Internal nodes**: Intermediate reasoning steps
            4. **Leaf nodes**: Potential solutions
            
            #### Implementation methods:
            
            1. **Breadth-first search (BFS)**
            - Generate multiple approaches/starting points
            - Explore each approach to a fixed depth
            - Evaluate all paths and select the most promising one(s)
            - Continue exploration from selected path(s)
            
            2. **Depth-first search (DFS)**
            - Explore a single path deeply 
            - Evaluate whether to continue or backtrack
            - If backtracking, try alternative branches
            
            3. **Beam search**
            - Generate k different thoughts at each step
            - Keep only the top-k most promising paths
            - Balance exploration breadth with computational efficiency
            
            #### Ideal applications:
            - Complex planning problems
            - Creative tasks with multiple viable solutions
            - Game playing and strategic decision making
            - Multi-step reasoning where initial intuitions may be misleading
            
            #### Technical considerations:
            - Requires careful prompt engineering to structure the tree exploration
            - Needs evaluation strategies to determine which paths to pursue
            - Can be implemented with self-evaluation (model judges its own thoughts)
            - Often requires breaking token limits through multiple API calls
            
            #### Advanced implementations:
            - **Human-in-the-loop ToT**: Human selects which branches to explore
            - **Automated ToT**: Model generates, evaluates, and selects paths autonomously
            - **Hybrid approaches**: Automated exploration with human guidance at key decision points
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
            Problem: {problem}
            
            Generate {breadth} different initial approaches or perspectives to solve this problem.
            Each approach should be distinct and creative.
            
            Format your response as:
            Approach 1: [brief description]
            Approach 2: [brief description]  
            Approach 3: [brief description]
            """

            messages = [{"role": "user", "content": initial_prompt}]
            initial_response = pe_demo.get_completion(
                messages,
                temperature=0.8,
                max_tokens=st.session_state.max_tokens,
                top_p=st.session_state.top_p,
                model=st.session_state.model,
            )

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
                expanded_response = pe_demo.get_completion(
                    messages,
                    temperature=0.7,
                    max_tokens=st.session_state.max_tokens,
                    top_p=st.session_state.top_p,
                    model=st.session_state.model,
                )

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
                final_solution = pe_demo.get_completion(
                    messages,
                    temperature=0.3,
                    max_tokens=st.session_state.max_tokens,
                    top_p=st.session_state.top_p,
                    model=st.session_state.model,
                )

                st.write(final_solution)

    elif section == "7. ReAct Framework":
        st.write("## ReAct Framework")
        st.write(
            "Combining reasoning and acting to solve problems by interacting with tools."
        )

        with st.expander("About ReAct", expanded=False):
            st.write("""
            ### ReAct Framework (Reasoning + Acting)
            
            ReAct (Reasoning + Acting) is a framework that combines chain-of-thought reasoning with the 
            ability to interact with external tools. This allows models to break down complex problems,
            use tools when needed, and integrate information from those tools into their reasoning process.
            
            #### Core components:
            
            1. **Reasoning**: Internal deliberation about what to do next
            - Similar to chain-of-thought
            - Breaks down complex problems
            - Decides when tool use is necessary
            
            2. **Acting**: Taking actions through tool use
            - Calls to external APIs and tools
            - Information retrieval
            - Calculations and data processing
            - Environment interaction
            
            3. **Observation**: Processing results from actions
            - Interpreting tool outputs
            - Incorporating new information into reasoning
            - Deciding next steps based on observations
            
            #### ReAct pattern:
            ```
            Thought: [Internal reasoning about the problem]
            Action: [Tool name][Tool input]
            Observation: [Result from tool]
            Thought: [Reasoning incorporating the observation]
            ... (repeat as needed)
            Final Answer: [Solution based on the reasoning and acting process]
            ```
            
            #### Common tools integrated with ReAct:
            - **Search engines**: For retrieving factual information
            - **Calculators**: For mathematical operations
            - **Code interpreters**: For executing code and processing data
            - **API calls**: Weather, stocks, sports scores, etc.
            - **Database queries**: For retrieving structured information
            - **Document retrieval**: For accessing relevant documents
            
            #### Benefits:
            - **Factual accuracy**: Reduces hallucination by grounding in external data
            - **Up-to-date information**: Overcomes knowledge cutoff limitations
            - **Computational accuracy**: Delegates calculations to specialized tools
            - **Complex task solving**: Enables multi-step tasks requiring diverse capabilities
            
            #### Implementation considerations:
            - Requires clear formatting conventions for tool calls
            - Needs robust error handling for tool failures
            - Benefits from clear schemas for each tool's capabilities
            - Often implemented with specialized agents frameworks (LangChain, AutoGPT, etc.)
            
            #### Recent developments:
            - Integration with function calling in LLM APIs
            - Tool verification mechanisms to ensure proper tool use
            - Multi-agent systems where different ReAct agents collaborate
            - Specialized agents for particular domains or tool sets
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
                messages,
                temperature=0.4,
                max_tokens=1000,
                top_p=st.session_state.top_p,
                model=st.session_state.model,
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
                category = pe_demo.get_completion(
                    messages,
                    temperature=0.1,
                    max_tokens=st.session_state.max_tokens,
                    top_p=st.session_state.top_p,
                    model=st.session_state.model,
                )

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
                sentiment = pe_demo.get_completion(
                    messages,
                    temperature=0.1,
                    max_tokens=st.session_state.max_tokens,
                    top_p=st.session_state.top_p,
                    model=st.session_state.model,
                )

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
                insights = pe_demo.get_completion(
                    messages,
                    temperature=0.3,
                    max_tokens=st.session_state.max_tokens,
                    top_p=st.session_state.top_p,
                    model=st.session_state.model,
                )

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
                classification = pe_demo.get_completion(
                    messages,
                    temperature=0.2,
                    max_tokens=st.session_state.max_tokens,
                    top_p=st.session_state.top_p,
                    model=st.session_state.model,
                )

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
                solution = pe_demo.get_completion(
                    messages,
                    temperature=0.4,
                    max_tokens=st.session_state.max_tokens,
                    top_p=st.session_state.top_p,
                    model=st.session_state.model,
                )

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
                basic_result = pe_demo.get_completion(
                    messages,
                    temperature=0.3,
                    max_tokens=st.session_state.max_tokens,
                    top_p=st.session_state.top_p,
                    model=st.session_state.model,
                )

                with st.expander("Basic Prompting Result", expanded=False):
                    st.write(basic_result)

                # Instruction-based
                st.write("#### Instruction-based Prompting")

                instruction_prompt = f"Task: Analyze and solve: {comparison_problem}\nFormat: Provide a structured solution with steps and reasoning."
                messages = [{"role": "user", "content": instruction_prompt}]
                instruction_result = pe_demo.get_completion(
                    messages,
                    temperature=0.3,
                    max_tokens=st.session_state.max_tokens,
                    top_p=st.session_state.top_p,
                    model=st.session_state.model,
                )

                with st.expander("Instruction-based Result", expanded=False):
                    st.write(instruction_result)

                # Chain-of-Thought
                st.write("#### Chain-of-Thought")

                cot_prompt = (
                    f"Problem: {comparison_problem}\n\nLet's solve this step-by-step:"
                )
                messages = [{"role": "user", "content": cot_prompt}]
                cot_result = pe_demo.get_completion(
                    messages,
                    temperature=0.3,
                    max_tokens=st.session_state.max_tokens,
                    top_p=st.session_state.top_p,
                    model=st.session_state.model,
                )

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
                comparison_result = pe_demo.get_completion(
                    messages,
                    temperature=0.3,
                    max_tokens=st.session_state.max_tokens,
                    top_p=st.session_state.top_p,
                    model=st.session_state.model,
                )

                st.write(comparison_result)


if __name__ == "__main__":
    main()
