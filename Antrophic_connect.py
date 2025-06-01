import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Install anthropic if needed: pip install anthropic
from anthropic import Anthropic

# Get API keys
# openai_api_key = os.environ["OPENAI_API_KEY"]

# Get Anthropic API key and verify it exists
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY is not set in the environment variables")

# Initialize Anthropic client with explicit api_key parameter
client = Anthropic(api_key=anthropic_api_key)

# Make a simple test request to Claude
response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": "Hello Claude, could you explain what you are in one short paragraph?",
        }
    ],
)

# Print the response content
print("Response from Claude:")
print(response.content[0].text)

# Ways to interact with the Anthropic API:
# 1. Messages API (current generation)
#    - client.messages.create() - Create a message with the model
#    - Parameters: model, max_tokens, messages, system (optional), temperature, top_p, top_k, etc.
#    - Supports multi-turn conversations with message history
#    - Supports images via content blocks
#
# 2. Streaming responses
#    - client.messages.create(stream=True) - Get tokens as they're generated
#    - Returns an iterator you can process in real-time
#
# 3. Tools/Function calling
#    - Define tools parameter with JSON schema for structured outputs
#    - Model can return JSON data in a specified format
#
# 4. System prompts
#    - Using system parameter to set context/instructions for the model
#
# 5. Completions API (legacy)
#    - client.completions.create() - Older style API
#    - Parameters: model, prompt, max_tokens_to_sample, etc.
#
# 6. Content moderation
#    - Happens automatically on inputs and outputs
#    - Can be configured with specific parameters
#
# 7. File/image handling
#    - Include images in messages via content blocks with image URLs or base64 data
