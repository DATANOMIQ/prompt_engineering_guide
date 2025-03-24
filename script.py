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
