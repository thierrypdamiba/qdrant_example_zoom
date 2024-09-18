import sys
import os
import re
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import anthropic
import json
import warnings
import os

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="huggingface/tokenizers")

# Set the environment variable to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_api_key_from_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config.js')
    
    with open(config_path, 'r') as file:
        content = file.read()
        match = re.search(r'anthropicApiKey:\s*[\'"](.+?)[\'"]', content)
        if match:
            return match.group(1)
        else:
            raise ValueError("Anthropic API key not found in config.js")

# Get the API key
try:
    anthropic_api_key = get_api_key_from_config()
    print(f"API Key (first 10 characters): {anthropic_api_key[:10]}...")
except Exception as e:
    print(f"Error reading API key: {e}")
    raise

# Initialize Qdrant client
client = QdrantClient("localhost", port=6333)

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define collection name
collection_name = 'user_recordings'

def query_vector_db(meeting_purpose):
    purpose_vector = model.encode(meeting_purpose).tolist()
    
    # Perform search
    search_result = client.search(
        collection_name=collection_name,
        query_vector=purpose_vector,
        limit=10  # Increased limit for more context
    )

    # Format the search results
    formatted_results = []
    for hit in search_result:
        result = {
            "Score": hit.score,
            "Topic": hit.payload.get('topic', 'N/A'),
            "Participants": hit.payload.get('participants', []),
            "Summary": hit.payload.get('summary', {}).get('summary_overview', 'N/A')
        }
        formatted_results.append(result)
    
    return formatted_results

def get_anthropic_response(meeting_purpose, search_results):
    if not anthropic_api_key:
        raise ValueError("Anthropic API key not set.")

    client = anthropic.Anthropic(api_key=anthropic_api_key)

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are an AI assistant tasked with suggesting meetings and participants based on a given purpose.",
        messages=[
            {
                "role": "user",
                "content": f"Based on the following meeting records, please suggest a meeting structure and potential participants for the following purpose: '{meeting_purpose}'\n\nRelevant Meeting Records:\n{json.dumps(search_results, indent=2)}\n\nPlease provide:\n1. A suggested meeting title\n2. A brief meeting agenda\n3. A list of recommended participants and why they should be invited\n4. Any additional suggestions for making the meeting effective"
            }
        ]
    )
    
    return message.content

if __name__ == "__main__":
    if len(sys.argv) > 1:
        meeting_purpose = " ".join(sys.argv[1:])
        search_results = query_vector_db(meeting_purpose)
        suggestions = get_anthropic_response(meeting_purpose, search_results)
        print(f"Meeting Suggestions:\n{suggestions}")
    else:
        print("No meeting purpose provided. Please specify what you want to do.")
