import sys
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import anthropic
import warnings
import re

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

def query_vector_db(search_query):
    query_vector = model.encode(search_query).tolist()
    
    # Perform search
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=10
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

def get_anthropic_response(search_query, search_results):
    if not anthropic_api_key:
        raise ValueError("Anthropic API key not set.")

    client = anthropic.Anthropic(api_key=anthropic_api_key)

    prompt = f"""
    You are an AI assistant tasked with searching through past meeting transcripts.
    Your goal is to find relevant information based on the user's query.
    
    Please search for the following query:
    {search_query}
    
    Based on the following meeting summaries:
    {search_results}
    
    Provide a concise summary of the relevant information found in the meeting transcripts.
    Include the date of the meeting if available, and any key points or decisions made related to the query.
    If multiple relevant meetings are found, summarize the information from each meeting separately.
    
    If no relevant information is found, please state that clearly.
    """

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return message.content

def search_meetings(query):
    search_results = query_vector_db(query)
    meeting_summary = get_anthropic_response(query, search_results)
    return meeting_summary

if __name__ == "__main__":
    if len(sys.argv) > 1:
        search_query = " ".join(sys.argv[1:])
        result = search_meetings(search_query)
        print(f"Meeting Search Results:\n{result}")
    else:
        print("No search query provided. Please specify a query to search the meeting transcripts.")