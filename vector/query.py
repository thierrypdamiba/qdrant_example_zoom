import sys
import os
import re
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import anthropic
import json
import warnings
import traceback

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Load configuration
def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    config_path = os.path.join(root_dir, '.env.local')
    
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                config[key] = value.strip('"')
    
    return config

config = load_config()

# Initialize Qdrant client
qdrant_url = config.get('QDRANT_URL')
qdrant_api_key = config.get('QDRANT_API_KEY')

if not qdrant_url or not qdrant_api_key:
    raise ValueError("QDRANT_URL or QDRANT_API_KEY environment variables are not set")

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define collection name
collection_name = 'zommers'

def query_vector_db(query):
    try:
        # Perform search
        search_result = client.search(
            collection_name=collection_name,
            query_vector=model.encode(query).tolist(),
            limit=5  # Adjust limit as needed
        )

        # Format the search results
        formatted_results = []
        for hit in search_result:
            result = {
                "Score": hit.score,
                "Topic": hit.payload.get('topic', 'N/A'),
                "Start Time": hit.payload.get('start_time', 'N/A'),
                "Duration": hit.payload.get('duration', 'N/A'),
                "Summary": hit.payload.get('summary', {}).get('summary_overview', 'N/A')
            }
            formatted_results.append(result)
        
        return formatted_results

    except Exception as e:
        print(f"Error in query_vector_db: {str(e)}")
        raise

def get_anthropic_response(query, search_results):
    api_key = config.get('ANTHROPIC_API_KEY')
    
    if not api_key:
        raise ValueError("Anthropic API key not set in config.")

    client = anthropic.Anthropic(api_key=api_key)
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system="You are an AI assistant tasked with answering queries based on search results. Provide a concise and informative summary of the relevant information.",
        messages=[
            {
                "role": "user",
                "content": f"Based on the following search results, please provide a concise answer to the query: '{query}'\n\nSearch Results:\n{json.dumps(search_results, indent=2)}\n\nPlease synthesize the information from these results to directly answer the query. If the information is not sufficient to answer the query, please state that clearly."
            }
        ]
    )
    
    # Handle the case where content is a list
    if isinstance(message.content, list):
        # Assuming the first item contains the text content
        content = message.content[0].text if message.content else ""
    else:
        content = message.content

    # Remove any newline characters and strip whitespace
    return content.replace('\n', ' ').strip()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Please provide a query as a command-line argument."}))
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"Received query: {query}")  # Add this line
    
    try:
        search_results = query_vector_db(query)
        print(f"Search results: {json.dumps(search_results, indent=2)}")  # Add this line
        answer = get_anthropic_response(query, search_results)
        print(f"Raw Anthropic response: {answer}")  # Add this line for debugging
        
        result = {
            "query": query,
            "answer": answer
        }
        print(json.dumps(result))
    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
        print(json.dumps({"error": error_message}), file=sys.stderr)
        sys.exit(1)