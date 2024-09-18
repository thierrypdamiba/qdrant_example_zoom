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

def query_vector_db(sentiment_query):
    query_vector = model.encode(sentiment_query).tolist()
    
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

def get_anthropic_response(sentiment_query, search_results):
    if not anthropic_api_key:
        raise ValueError("Anthropic API key not set.")

    client = anthropic.Anthropic(api_key=anthropic_api_key)

    prompt = f"""
    You are an AI assistant tasked with analyzing the sentiment of company meeting transcripts.
    Your goal is to assess the overall sentiment on a scale from 1 to 10, where:
    
    1 = Extremely negative
    5 = Neutral
    10 = Extremely positive
    
    Please analyze the following meeting summaries and provide a sentiment analysis:
    
    Query: {sentiment_query}
    
    Meeting Summaries:
    {search_results}
    
    In your response, include:
    1. The overall sentiment score (1-10) for the meetings related to the query
    2. A brief explanation of why you assigned this score
    3. Key phrases or topics that influenced your sentiment analysis
    4. If multiple meetings are relevant, provide a sentiment analysis for each
    
    Format your response as follows:
    Overall Sentiment Score: [score]
    Explanation: [your explanation]
    Key Factors: [list of key phrases or topics]
    Individual Meeting Analyses: [if applicable]
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

def analyze_sentiment(query):
    search_results = query_vector_db(query)
    sentiment_analysis = get_anthropic_response(query, search_results)
    return sentiment_analysis

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentiment_query = " ".join(sys.argv[1:])
        result = analyze_sentiment(sentiment_query)
        print(f"Sentiment Analysis:\n{result}")
    else:
        print("No sentiment query provided. Please specify a query to analyze sentiment in the meeting transcripts.")