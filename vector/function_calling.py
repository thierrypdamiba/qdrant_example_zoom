import sys
import os
import re
from llama_index.llms.anthropic import Anthropic
from llama_index.core.tools import FunctionTool, ToolMetadata
from llama_index.core.agent import FunctionCallingAgent
from typing import Any, List, Dict
from qdrant_client import QdrantClient
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import anthropic
from datetime import datetime

# Load configuration
def load_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    config_path = os.path.join(root_dir, 'config.js')
    
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    config = {}
    pattern = r"(\w+):\s*['\"]?([\w\-\.]+)['\"]?"
    matches = re.findall(pattern, config_content)
    for key, value in matches:
        config[key] = value
    
    return config

config = load_config()

# Initialize clients
llm = Anthropic(
    model="claude-3-opus-20240229",
    api_key=config.get('anthropic_api_key')
)
openai_client = OpenAI(api_key=config.get('openai_api_key'))
qdrant_client = QdrantClient("localhost", port=6333)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define calculator tools
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and return the result"""
    return a + b

# Define Qdrant search function
def search_meetings(query: str) -> List[Dict]:
    """
    Search through meeting recordings using Qdrant vector similarity
    Args:
        query: The search query
    Returns:
        List of relevant meeting recordings with their metadata
    """
    # Get embedding using sentence-transformers
    query_vector = embedding_model.encode(query).tolist()

    # Search Qdrant
    search_results = qdrant_client.search(
        collection_name='user_recordings',
        query_vector=query_vector,
        limit=5
    )

    # Format results
    formatted_results = []
    for hit in search_results:
        result = {
            "score": hit.score,
            "topic": hit.payload.get('topic', 'N/A'),
            "start_time": hit.payload.get('start_time', 'N/A'),
            "duration": hit.payload.get('duration', 'N/A'),
            "summary": hit.payload.get('summary', {}).get('summary_overview', 'N/A')
        }
        formatted_results.append(result)
    
    return formatted_results

# Create tools
multiply_tool = FunctionTool.from_defaults(
    fn=multiply,
    name="multiply",
    description="Multiply two integers together"
)

add_tool = FunctionTool.from_defaults(
    fn=add,
    name="add",
    description="Add two integers together"
)

search_tool = FunctionTool.from_defaults(
    fn=search_meetings,
    name="search_meetings",
    description="Search through meeting recordings to find relevant information"
)

def analyze_meeting_content(meeting_data: Dict) -> Dict:
    """
    Analyze a specific meeting's content using Claude
    Args:
        meeting_data: Dictionary containing meeting information
    Returns:
        Dictionary with analysis results
    """
    client = anthropic.Anthropic(api_key=config.get('anthropic_api_key'))
    
    prompt = f"""
    Please analyze this meeting:
    Topic: {meeting_data.get('topic')}
    Summary: {meeting_data.get('summary')}
    
    Provide:
    1. Key discussion points
    2. Main decisions or action items
    3. Overall sentiment
    """
    
    message = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=500,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "meeting_topic": meeting_data.get('topic'),
        "analysis": message.content,
        "timestamp": datetime.now().isoformat()
    }

analysis_tool = FunctionTool.from_defaults(
    fn=analyze_meeting_content,
    name="analyze_meeting",
    description="Analyze a specific meeting's content to extract key points and sentiment"
)

def get_smart_agent():
    """
    Get an agent that automatically decides which tools to use based on the query context
    """
    return FunctionCallingAgent.from_tools(
        [multiply_tool, add_tool, search_tool, analysis_tool],
        llm=llm,
        verbose=True,
        system_prompt="""You are an intelligent assistant that can handle various types of queries.
        For each query, analyze its intent and choose the most appropriate tools:
        
        - For mathematical calculations, use the calculator tools (multiply, add)
        - For meeting information queries, use the search_meetings tool
        - For in-depth meeting analysis, first use search_meetings then analyze_meeting
        
        Always provide clear, structured responses and explain your reasoning.
        If a query requires multiple tools, use them in the most logical sequence.
        """
    )

def process_query(query: str) -> str:
    """
    Process any query by letting the agent decide which tools to use
    """
    agent = get_smart_agent()
    response = agent.chat(query)
    return str(response)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
        try:
            result = process_query(query)
            print(f"\nResult: {result}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")
    else:
        print("Please provide a query as a command line argument.") 