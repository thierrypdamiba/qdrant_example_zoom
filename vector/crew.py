import sys
import os
import re
from crewai import Agent, Task, Crew
from qdrant_client import QdrantClient
from openai import OpenAI
from langchain.tools import BaseTool
from typing import Any, Type
from pydantic import BaseModel, Field
import anthropic

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

# Set API keys from config
os.environ['OPENAI_API_KEY'] = config.get('openai_api_key')
ANTHROPIC_API_KEY = config.get('anthropic_api_key')

class SearchInput(BaseModel):
    """Input for the search tool."""
    query: str = Field(..., description="The search query to find relevant meeting recordings")

class AnthropicInput(BaseModel):
    """Input for the Anthropic analysis tool."""
    query: str = Field(..., description="The original query")
    search_results: list = Field(..., description="The search results to analyze")

class QdrantSearchTool(BaseTool):
    name: str = "search_meetings"
    description: str = "Search through meeting recordings to find relevant information"
    args_schema: Type[BaseModel] = SearchInput
    
    qdrant_client: QdrantClient
    openai_client: OpenAI
    
    def __init__(self, qdrant_client: QdrantClient, openai_client: OpenAI):
        super().__init__()
        self.qdrant_client = qdrant_client
        self.openai_client = openai_client
        
    def _run(self, query: str) -> Any:
        # Get embedding from OpenAI
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_vector = response.data[0].embedding

        # Search Qdrant
        search_result = self.qdrant_client.search(
            collection_name='user_recordings',
            query_vector=query_vector,
            limit=5
        )

        # Format results
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

    async def _arun(self, query: str) -> Any:
        raise NotImplementedError("Async not implemented")

class AnthropicAnalysisTool(BaseTool):
    name: str = "analyze_with_anthropic"
    description: str = "Analyze search results using Anthropic's Claude to provide insights"
    args_schema: Type[BaseModel] = AnthropicInput
    
    def __init__(self):
        super().__init__()
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    def _run(self, query: str, search_results: list) -> Any:
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0,
            system="You are an AI assistant tasked with answering queries based on search results.",
            messages=[
                {
                    "role": "user",
                    "content": f"Based on the following search results, please provide a concise answer to the query: '{query}'\n\nSearch Results:\n{search_results}\n\nPlease synthesize the information from these results to directly answer the query. If the information is not sufficient to answer the query, please state that clearly."
                }
            ]
        )
        return message.content

    async def _arun(self, query: str, search_results: list) -> Any:
        raise NotImplementedError("Async not implemented")

def get_crew_response(query):
    # Initialize clients
    client = QdrantClient("localhost", port=6333)
    openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    # Create tool instances
    search_tool = QdrantSearchTool(qdrant_client=client, openai_client=openai_client)
    analysis_tool = AnthropicAnalysisTool()
    
    # Create agents
    researcher = Agent(
        role="Research Analyst",
        goal="Search through meeting recordings and extract relevant information",
        backstory="""You are an expert at analyzing meeting recordings and extracting 
                  key insights. You excel at understanding context and finding relevant information.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool]
    )

    analyst = Agent(
        role="Content Analyst",
        goal="Analyze search results and create comprehensive responses",
        backstory="""You specialize in analyzing search results and creating clear, 
                  insightful responses using advanced AI analysis tools.""",
        verbose=True,
        allow_delegation=False,
        tools=[analysis_tool]
    )

    # Create tasks
    search_task = Task(
        description=f"""Search through the meeting recordings for information about: '{query}'
                    Use the search_meetings tool to find relevant information.""",
        agent=researcher
    )

    analysis_task = Task(
        description="""Analyze the search results using the Anthropic analysis tool to create 
                    a comprehensive and accurate response.""",
        agent=analyst
    )

    # Create and run crew
    crew = Crew(
        agents=[researcher, analyst],
        tasks=[search_task, analysis_task],
        verbose=2
    )

    result = crew.kickoff()
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])  # Allow multi-word queries
        answer = get_crew_response(query)
        print(f"Answer: {answer}")
    else:
        print("No query provided.") 