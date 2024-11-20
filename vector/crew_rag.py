import sys
import os
import re
from crewai import Agent, Task, Crew
from qdrant_client import QdrantClient
from openai import OpenAI
from langchain.tools import BaseTool
from typing import Any, Type
from pydantic import BaseModel, Field

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
os.environ['OPENAI_API_KEY'] = config.get('openai_api_key')

class SearchInput(BaseModel):
    query: str = Field(..., description="The search query for meeting recordings")

class QdrantSearchTool(BaseTool):
    name: str = "search_meetings"
    description: str = "Search through meeting recordings using vector similarity"
    args_schema: Type[BaseModel] = SearchInput
    
    def __init__(self, qdrant_client: QdrantClient, openai_client: OpenAI):
        super().__init__()
        self.qdrant_client = qdrant_client
        self.openai_client = openai_client
        
    def _run(self, query: str) -> list:
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_vector = response.data[0].embedding

        search_results = self.qdrant_client.search(
            collection_name='user_recordings',
            query_vector=query_vector,
            limit=5
        )
        
        return [
            {
                "content": hit.payload.get('summary', {}).get('summary_overview', ''),
                "metadata": {
                    "topic": hit.payload.get('topic'),
                    "score": hit.score,
                    "start_time": hit.payload.get('start_time')
                }
            }
            for hit in search_results
        ]

def get_rag_response(query: str) -> str:
    # Initialize clients
    client = QdrantClient("localhost", port=6333)
    openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    # Create search tool
    search_tool = QdrantSearchTool(qdrant_client=client, openai_client=openai_client)
    
    # Create agents
    researcher = Agent(
        role='Research Assistant',
        goal='Search and retrieve relevant meeting information',
        backstory="""You are a research assistant specialized in searching through meeting recordings
                  and finding the most relevant information. You understand context and can identify
                  key information effectively.""",
        tools=[search_tool],
        verbose=True
    )
    
    writer = Agent(
        role='Technical Writer',
        goal='Create comprehensive and accurate responses',
        backstory="""You are a technical writer specialized in creating clear and concise responses
                  based on research findings. You excel at synthesizing information and presenting
                  it in an easily digestible format.""",
        verbose=True
    )
    
    # Create tasks
    research_task = Task(
        description=f"""Search through the meeting recordings database for information about: '{query}'
                    Analyze the search results and identify the most relevant information.
                    Make sure to note any relevant context, dates, or specific details.""",
        agent=researcher
    )
    
    writing_task = Task(
        description="""Using the research findings, create a clear and comprehensive response.
                    Include specific details and context from the meetings where relevant.
                    Structure the response in a logical way and ensure it directly addresses
                    the original query.""",
        agent=writer
    )
    
    # Create and run crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=2
    )
    
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
        result = get_rag_response(query)
        print(f"\nFinal Answer: {result}")
    else:
        print("Please provide a query as a command line argument.") 