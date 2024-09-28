import json
import os
import uuid
import base64
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import warnings
from qdrant_client.http.exceptions import ResponseHandlingException
import time

warnings.filterwarnings('ignore', category=FutureWarning)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Debug: Print all environment variables
print("Debug: All environment variables:")
for key, value in os.environ.items():
    print(f"{key}: {value[:10]}..." if key.lower().endswith('key') else f"{key}: {value}")

# Initialize Qdrant client
qdrant_url = os.getenv('QDRANT_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')

print(f"Debug: QDRANT_URL = {qdrant_url}")
print(f"Debug: QDRANT_API_KEY = {qdrant_api_key[:5]}..." if qdrant_api_key else "Debug: QDRANT_API_KEY is not set")

if not qdrant_url or not qdrant_api_key:
    raise ValueError("QDRANT_URL or QDRANT_API_KEY environment variables are not set")

try:
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    print("Debug: QdrantClient initialized successfully")
except Exception as e:
    print(f"Debug: Error initializing QdrantClient: {str(e)}")
    raise

# Define collection name
collection_name = "zommers"

def ensure_collection_exists():
    print(f"Debug: QDRANT_URL = {qdrant_url}")
    print(f"Debug: QDRANT_API_KEY = {qdrant_api_key[:5]}..." if qdrant_api_key else "Debug: QDRANT_API_KEY is not set") # Only print the first 5 characters of the API key for security

    try:
        collections = client.get_collections().collections
        print(f"Successfully connected to Qdrant Cloud. Found {len(collections)} collections.")
        
        # Check if the collection exists
        collection_names = [collection.name for collection in collections]
        if collection_name not in collection_names:
            print(f"Collection '{collection_name}' does not exist. Creating it now...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists.")
        
    except ResponseHandlingException as e:
        print(f"Failed to connect to Qdrant Cloud: {e}")
        print(f"Host: {qdrant_url}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def base64_to_uuid(base64_string):
    try:
        # Remove any padding and convert to bytes
        base64_string = base64_string.rstrip('=')
        byte_string = base64.urlsafe_b64decode(base64_string + '=='*(-len(base64_string) % 4))
        
        # Convert bytes to UUID
        return str(uuid.UUID(bytes=byte_string[:16]))
    except:
        # If conversion fails, generate a new UUID
        return str(uuid.uuid4())

def insert_with_retry(client, collection_name, points, max_retries=3, initial_delay=1):
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=collection_name, points=points)
            print(f"Successfully inserted {len(points)} points")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"Insertion failed. Retrying in {delay} seconds... Error: {str(e)}")
                time.sleep(delay)
            else:
                print(f"Failed to insert data after {max_retries} attempts. Error: {str(e)}")
                return False

def insert_data_to_qdrant(data):
    try:
        points = []
        for i, recording in enumerate(data.get('recordings', [])):
            summary = recording.get('summary', {})
            if isinstance(summary, dict):
                summary_text = summary.get('summary_overview', recording.get('topic', ''))
            else:
                summary_text = recording.get('topic', '')

            if not summary_text:
                summary_text = "No summary or topic available"

            vector = model.encode(summary_text).tolist()

            point_id = base64_to_uuid(recording['uuid'])

            print(f"Inserting recording {i + 1}:")
            print(f"  UUID: {point_id}")
            print(f"  Topic: {recording['topic']}")
            print(f"  Summary text: {summary_text}")
            print(f"  Vector (first 5 elements): {vector[:5]}")

            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    'topic': recording['topic'],
                    'start_time': recording['start_time'],
                    'duration': recording['duration'],
                    'summary': summary if isinstance(summary, dict) else {}
                }
            )
            points.append(point)

        if points:
            print(f"Total points prepared for insertion: {len(points)}")
            response = insert_with_retry(client, collection_name, points)
            print(f"Qdrant response: {response}")
            print(f"Inserted {len(points)} points into Qdrant.")
        else:
            print("No valid points to insert.")

        # Verify insertion
        collection_info = client.get_collection(collection_name)
        print(f"Collection info after insertion: {collection_info}")

    except Exception as e:
        print(f"Error inserting data to Qdrant: {e}")

if __name__ == "__main__":
    ensure_collection_exists()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(data_dir, file_name)
            print(f"Processing file: {file_path}")
            data = load_data(file_path)
            insert_data_to_qdrant(data)

print("Data insertion complete.")