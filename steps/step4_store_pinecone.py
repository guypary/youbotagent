import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm

# Load environment variables
load_dotenv()

# API keys and environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "youtube-transcripts"
DIMENSION = 1536  # Dimensionality for OpenAI's ADA model embeddings

# Create a Pinecone instance
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Specify the serverless environment
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)

# Ensure the index exists or create it
if INDEX_NAME not in [index["name"] for index in pinecone_client.list_indexes()]:
    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="dotproduct",  # Use dot product as the similarity metric
        spec=spec
    )
    print(f"Index '{INDEX_NAME}' created successfully.")

    # Wait for the index to be ready
    while not pinecone_client.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)
else:
    print(f"Index '{INDEX_NAME}' already exists.")

# Connect to the index
index = pinecone_client.Index(INDEX_NAME)


def store_in_pinecone(chunks, embedding_function):
    """
    Stores transcript chunks in Pinecone.
    :param chunks: List of text chunks.
    :param embedding_function: Function to generate embeddings for text chunks.
    """
    try:
        # Prepare data for batch upload
        batch_size = 100
        for i in tqdm(range(0, len(chunks), batch_size), desc="Upserting to Pinecone"):
            # Batch end
            i_end = min(len(chunks), i + batch_size)
            batch = chunks[i:i_end]

            # Generate embeddings for the batch
            embeddings = embedding_function(batch)  # embed_documents returns a list of lists
            if not all(isinstance(e, list) and all(isinstance(v, float) for v in e) for e in embeddings):
                raise ValueError("Embeddings must be a list of lists of floats.")

            # Prepare metadata and IDs
            metadata = [{"text": text} for text in batch]
            ids = [f"chunk_{idx}" for idx in range(i, i_end)]

            # Upsert into Pinecone
            index.upsert(vectors=list(zip(ids, embeddings, metadata)))
        print(f"Successfully stored {len(chunks)} chunks in Pinecone index '{INDEX_NAME}'.")
        return index
    except Exception as e:
        print(f"An error occurred while storing data in Pinecone: {e}")
        raise