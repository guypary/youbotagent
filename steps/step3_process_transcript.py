from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_transcript(transcript):
    """
    Splits the transcript into smaller chunks for efficient storage and retrieval.
    
    Args:
        transcript (str): The full transcript text to be processed.
        
    Returns:
        list: A list of smaller text chunks from the original transcript.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_text(transcript)
        return chunks
    except Exception as e:
        print(f"An error occurred while processing the transcript: {e}")
        return []