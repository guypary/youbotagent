from dotenv import load_dotenv
import os
from steps.step1_user_input import get_user_input
from steps.step2_fetch_video_data import fetch_video_data
from steps.step3_process_transcript import process_transcript
from steps.step4_store_pinecone import store_in_pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from getpass import getpass
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent

# Load environment variables
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or getpass("Enter your OpenAI API key: ")


# LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="youtube-agent"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Initialize OpenAI Embeddings
model_name = "text-embedding-ada-002"
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

def main():
    # Prompt user to choose input type once
    input_type = input("Choose input type - (1) Text, (2) Voice: ").strip()
    if input_type not in ["1", "2"]:
        print("Invalid input type. Please restart and choose either (1) Text or (2) Voice.")
        return

    # Ask for video ID at the beginning of the session
    video_id = input("Enter YouTube video ID: ").strip()
    transcript = fetch_video_data(video_id)
    if not transcript:
        print("No transcript available for the given video. Exiting...")
        return

    # Process transcript into chunks
    chunks = process_transcript(transcript)
    if not chunks:
        print("Failed to process transcript into chunks. Exiting...")
        return

    # Store transcript chunks in Pinecone
    index = store_in_pinecone(chunks, embed.embed_documents)
    vectorstore = Pinecone(index, embed.embed_query, 'text')

    #print(vectorstore.similarity_search( "What is LLM", k=3 ))

        # chat completion llm
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-4o-mini',
        temperature=0.0
    )
  
    # retrieval qa chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
        # conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )

    from langchain.agents import Tool

    tools = [
        Tool(
            name='Knowledge Base',
            func=qa.run,
            description=(
                'use this tool when answering general knowledge queries to get '
                'more information about the topic'
            )
        )
    ]

    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=conversational_memory
    )
    

    while True:
        try:
            user_query = input("You: ").strip()
            if user_query.lower() == "/bye":
                print("Goodbye!")
                break
            print(agent(user_query))
        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break


        



if __name__ == "__main__":
    main()