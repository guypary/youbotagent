from dotenv import load_dotenv
import os
import gradio as gr
import speech_recognition as sr
from steps.step2_fetch_video_data import fetch_video_data
from steps.step3_process_transcript import process_transcript
from steps.step4_store_pinecone import store_in_pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool

from langchain.agents import AgentType
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, AgentType


# Load environment variables
load_dotenv()

# Initialize global components
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embed = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

# LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "youtube-agent"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Global variables for vectorstore, QA, and agent
vectorstore = None
qa = None
agent = None


def process_video(video_id):
    """Process video and store data in Pinecone"""
    global vectorstore

    # Fetch and process transcript
    transcript = fetch_video_data(video_id)
    if not transcript:
        raise gr.Error("No transcript available for this video")

    chunks = process_transcript(transcript)
    if not chunks:
        raise gr.Error("Failed to process transcript")

    # Store in Pinecone
    index = store_in_pinecone(chunks, embed.embed_documents)
    vectorstore = Pinecone(index, embed.embed_query, 'text')

    return "Video processed successfully! You can now ask questions."


def initialize_qa_and_agent():
    """Initialize LLM, QA, and Agent using correct retrieval"""
    global vectorstore, qa, agent

    if vectorstore is None:
        raise ValueError("Vectorstore is not initialized. Please process a video first.")

    # Initialize LLM
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-4',
        temperature=0.0
    )

    # Verify retrieval is working
    retriever = vectorstore.as_retriever()
    
    #test_query = "What is this video about?"
    #retrieved_docs = retriever.get_relevant_documents(test_query)
    
    #if not retrieved_docs:
    #    raise ValueError("❌ ERROR: Pinecone is not returning documents! Check storage.")

    # Create QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # Define tool using RetrieverTool (Ensures proper retrieval)
    knowledge_tool = create_retriever_tool(
        retriever=retriever,
        name="video_qa",
        description="Retrieves knowledge from the video transcript."
    )


    system_prompt = """Assistant is an AI-powered agent designed to answer questions strictly based on data stored in the vector database. Assistant does not use external knowledge, assumptions, or general information when responding. 

                        When a query is received, Assistant will:  
                        1. Attempt to retrieve relevant data from the vector database.  
                        2. If relevant data is found, generate a response based strictly on the retrieved content.  
                        3. If no relevant data is found, respond with: "I don’t have data about your query."

                        Assistant must not speculate, assume information, or provide answers outside the scope of the data retrieved from the vector database.

                        Example interactions:

                        User: "What is the revenue of Company X in 2023?"  
                        Assistant: "According to the data, Company X’s revenue in 2023 was $5.2 million."

                        User: "What is the revenue of Company Y?" (Data not found in DB)  
                        Assistant: "I don’t have data about your query."

                        Assistant ensures that all responses are clear, precise, and solely based on available data."""

    # Initialize Agent
    agent = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=[knowledge_tool],
        llm=llm,
        memory=ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=5,
            return_messages=True
        ),
        verbose=True,  # Enable debugging
        max_iterations=3,
        early_stopping_method='generate',
        handle_parsing_errors=True,
        agent_kwargs={ "system_message": system_prompt,}
    )
    print("✅ Agent successfully initialized with Retriever Tool.")
    return qa, agent


def transcribe_audio(audio_path):
    """Convert speech to text using Google's speech recognition"""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "[Error] Could not understand audio"
    except Exception as e:
        return f"[Error] Audio processing failed: {str(e)}"


def handle_query(input_type, query, audio_input, chat_history):
    """Handle user input and generate response"""
    global agent

    # Check if the agent is initialized
    if agent is None:
        return [("System", "Please process a video first!")]

    # Process audio input
    if input_type == "Voice" and audio_input:
        query = transcribe_audio(audio_input)
        if "Error" in query:
            chat_history.append(("Voice Input", query))
            return chat_history

    # Ensure query is valid
    if not query:
        return chat_history

    # Get agent response
    try:
        response = agent.run(query)
        chat_history.append((query, response))
    except Exception as e:
        response = f"Error processing query: {str(e)}"
        chat_history.append((query, response))

    return chat_history


def main():
    """Main function to initialize and run the Gradio app"""
    global vectorstore, qa, agent

    # Create Gradio interface
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# YouTube Video QA Assistant 🎬")

        with gr.Row():
            video_id = gr.Textbox(label="YouTube Video ID", placeholder="Enter YouTube video ID...")
            process_btn = gr.Button("Process Video")

        status = gr.Textbox(label="Processing Status", interactive=False)

        with gr.Tabs():
            with gr.TabItem("Text Chat"):
                text_input = gr.Textbox(label="Type your question", placeholder="Ask about the video content...")
                text_submit = gr.Button("Send Text")

            with gr.TabItem("Voice Chat"):
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record your question")
                audio_submit = gr.Button("Send Voice")

        chatbot = gr.Chatbot(label="Conversation History")
        clear_btn = gr.Button("Clear Chat")

        # Event handlers
        def process_and_initialize(video_id):
            """Processes video and initializes QA/Agent"""
            status_text = process_video(video_id)
            initialize_qa_and_agent()
            return status_text

        process_btn.click(
            process_and_initialize,
            inputs=video_id,
            outputs=status
        )

        text_submit.click(
            handle_query,
            inputs=[gr.Text("Text", visible=False), text_input, gr.Audio(None, visible=False), chatbot],
            outputs=chatbot
        ).then(lambda: "", None, text_input)

        audio_submit.click(
            handle_query,
            inputs=[gr.Text("Voice", visible=False), gr.Textbox(None, visible=False), audio_input, chatbot],
            outputs=chatbot
        )

        clear_btn.click(lambda: [], None, chatbot)

    # Launch Gradio app
    demo.launch(share=True)


if __name__ == "__main__":
    main()