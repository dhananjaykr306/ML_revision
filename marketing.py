# @Author: Dhananjay Kumar
# @Date: 17-01-2025
# @Last Modified by: Dhananjay Kumar
# @Last Modified time: 17-01-2025
# @Title: Python program to designing a prompt for an LLM to generate a marketing campaign idea for a new product. Describe your approach to creating this prompt, including any specific techniques you would use to ensure creativity and relevance in the generated output.


import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama

LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"

# Load environment variables
load_dotenv()

# Langsmith Tracking Configuration
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACKING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "default_project")

# Streamlit UI Setup
st.title("Langchain Chatbot")
input_text = st.text_input("What question do you have in your mind related to marketing?")

# Define Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helping assiatnce for a marketing campaign idea for a new product. Please respond to the question asked ."),
    ("user", "Question: {question}")
])
def main():
    # Initialize Chroma client with local persist directory
    chroma_client = Client(persist_directory="path_to_your_local_directory")

    # App Mode Selection
    app_mode = st.sidebar.selectbox("Select Mode", ["Upload PDF", "Ask Query"])

    if app_mode == "Upload PDF":
        # PDF File Upload
        st.title("PDF File Uploader and Text Processing with LangChain")
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        if uploaded_file is not None:
            # Save the uploaded file temporarily
            st.info("Generating PDF embedding...")
            with open("temp_uploaded_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load the PDF using PyPDFLoader
            loader = PyPDFLoader("temp_uploaded_file.pdf")
            docs = loader.load()

            # Use RecursiveCharacterTextSplitter to split the document
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            final_docs = splitter.split_documents(docs)

            # Create or load a collection in ChromaDB
            collection = chroma_client.get_or_create_collection("pdf_embeddings")

            # Insert the documents into ChromaDB
            embeddings = OllamaEmbeddings(model="gemma:2b")
            for i, doc in enumerate(final_docs):
                embedding: Embedding = embeddings.embed_documents([doc.page_content])[0]
                collection.add(
                    ids=[f"chunk_{i}"],
                    documents=[doc.page_content],
                    metadatas=[{"chunk_id": i}],
                    embeddings=[embedding],
                )
            st.success(f"Inserted {len(final_docs)} documents into ChromaDB.")

    elif app_mode == "Ask Query":
        # Query Tab
        st.title("Ask a Query Based on Uploaded PDF")
        query = st.text_input("Enter your query:")

        if query:
            # Generate embedding for the query
            st.info("Generating query embedding...")
            embeddings = OllamaEmbeddings(model="gemma:2b")
            query_embedding: Embedding = embeddings.embed_documents([query])[0]

            # Get or create the collection
            collection = chroma_client.get_or_create_collection("pdf_embeddings")

            # Perform similarity search
            st.info("Performing similarity search...")
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3,  # Get top 3 similar results
            )

            # Combine the top results into a single string for summarization
            combined_content = " ".join([" ".join(doc) if isinstance(doc, list) else doc for doc in results["documents"]])

            # Load API key from environment variable and generate summary
            try:
                load_dotenv()
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("API key not found. Ensure it's set in the .env file.")
                genai.configure(api_key=api_key)

                # Create a generative model
                generation_config = {
                    "temperature": 0,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
                model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
                chat_session = model.start_chat()

                summary = summarization(combined_content, chat_session, query)
                st.subheader("Summary of Results:")
                st.write(summary)

                # Insert the query and summary into ChromaDB for future use
                embedding_summary: Embedding = embeddings.embed_documents([summary])[0]
                collection.add(
                    ids=[f"query_{query}"],
                    documents=[summary],
                    metadatas=[{"query": query}],
                    embeddings=[embedding_summary],
                )
                st.success("Query and summary inserted into ChromaDB.")

            except Exception as e:
                st.error(f"Error: {e}")

    # Help Button at the bottom of the page
    if st.button("Help/Status"):
        st.subheader("Getting Help or Status")
        st.write(""" 
            1. **Upload PDF**: Upload a PDF file, and the app will process its content and generate document embeddings for storage in ChromaDB.
            2. **Ask Query**: Enter a query based on the uploaded PDF, and the app will search the most relevant content in the database, summarize it, and store the query and summary back into ChromaDB.
            3. **Commands**:
               - "Upload PDF": Uploads a PDF for processing.
               - "Ask Query": Asks a query based on the processed PDF.
        """)

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    st.write(f"Total time taken: {t2 - t1} seconds.")

# Configure LLM (Ollama Model)
try:
    llm = Ollama(model="gemma:2b")
except Exception as e:
    st.error("Failed to initialize Ollama model. Please check your setup.")
    st.stop()

# Combine Prompt and LLM into a Chain
chain = LLMChain(prompt=prompt, llm=llm)

# Process User Input and Generate Response
if input_text:
    try:
        response = chain.run({"question": input_text})
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")
