import os
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chromadb.utils import embedding_functions

load_dotenv()

CHROMA_PATH = r"chroma_db"
COLLECTION_NAME = "json_documents_collection"

app = FastAPI(
    title="Minneapolis Institute of Art RAG API",
    description="API for querying the Mia art object knowledge base using Gemini Flash.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str

chroma_client = None
collection = None
gemini_model = None
embedding_model = None 

SYSTEM_PROMPT_TEMPLATE = """
You are an expert art historian and guide for the Minneapolis Institute of Art (Mia).
Your primary task is to answer user questions exclusively based on the provided "ART OBJECT CONTEXT".
Your goal is to provide accurate, concise, and informative answers about the museum's collection.

**Guidelines:**
- **Strictly use the provided context:** Do not use any external knowledge.
- **Do not invent information:** If the "ART OBJECT CONTEXT" does not contain the answer, state clearly that the information is not available in the provided data.
- **Focus on art object details:** Prioritize information about the artwork, artist, date, medium, collection, and relevant historical or artistic context found in the provided text.
- **Be helpful and polite, but concise.**

**ART OBJECT CONTEXT:**
{context}

**User Question:**
{question}

**Answer:**
"""

@app.on_event("startup")
async def startup_event():
    global chroma_client, collection, gemini_model, embedding_model

    try:
        gemini_api_key = os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=gemini_api_key)
        print("Gemini API configured.")
    except KeyError:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not found in environment variables. Please set it."
        )

    try:
        embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        print("Embedding model 'all-MiniLM-L6-v2' initialized.")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error initializing embedding model: {e}. Ensure 'sentence-transformers' is installed."
        )

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
       
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_model
        )
        print(f"ChromaDB collection '{COLLECTION_NAME}' loaded/created.")
        if collection.count() == 0:
            print(f"WARNING: ChromaDB collection '{COLLECTION_NAME}' is empty. Have you run your indexing script?")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error initializing ChromaDB: {e}. Check CHROMA_PATH and collection integrity."
        )

    try:
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')
        print("Gemini model 'gemini-2.0-flash-lite' initialized.")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error initializing Gemini model: {e}. Check model name and API key access."
        )

@app.get("/health")
async def health_check():
    """
    Checks the health of the API and its dependencies.
    """
    status = {
        "api_status": "online",
        "chromadb_connected": chroma_client is not None,
        "gemini_model_loaded": gemini_model is not None,
        "embedding_model_loaded": embedding_model is not None,
        "collection_count": collection.count() if collection else "N/A"
    }
    if not all(status.values()) and status["api_status"] == "online": 
         raise HTTPException(status_code=500, detail="One or more RAG components failed to initialize properly.")
    return status

@app.post("/query")
async def query_rag(request: QueryRequest):
    """
    Receives a user query, performs RAG, and returns a generated response.
    """
    if not all([collection, gemini_model, embedding_model]):
        raise HTTPException(status_code=500, detail="RAG components not fully initialized. Please check server logs.")

    user_query = request.query
    print(f"Received query: {user_query}")

    try:
        results = collection.query(
            query_texts=[user_query],
            n_results=4
        )

        if not results or not results['documents'] or not results['documents'][0]:
            context_str = "No relevant context found in the knowledge base."
        else:
            retrieved_documents = []
            for doc_list in results['documents']:
                for doc_content in doc_list:
                    retrieved_documents.append(doc_content)
            context_str = "\n\n".join(retrieved_documents)

        final_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            context=context_str,
            question=user_query
        )

        messages = [
            {"role": "user", "parts": [final_system_prompt]}
        ]

        response = gemini_model.generate_content(messages)

        if not response.parts or not response.parts[0].text:
             raise HTTPException(status_code=500, detail="Gemini model returned an empty response.")

        return {"query": user_query, "response": response.text, "context": retrieved_documents}

    except genai.APIError as e:
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
