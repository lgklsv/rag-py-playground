import chromadb
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    exit()

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(name="json_documents_collection")

model = genai.GenerativeModel('gemini-2.0-flash-lite')
print("Gemini model 'gemini-2.0-flash-lite' initialized.")

user_query = input("What do you want to know about Minneapolis Institute of Art ?\n\n")

print("Searching ChromaDB for relevant context...")
results = collection.query(
    query_texts=[user_query],
    n_results=4
)

retrieved_documents = []
for doc_list in results['documents']:
    for doc_content in doc_list:
        retrieved_documents.append(doc_content)

context_str = "\n\n".join(retrieved_documents)

system_prompt_template = """
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

final_system_prompt = system_prompt_template.format(
    context=context_str,
    question=user_query
)

messages = [
    {"role": "user", "parts": [final_system_prompt]},
]

print("\n\nGenerating response with Gemini...\n")

try:
    response = model.generate_content(messages)

    print("\n\n---------------------\n\n")
    print(response.text)

except genai.APIError as e:
    print(f"Error calling Gemini API: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    