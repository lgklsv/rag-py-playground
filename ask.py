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

collection = chroma_client.get_or_create_collection(name="growing_vegetables")


user_query = input("What do you want to know about growing vegetables?\n\n")

results = collection.query(
    query_texts=[user_query],
    n_results=4
)

print(results['documents'])
#print(results['metadatas'])

# client = OpenAI()
model = genai.GenerativeModel('gemini-2.0-flash-lite')

system_prompt = """
You are a helpful assistant. You answer questions about growing vegetables in Florida. 
But you only answer based on knowledge I'm providing you. You don't use your internal 
knowledge and you don't make thins up.
If you don't know the answer, just say: I don't know
--------------------
The data:
"""+str(results['documents'])+"""
"""

#print(system_prompt)

messages = [
    {"role": "user", "parts": [system_prompt]},
    {"role": "model", "parts": ["Understood. I will answer based only on the provided data."]}, # Optional: to simulate the AI acknowledging the system prompt
    {"role": "user", "parts": [user_query]}
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
