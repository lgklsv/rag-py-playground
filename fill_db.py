import os
import time
import chromadb
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from tqdm import tqdm 

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
COLLECTION_NAME = "json_documents_collection"
BATCH_SIZE = 2500 

if not os.path.exists(DATA_PATH):
    print(f"Error: The data directory '{DATA_PATH}' does not exist.")
    print("Please create it and place your JSON files inside.")
    exit()

# --- 1. Initialize Embedding Model using Chroma's native wrapper ---
# This is the embedding model you want to use for your documents.
# Make sure it's the same one you plan to use for queries later.
try:
    embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    print("Embedding model 'all-MiniLM-L6-v2' initialized using Chroma's native function.")
except Exception as e:
    print(f"Error initializing embedding model: {e}")
    print("Please ensure 'sentence-transformers' is installed: pip install sentence-transformers")
    exit()


# --- 2. Initialize ChromaDB Client and Collection ---
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Attempt to delete the collection for a clean start if it exists
    # This is recommended after fixing a hanging upsert, as the previous state might be corrupt
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print(f"Existing collection '{COLLECTION_NAME}' deleted for a fresh start.")
    except Exception as e:
        print(f"No existing collection '{COLLECTION_NAME}' to delete or error during deletion: {e}")

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_model
    )
    print(f"ChromaDB collection '{COLLECTION_NAME}' initialized/created with embedding function.")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    exit()


# --- 3. Loading the Documents ---
print(f"Loading documents from {DATA_PATH}...")
loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.json",
    loader_cls=JSONLoader,
    loader_kwargs={"jq_schema": ".", "text_content": False},
    silent_errors=True 
)
raw_documents = loader.load()
print(f"Loaded {len(raw_documents)} raw documents.")

# --- 4. Splitting the Documents ---
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(raw_documents)
print(f"Generated {len(chunks)} chunks.")

# --- 5. Prepare Chunks for ChromaDB ---
documents = []
metadata = []
ids = []

for i, chunk in enumerate(chunks):
    documents.append(chunk.page_content)
    ids.append(f"json_doc_{i}")
    metadata.append(chunk.metadata)

print(f"Prepared {len(documents)} chunks for upsert.")

# --- 6. Batch Upsert to ChromaDB ---
total_chunks = len(documents)
print(f"Starting upsert for {total_chunks} document chunks with batching...")

for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Upserting Batches"):
    batch_docs = documents[i:i + BATCH_SIZE]
    batch_metadatas = metadata[i:i + BATCH_SIZE]
    batch_ids = ids[i:i + BATCH_SIZE]

    batch_start_time = time.time()
    try:
        collection.upsert(
            documents=batch_docs,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
    except Exception as e:
        print(f"\nError during upsert of batch starting at index {i}: {e}")
        break 

    batch_end_time = time.time()
   
    time.sleep(0.1) # Small sleep to ensure tqdm updates cleanly and reduce rapid CPU cycling if needed

print("\nAll document chunks successfully added to ChromaDB via batching.")
print(f"Total documents in collection: {collection.count()}")
