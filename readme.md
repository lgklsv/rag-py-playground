# RAG-Py-Playground

## Prerequisites

- Python 3.11+
- [Gemini API Key](https://aistudio.google.com/app/apikey)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/lgklsv/rag-py-playground.git
   cd rag-py-playground
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Mac/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Gemini API Key:**
   - Get your Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).
   - Create a `.env` file in the project root with the following content:
     ```env
     GEMINI_API_KEY=your_api_key_here
     ```

## Running the API

Start the FastAPI server using [uvicorn](https://www.uvicorn.org/):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- The API will be available at `http://127.0.0.1:8000` by default.
- Interactive API docs: `http://127.0.0.1:8000/docs`

## API Endpoints

### `GET /health`

- Checks the health of the API and its dependencies (ChromaDB, Gemini model, embedding model).
- Returns status and collection count.

### `POST /query`

- Receives a user query and returns a generated response using Retrieval-Augmented Generation (RAG).
- **Request body:**
  ```json
  {
    "query": "Who is the artist of object 12345?"
  }
  ```
- **Response:**
  ```json
  {
    "query": "...",
    "response": "...",
    "context": ["...retrieved context..."]
  }
  ```

## Notes

- Ensure your ChromaDB collection is populated (see project scripts or instructions for indexing if needed).
- Populating the ChromaDB collection may take a while, on my Macbook Air M1 8RAM it took 1.5 hours.
- If you see a warning about an empty collection, you may need to run your data indexing script first.

---

For further details, see the code in `main.py`.
