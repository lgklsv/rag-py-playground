# RAG-Py-Playground

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.11+</li>
</ul>

<h2>Installation</h2>
<h3>1. Clone the repository:</h3>

```
git clone https://github.com/ThomasJanssen-tech/Retrieval-Augmented-Generation.git
cd Retrieval-Augmented-Generation
```

<h3>2. Create a virtual environment</h3>

```
python -m venv venv
```

<h3>3. Activate the virtual environment</h3>

```
venv\Scripts\Activate
(or on Mac): source venv/bin/activate
```

<h3>4. Install libraries</h3>

```
pip install -r requirements.txt
```

<h3>5. Add Gemini API Key</h3>
Get a Gemini API Key from here: https://aistudio.google.com/app/apikey<BR>
Add it to .env.example as GEMINI_API_KEY<BR>
Rename to .env<BR>

<h2>Executing the scripts</h2>

- Open a terminal in VS Code

- Execute the following command:

```
python fill_db.py
python ask.py
```
