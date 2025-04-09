

# Book RAG Chatbot

A Streamlit-based Q&A system that allows users to upload PDF books, process their content, and ask questions about them using Retrieval-Augmented Generation (RAG) with Google's Gemini API and an in-memory FAISS vector database.

## Overview

This project enables users to:
- **Upload PDFs**: Extract text and images from uploaded books.
- **Ask Questions**: Query the content with natural language questions.
- **Get Answers**: Receive concise, context-aware responses powered by Gemini, with source citations.

The system processes PDFs into chunks, generates embeddings, stores them in a vector database, and retrieves relevant chunks to answer user queries. It includes user authentication and conversation history persistence.

## Features

- **PDF Processing**: Extracts text and optionally image-based text using `PyMuPDF` and Gemini Vision.
- **Vector Storage**: Stores embeddings in a local Chroma database for fast retrieval.
- **Conversational AI**: Uses Gemini 1.5 Flash for response generation with context from retrieved chunks and conversation history.
- **User Management**: Simple login/register system with JSON-based storage.
- **Streamlit UI**: Intuitive interface for uploading books, managing processed files, and chatting.

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.12
- **PDF Processing**: PyMuPDF (`fitz`), NLTK
- **Embedding & Generation**: Google Generative AI (`google-generativeai`, `langchain-google-genai`)
- **Vector Database**: FAISS (`langchain-community`) - in-memory storage
- **Authentication**: Custom JSON-based system
- **Utilities**: `python-dotenv`, `tenacity`, `PIL`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/book-rag-chatbot.git
   cd book-rag-chatbot
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**:
   - Create a `.env` file in the root directory:
     ```
     GEMINI_API_KEY_1=your_gemini_api_key_here
     ```
   - Obtain a Gemini API key from [Google AI Studio](https://makersuite.google.com/).

5. **Run the App**:
   ```bash
   streamlit run main.py
   ```
   Access it at `http://localhost:8501`.

## Usage

1. **Login/Register**: Use the sidebar to create an account or log in.
2. **Upload PDFs**: Upload one or more PDF books via the file uploader.
3. **Process Books**: Text is extracted automatically; opt-in for image text extraction if images are detected.
4. **Ask Questions**: Select books to query and type your question in the chat input.
5. **View Responses**: Answers appear with source citations (book name and pages).

## Project Structure

```
book-rag-chatbot/
├── auth.py              # User authentication logic
├── conversation.py      # Conversation management with Gemini
├── doc_process.py       # PDF processing and embedding generation
├── main.py              # Streamlit app entry point
├── utils.py             # File hashing utility
├── vector.py            # Vector database management with Chroma
├── requirements.txt     # Project dependencies
├── .env                 # Environment variables (not tracked)
├── users.json           # User data (generated)
├── processed_books_*.json  # Processed book metadata (generated)
├── conversation_history_*.json  # Conversation history (generated)
└── README.md            # This file
```

## How It Works

1. **PDF Processing (`doc_process.py`)**:
   - PDFs are split into text chunks (1500 chars) and optionally image text using Gemini Vision.
   - Embeddings are generated with Google’s `embedding-001` model.

2. **Vector Storage (`vector.py`)**:
   - Chunks and embeddings are stored in an in-memory FAISS database, keyed by username. Note: This resets on app restart.

3. **Query Handling (`conversation_manager.py`)**:
   - User questions trigger a similarity search in FAISS.
   - Retrieved chunks and conversation history form a prompt for Gemini to generate answers.

4. **UI (`main.py`)**:
   - Streamlit manages user interaction, file uploads, and chat display.

## Limitations

- **Temporary Memory**: FAISS keeps book data in the app’s memory, so it forgets everything when Streamlit Cloud restarts (e.g., after closing or refreshing). You’ll need to re-upload books each time you use the app.
- **Space Limit**: Only about 500 MB of book fingerprints fit in memory—roughly 277 books of 250 MB each—before the app runs out of room and might crash.
- **Slow for Big Books**: Processing large PDFs (like 250 MB) can take a while because it reads them one page at a time.
- **Needs Google**: The app depends on Google’s Gemini tool for understanding books and answering questions. If it’s down or you run out of “tickets” (tokens), it stops working.
- **Simple Login Safety**: Your username and password are saved in a file (`users.json`) without extra locks, so they’re not super secure.

## Potential Improvements

Here’s how to make the Book RAG Chatbot even better—faster, safer, and ready for more users and bigger books!

### 1. Scalability

#### Cloud Storage with Pinecone
- **Now**: FAISS forgets books when the app restarts because it’s just in memory.
- **Fix**: Use Pinecone, a cloud shelf that keeps book fingerprints forever.
- **Why It’s Great**: Holds tons of books (e.g., 500 GB), works for many users, and no re-uploading needed.
- **How**: Add `pinecone-client` to `requirements.txt`, tweak `vector_db.py` with a Pinecone key, and let it save your books online.

#### Real Database for Logins and Chats
- **Now**: User info and chats are in simple files (`users.json`, `conversation_history_*.json`).
- **Fix**: Switch to PostgreSQL, a sturdy storage box.
- **Why It’s Great**: Keeps everything safe, organized, and ready to grow.
- **How**: Set up a database, update `auth.py` and `main.py` with `psycopg2` to use it.

### 2. Security

#### Safer Passwords
- **Now**: Passwords are plain text in `users.json`—easy to peek at.
- **Fix**: Scramble them with `bcrypt` so they’re secret codes.
- **Why It’s Great**: Keeps your login safe from snoopers.
- **How**: Change `auth.py` to scramble passwords when you sign up.

#### Hide API Key Better
- **Now**: Your Google key is in `.env` or app memory, not locked tight.
- **Fix**: Use a secret lock (encryption) or keep it only in `.env`.
- **Why It’s Great**: Stops others from grabbing it.
- **How**: Add `cryptography` if you need to lock it up extra.

### 3. Speed

#### Faster Book Reading
- **Now**: Reads PDFs one page at a time—slow for big books.
- **Fix**: Use `asyncio` to read lots of pages together.
- **Why It’s Great**: Cuts waiting time, especially for 250 MB books.
- **How**: Update `doc_processor.py` to handle pages all at once.

#### Quick Answers
- **Now**: Asks Google every time, even for repeat questions.
- **Fix**: Save common answers in a memory trick (cache).
- **Why It’s Great**: Speeds up chats and saves “tickets” (tokens).
- **How**: Add a save-spot in `conversation_manager.py`.

### 4. Keep It Running

#### Catch Mistakes
- **Now**: Some oopsies (like missing info) stop the app.
- **Fix**: Catch them and show friendly notes instead.
- **Why It’s Great**: App stays alive, not cranky.
- **How**: Add safety nets in `main.py` to handle slip-ups.

#### Backup Helper
- **Now**: Only uses Google’s Gemini—if it’s off, you’re stuck.
- **Fix**: Add a local friend (like LLaMA) to step in.
- **Why It’s Great**: Keeps answers coming even if Google naps.
- **How**: Plug a backup into `conversation_manager.py`.

### 5. Make It Fun

#### Nicer Look
- **Now**: Plain buttons and boxes.
- **Fix**: Add progress bars, prettier layouts, and book peeks.
- **Why It’s Great**: Feels fun and easy to use.
- **How**: Sprinkle Streamlit extras in `main.py`.

#### Smarter Chats
- **Now**: Remembers just a few past chats (last 4).
- **Fix**: Keep more or sum them up.
- **Why It’s Great**: Better answers when you ask “What next?”
- **How**: Tweak `conversation_manager.py` to hold more history.

### 6. Share It

#### Go Big Online
- **Now**: Runs on your computer or Streamlit Cloud’s small space.
- **Fix**: Put it on a big server (AWS, Heroku) with a safety lock (SSL).
- **Why It’s Great**: Everyone can use it anytime, safely.
- **How**: Set up a server and add a web lock with Certbot.

#### Watch It Work
- **Now**: Messages pop up on screen only.
- **Fix**: Save them in a logbook file.
- **Why It’s Great**: Easy to spot and fix problems.
- **How**: Update all files to write logs to a file.

### Next Steps
1. **Start Here**: Lock passwords and make sure books don’t vanish mid-chat.
2. **Try It Out**: Test with big books (like 250 MB) and a few friends.
3. **Grow Up**: Add Pinecone and a database, then share it with the world!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (add one if desired).

## Acknowledgments

- Built with help from xAI’s Grok for debugging and optimization.
- Inspired by RAG techniques for knowledge-based Q&A systems.
```
