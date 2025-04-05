

# Book RAG Chatbot

A Streamlit-based Q&A system that allows users to upload PDF books, process their content, and ask questions about them using Retrieval-Augmented Generation (RAG) with Google's Gemini API and a local Chroma vector database.

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
- **Vector Database**: Chroma (`langchain-community`)
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
├── chroma_db_*          # Local Chroma database (generated)
└── README.md            # This file
```

## How It Works

1. **PDF Processing (`doc_process.py`)**:
   - PDFs are split into text chunks (1500 chars) and optionally image text using Gemini Vision.
   - Embeddings are generated with Google’s `embedding-001` model.

2. **Vector Storage (`vector.py`)**:
   - Chunks and embeddings are stored in a Chroma database, keyed by username.

3. **Query Handling (`conversation.py`)**:
   - User questions trigger a similarity search in Chroma.
   - Retrieved chunks and conversation history form a prompt for Gemini to generate answers.

4. **UI (`main.py`)**:
   - Streamlit manages user interaction, file uploads, and chat display.

## Limitations

- **Single-User Focus**: Local Chroma storage may fail with multiple concurrent users.
- **Security**: Plaintext passwords in `users.json` are insecure.
- **Performance**: Sequential PDF processing can be slow for large files.
- **API Dependency**: Relies heavily on Gemini API availability and quota.

## Potential Improvements

This section outlines enhancements to make the Book RAG Chatbot production-ready, addressing scalability, security, performance, and user experience.

### 1. Scalability

#### Cloud Vector Database (Pinecone)
- **Current**: Local Chroma stores embeddings, limiting scalability and causing file-locking issues (`PermissionError`).
- **Improvement**: Use Pinecone for cloud-based vector storage.
- **Benefits**: Scalable, multi-user support, no local file management.
- **Implementation**:
  - Replace `Chroma` with `pinecone-client` in `vector.py`.
  - Initialize Pinecone index with API key and upsert/query embeddings.

#### Database for Structured Data
- **Current**: JSON files (`users.json`, `conversation_history_*.json`) for authentication and history.
- **Improvement**: Use PostgreSQL for users and conversations.
- **Benefits**: Secure, persistent, queryable storage with ACID compliance.
- **Implementation**:
  - Set up tables for `users` (username, password_hash) and `conversations` (username, role, content, timestamp).
  - Update `auth.py` and `main.py` with `psycopg2` for DB access.

### 2. Security

#### Password Hashing
- **Current**: Plaintext passwords in `users.json`.
- **Improvement**: Hash passwords with `bcrypt` or `argon2`.
- **Benefits**: Protects user credentials from exposure.
- **Implementation**:
  - Modify `AuthManager` in `auth.py` to hash passwords on registration and verify on login.

#### API Key Security
- **Current**: API keys in `.env` or `st.session_state`, unencrypted.
- **Improvement**: Encrypt keys in storage and use environment variables exclusively.
- **Benefits**: Reduces risk of key leakage.
- **Implementation**:
  - Use `cryptography` library for encryption if stored outside `.env`.

### 3. Performance

#### Asynchronous Processing
- **Current**: Sequential PDF processing and API calls.
- **Improvement**: Use `asyncio` for batch processing and concurrent API requests.
- **Benefits**: Faster handling of large PDFs and multiple users.
- **Implementation**:
  - Refactor `process_pdf` and `process_images` in `doc_process.py` to run batches asynchronously.

#### Caching
- **Current**: No response caching.
- **Improvement**: Cache frequent queries in memory (e.g., `functools.lru_cache`) or Redis.
- **Benefits**: Reduces API calls and improves response time.
- **Implementation**:
  - Add caching layer in `conversation.py` for `generate_answer`.

### 4. Reliability

#### Robust Error Handling
- **Current**: Some exceptions (e.g., `UnboundLocalError`) crash the app.
- **Improvement**: Wrap critical sections in try-except with user-friendly messages.
- **Benefits**: Graceful degradation instead of crashes.
- **Implementation**:
  - Enhance `main.py` with comprehensive error handling.

#### API Fallback
- **Current**: Dependency on Gemini API without backup.
- **Improvement**: Add a local model (e.g., LLaMA via `langchain`) as fallback.
- **Benefits**: Maintains service during API outages.
- **Implementation**:
  - Integrate a local model in `conversation.py` with a fallback switch.

### 5. User Experience

#### Improved UI
- **Current**: Basic Streamlit interface.
- **Improvement**: Add progress indicators, better layout, and book previews.
- **Benefits**: More engaging and intuitive for users.
- **Implementation**:
  - Use Streamlit columns, expanders, and progress bars in `main.py`.

#### Conversation Context
- **Current**: Limited history (last 4 messages).
- **Improvement**: Expand context window or summarize history.
- **Benefits**: Better answers for follow-up questions.
- **Implementation**:
  - Adjust `_build_history` in `conversation.py` to include more messages or summarization.

### 6. Deployment

#### Production Deployment
- **Current**: Local Streamlit server.
- **Improvement**: Deploy on AWS EC2, Heroku, or Render with Nginx and SSL.
- **Benefits**: Public access, security, and uptime.
- **Implementation**:
  - Set up a systemd service or Docker container for persistence.
  - Use Certbot for HTTPS.

#### Monitoring
- **Current**: Logs to stdout.
- **Improvement**: Log to files or a service (e.g., ELK stack).
- **Benefits**: Easier debugging and performance tracking.
- **Implementation**:
  - Use `logging.handlers.RotatingFileHandler` in all modules.

### Next Steps for Improvements
1. **Prioritize**: Start with security (password hashing) and reliability (chunk retrieval consistency).
2. **Test**: Validate improvements locally with multiple users and large PDFs.
3. **Deploy**: Move to cloud services (Pinecone, PostgreSQL) and a production server after testing.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (add one if desired).

## Acknowledgments

- Built with help from xAI’s Grok for debugging and optimization.
- Inspired by RAG techniques for knowledge-based Q&A systems.
```