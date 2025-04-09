import streamlit as st
import os
import json
import asyncio
import logging
from doc_processor import PDFDocumentProcessor
from vector_db import VectorDBManager, VectorDBError
from conversation_manager import ConversationManager
from auth import AuthManager
from utils import compute_file_hash
import time
import shutil
from google.api_core import exceptions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    st.title("Book Q&A System")

    # Ensure username is set
    if "username" not in st.session_state:
        st.subheader("Login or Register")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if AuthManager.login(username, password):
                st.session_state.username = username
                st.success(f"Logged in as {username}")
                st.rerun()
            else:
                st.error("Invalid username or password")
        
        if st.button("Register"):
            if AuthManager.register(username, password):
                st.success(f"Registered as {username}. Please log in.")
            else:
                st.error("Username already exists")
        return
    
    username = st.session_state.username
    st.write(f"Logged in as {username}")
    
    # Initialize processor and vector_db
    if "processor" not in st.session_state:
        api_key = st.session_state.get("gemini_api_key", st.secrets.get("GOOGLE_API_KEY", None))
        try:
            processor = PDFDocumentProcessor(
                chunk_size=1500, 
                page_batch_size=20, 
                username=username, 
                api_key=api_key
            )
            st.session_state.processor = processor
            st.session_state.vector_db = VectorDBManager(processor.embedding_model, username)
        except ValueError as e:
            st.error(f"Invalid API key: {str(e)}")
            return
        except VectorDBError as e:
            st.error(f"Database connection failed: {str(e)}")
            return

    # Load conversation history
    if "messages" not in st.session_state:
        try:
            with open(f"conversation_history_{username}.json", "r") as f:
                st.session_state.messages = json.load(f)
        except FileNotFoundError:
            st.session_state.messages = []

    # Load processed books
    if "processed_books" not in st.session_state:
        try:
            with open(f"processed_books_{username}.json", "r") as f:
                st.session_state.processed_books = json.load(f)
        except FileNotFoundError:
            st.session_state.processed_books = {}

    # Sidebar settings
    st.sidebar.header(f"Settings for {username}")
    api_key_input = st.sidebar.text_input("Gemini API Key", type="password", value=st.session_state.get("gemini_api_key", ""))
    if st.sidebar.button("Save API Key"):
        if api_key_input:
            st.session_state.gemini_api_key = api_key_input
            try:
                st.session_state.processor = PDFDocumentProcessor(
                    chunk_size=1500, 
                    page_batch_size=20, 
                    username=username, 
                    api_key=api_key_input
                )
                st.session_state.vector_db = VectorDBManager(st.session_state.processor.embedding_model, username)
                st.sidebar.success("API Key updated!")
            except ValueError:
                st.sidebar.error("Invalid API key provided.")
            except VectorDBError as e:
                st.sidebar.error(f"Database connection failed: {str(e)}")
        else:
            st.sidebar.error("Please enter a valid API key.")

    # Processed Books section
    st.sidebar.subheader("Processed Books")
    if st.session_state.processed_books:
        books_to_delete = []
        for idx, (book_hash, book_name) in enumerate(st.session_state.processed_books.items(), 1):
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(f"Book {idx}: {book_name}")
            with col2:
                if st.button("Delete", key=f"delete_{book_hash}"):
                    books_to_delete.append(book_hash)
        
        for book_hash in books_to_delete:
            del st.session_state.processed_books[book_hash]
            with open(f"processed_books_{username}.json", "w") as f:
                json.dump(st.session_state.processed_books, f)
            # Reset vector store for simplicity (in-memory FAISS)
            st.session_state.vector_db = VectorDBManager(st.session_state.processor.embedding_model, username)
            st.sidebar.success("Deleted book. Re-upload books as needed.")

    # File uploader
    uploaded_files = st.file_uploader("Upload Books (PDFs)", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        total_books = len(uploaded_files)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            book_name = uploaded_file.name
            temp_pdf_path = f"temp_{username}_{book_name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            book_hash = compute_file_hash(temp_pdf_path)
            if book_hash in st.session_state.processed_books:
                st.write(f"Book '{book_name}' already processed. Skipping...")
                os.remove(temp_pdf_path)
                continue

            status_text.text(f"Processing {i+1} of {total_books}: {book_name}")
            with st.spinner(f"Processing text in {book_name}..."):
                try:
                    text_chunks, image_data = st.session_state.processor.process_pdf(temp_pdf_path)
                    text_chunks_with_embeddings = st.session_state.processor.generate_embeddings(text_chunks)
                    if not text_chunks_with_embeddings:
                        st.error(f"Failed to process '{book_name}': Network issues or invalid API key.")
                        continue
                    st.session_state.vector_db.store_in_vector_db(text_chunks_with_embeddings)
                    st.success(f"Processed {len(text_chunks_with_embeddings)} text chunks for '{book_name}'")
                except Exception as e:
                    st.error(f"Error processing '{book_name}': {str(e)}")
                    continue

            all_chunks = text_chunks_with_embeddings
            if image_data:
                st.warning(f"Detected {len(image_data)} images in '{book_name}'.")
                if st.button("Process Images", key=f"yes_images_{book_hash}"):
                    with st.spinner(f"Processing images in {book_name}..."):
                        try:
                            image_chunks = st.session_state.processor.process_images(temp_pdf_path, image_data)
                            image_chunks_with_embeddings = st.session_state.processor.generate_embeddings(image_chunks)
                            if image_chunks_with_embeddings:
                                st.session_state.vector_db.store_in_vector_db(image_chunks_with_embeddings)
                                st.success(f"Processed {len(image_chunks_with_embeddings)} image chunks for '{book_name}'")
                                all_chunks = text_chunks_with_embeddings + image_chunks_with_embeddings
                        except Exception as e:
                            st.error(f"Error processing images: {str(e)}")

            st.session_state.processed_books[book_hash] = book_name
            with open(f"processed_books_{username}.json", "w") as f:
                json.dump(st.session_state.processed_books, f)
            
            for attempt in range(5):
                try:
                    os.remove(temp_pdf_path)
                    break
                except Exception as e:
                    if attempt == 4:
                        st.warning(f"Failed to delete '{temp_pdf_path}': {e}")
                    time.sleep(1)
            
            progress_bar.progress((i + 1) / total_books)

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Book selection and chat
    book_options = list(st.session_state.processed_books.values())
    selected_books = st.multiselect("Select books to search in", book_options, default=book_options if book_options else [])

    if prompt := st.chat_input("Ask a question about the books:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                logger.info(f"Selected books: {selected_books}")
                try:
                    retrieved_chunks = st.session_state.vector_db.query_vector_db(prompt, k=10, selected_books=selected_books)
                    if not retrieved_chunks:
                        st.warning("No relevant content found in selected books for this query.")
                        answer = "The answer is not available in the provided context."
                    else:
                        async def get_answer():
                            return await ConversationManager.generate_answer(
                                st.session_state.processor.gemini_model,
                                prompt,
                                retrieved_chunks,
                                st.session_state.messages[:-1]
                            )
                        answer = asyncio.run(get_answer())
                        st.markdown(answer)
                        if answer != "The answer is not available in the provided books.":
                            sources = set()
                            for chunk in retrieved_chunks:
                                pages = chunk['metadata'].get('pages', 'Unknown')
                                sources.add(f"{chunk['metadata']['source']} (Pages: {pages})")
                            st.markdown("**Source(s):** " + ", ".join(sources))
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    logger.error(f"Chat error: {e}")
                    answer = "Sorry, I couldn’t process your request."
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with open(f"conversation_history_{username}.json", "w") as f:
                    json.dump(st.session_state.messages, f)

if __name__ == "__main__":
    main()
    
