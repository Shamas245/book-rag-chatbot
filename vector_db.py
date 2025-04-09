from typing import List, Dict, Any
import logging
import os
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

class VectorDBError(Exception):
    pass

class VectorDBManager:
    def __init__(self, embedding_model, username: str):
        self.embedding_model = embedding_model
        self.username = username
        self.vector_store = None
        self.index_path = f"/mnt/faiss_index_{self.username}"  # Use /mnt for persistence on Streamlit Cloud
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        try:
            # Load existing index if it exists, otherwise create a new one
            if os.path.exists(self.index_path):
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    self.embedding_model, 
                    allow_dangerous_deserialization=True  # Required for loading pickled files
                )
            else:
                # Initialize with a dummy document to create the index
                self.vector_store = FAISS.from_texts(
                    ["dummy text"], 
                    self.embedding_model
                )
            logger.info(f"Vector store initialized for user {self.username}")
        except Exception as e:
            logger.error(f"Vector store initialization failed: {str(e)}", exc_info=True)
            raise VectorDBError(f"Could not initialize vector database: {str(e)}")

    def store_in_vector_db(self, chunks_with_embeddings: List[Dict[str, Any]]):
        try:
            texts = [chunk["text"] for chunk in chunks_with_embeddings]
            metadatas = [chunk["metadata"] for chunk in chunks_with_embeddings]
            if texts:
                if not self.vector_store:
                    self.vector_store = FAISS.from_texts(texts, self.embedding_model, metadatas=metadatas)
                else:
                    self.vector_store.add_texts(texts, metadatas=metadatas)
                # Save the index for persistence
                self.vector_store.save_local(self.index_path)
                logger.info(f"Stored {len(texts)} chunks in vector database for user {self.username}")
        except Exception as e:
            logger.error(f"Failed to store chunks in vector DB: {str(e)}", exc_info=True)
            raise VectorDBError(f"Could not store chunks: {str(e)}")

    def query_vector_db(self, question: str, k: int = 15, selected_books: List[str] = None) -> List[Dict[str, Any]]:
        try:
            if not self.vector_store:
                logger.warning(f"Vector store not initialized for user {self.username}")
                return []
            # Generate embedding for the question
            question_embedding = self.embedding_model.embed_query(question)
            # Perform similarity search
            results = self.vector_store.similarity_search_by_vector(question_embedding, k=k)
            retrieved_chunks = [{"text": doc.page_content, "metadata": doc.metadata or {}} for doc in results]
            if selected_books:
                retrieved_chunks = [
                    chunk for chunk in retrieved_chunks 
                    if chunk["metadata"].get("book_id") in selected_books
                ]
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for user {self.username}")
            return retrieved_chunks
        except Exception as e:
            logger.error(f"Vector DB query failed: {str(e)}", exc_info=True)
            raise VectorDBError(f"Could not query vector database: {str(e)}")
            
