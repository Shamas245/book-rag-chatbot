from typing import List, Dict, Any
import logging
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

class VectorDBError(Exception):
    pass

class VectorDBManager:
    def __init__(self, embedding_model, username: str):
        self.embedding_model = embedding_model
        self.username = username
        self.vector_store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        try:
            self.vector_store = FAISS.from_texts(
                ["dummy text"], 
                self.embedding_model
            )
            logger.info(f"In-memory vector store initialized for user {self.username}")
        except Exception as e:
            logger.error(f"Vector store initialization failed: {str(e)}", exc_info=True)
            raise VectorDBError(f"Could not initialize vector database: {str(e)}")

    def store_in_vector_db(self, chunks_with_embeddings: List[Dict[str, Any]]):
        try:
            texts = [chunk["text"] for chunk in chunks_with_embeddings]
            metadatas = [chunk["metadata"] for chunk in chunks_with_embeddings]
            if texts:
                if not self.vector_store:
                    self.vector_store = FAISS.from_texts(
                        texts, 
                        self.embedding_model, 
                        metadatas=metadatas
                    )
                else:
                    self.vector_store.add_texts(texts, metadatas=metadatas)
                # Log metadata for debugging
                for i, meta in enumerate(metadatas):
                    logger.debug(f"Stored chunk {i+1} metadata: {meta}")
                logger.info(f"Stored {len(texts)} chunks in in-memory vector database for user {self.username}")
        except Exception as e:
            logger.error(f"Failed to store chunks in vector DB: {str(e)}", exc_info=True)
            raise VectorDBError(f"Could not store chunks: {str(e)}")

    def query_vector_db(self, question: str, k: int = 15, selected_books: List[str] = None) -> List[Dict[str, Any]]:
        try:
            if not self.vector_store:
                logger.warning(f"Vector store not initialized for user {self.username}")
                return []
            question_embedding = self.embedding_model.embed_query(question)
            results = self.vector_store.similarity_search_by_vector(question_embedding, k=k)
            retrieved_chunks = [{"text": doc.page_content, "metadata": doc.metadata or {}} for doc in results]
            
            # Log all retrieved chunks before filtering
            for i, chunk in enumerate(retrieved_chunks):
                logger.debug(f"Pre-filter chunk {i+1}: {chunk['metadata']}")
            
            if selected_books:
                retrieved_chunks = [
                    chunk for chunk in retrieved_chunks 
                    if chunk["metadata"].get("source", "") in selected_books  # Adjust key to match your metadata
                ]
            
            # Log filtered results
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for user {self.username} after filtering with {selected_books}")
            for i, chunk in enumerate(retrieved_chunks):
                logger.debug(f"Filtered chunk {i+1}: {chunk['metadata']}")
            return retrieved_chunks
        except Exception as e:
            logger.error(f"Vector DB query failed: {str(e)}", exc_info=True)
            raise VectorDBError(f"Could not query vector database: {str(e)}")
            
