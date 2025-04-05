from typing import List, Dict, Any
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
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
            self.vector_store = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=f"./chroma_db_{self.username}"
            )
            logger.info(f"Vector store initialized for user {self.username}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise VectorDBError(f"Could not initialize vector database: {str(e)}")

    def store_in_vector_db(self, chunks_with_embeddings: List[Dict[str, Any]]):
        documents = [Document(page_content=chunk["text"], metadata=chunk["metadata"]) 
                     for chunk in chunks_with_embeddings]
        filtered_documents = filter_complex_metadata(documents)
        
        if not filtered_documents:
            logger.warning("No valid documents to store after filtering")
            return
        
        embeddings = [chunk["embedding"] for chunk in chunks_with_embeddings]
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=filtered_documents,
                embedding=self.embedding_model,
                persist_directory=f"./chroma_db_{self.username}"
            )
        else:
            self.vector_store.add_documents(documents=filtered_documents, embeddings=embeddings)
        logger.info(f"Stored {len(filtered_documents)} chunks in vector DB for user {self.username}")

    def query_vector_db(self, question: str, k: int = 15, selected_books: List[str] = None) -> List[Dict[str, Any]]:
        if not self.vector_store:
            raise ValueError("Vector store not initialized.")
        
        question_embedding = self.embedding_model.embed_query(question)
        filter_condition = {"source": {"$in": selected_books}} if selected_books else None
        
        results = self.vector_store.similarity_search_by_vector(
            embedding=question_embedding,
            k=k
        )
        
        retrieved_chunks = [{"text": doc.page_content, "metadata": doc.metadata} for doc in results]
        logger.info(f"Query '{question}': Retrieved {len(retrieved_chunks)} chunks (k={k}, filter={filter_condition})")
        
        # Log all stored sources for debugging
        all_docs = self.vector_store.similarity_search_by_vector(question_embedding, k=100, filter=None)
        all_sources = set(doc.metadata.get("source", "Unknown") for doc in all_docs)
        logger.debug(f"All stored sources in DB: {all_sources}")
        logger.debug(f"Selected books filter: {selected_books}")
        
        for i, chunk in enumerate(retrieved_chunks):
            logger.debug(f"Chunk {i+1}: {chunk['text'][:300]}... (source: {chunk['metadata'].get('source', 'Unknown')}, pages: {chunk['metadata'].get('pages', 'Unknown')})")
        
        return retrieved_chunks