import os
import fitz
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from concurrent.futures import ThreadPoolExecutor
import nltk
from PIL import Image
import io
import logging
import threading
from google.api_core import exceptions

nltk.download('punkt')
nltk.download('punkt_tab')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PDFDocumentProcessor:
    def __init__(self, chunk_size: int = 1000, page_batch_size: int = 50, username: str = None, 
                 overlap: int = 200, max_image_size: int = 5_000_000, max_workers: int = None, 
                 api_key: str = None):
        load_dotenv()
        self.api_keys = [api_key] if api_key else [os.getenv("GEMINI_API_KEY_1"), os.getenv("GEMINI_API_KEY_2")]
        self.api_keys = [key for key in self.api_keys if key]
        if not self.api_keys:
            raise ValueError("No valid Gemini API key provided via UI or .env")
        
        self.current_api_key_index = 0
        self.current_api_key = self.api_keys[self.current_api_key_index]
        self.chunk_size = chunk_size
        self.page_batch_size = page_batch_size
        self.username = username
        self.overlap = overlap
        self.max_image_size = max_image_size
        self.max_workers = max_workers or os.cpu_count()
        
        self._lock = threading.Lock()
        genai.configure(api_key=self.current_api_key)
        self.embedding_model = None
        self.gemini_model = None
        self._initialize_models()

    def _initialize_models(self):
        try:
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.current_api_key
            )
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def process_pdf(self, pdf_path: str) -> Tuple[List[Dict[str, Any]], List[Tuple[int, Any]]]:
        """Process the PDF to extract text and image data."""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        all_text_chunks = []
        all_image_data = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for start_page in range(0, total_pages, self.page_batch_size):
                end_page = min(start_page + self.page_batch_size, total_pages)
                futures.append(executor.submit(self._process_pdf_batch, pdf_path, start_page, end_page))
            
            for future in futures:
                text_chunks, image_data = future.result()
                all_text_chunks.extend(text_chunks)
                all_image_data.extend(image_data)
        
        doc.close()
        logger.info(f"Processed {len(all_text_chunks)} text chunks and detected {len(all_image_data)} images from {pdf_path}")
        return all_text_chunks, all_image_data

    def _process_pdf_batch(self, pdf_path: str, start_page: int, end_page: int) -> Tuple[List[Dict[str, Any]], List[Tuple[int, Any]]]:
        """Process a batch of pages for text and images."""
        doc = fitz.open(pdf_path)
        text_chunks = []
        image_data = []
        text_buffer = ""
        page_refs = []
        
        for page_num in range(start_page, end_page):
            page = doc[page_num]
            page_dict = page.get_text("dict")
            page_text = ""
            if "blocks" in page_dict:
                for block in page_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            line_text = "".join([span["text"] for span in line["spans"] if "text" in span])
                            page_text += line_text + "\n"
            if not page_text.strip():
                page_text = page.get_text()
            if page_text.strip():
                text_buffer += f"[Text from Page {page_num + 1}]:\n{page_text}\n"
                logger.info(f"Extracted text from page {page_num + 1}: {len(page_text)} chars")
            else:
                logger.warning(f"No text extracted from page {page_num + 1}")
            page_refs.append(page_num + 1)
            
            image_list = page.get_images(full=True)
            filtered_images = [(page_num, img) for img in image_list if self._is_valid_image(doc, img)]
            image_data.extend(filtered_images)
            logger.info(f"Page {page_num + 1}: Found {len(filtered_images)} valid images")
            
            if len(text_buffer) >= self.chunk_size:
                chunks = self._create_text_chunks(text_buffer, os.path.basename(pdf_path), page_refs, "text")
                text_chunks.extend(chunks)
                logger.info(f"Chunked {len(chunks)} text pieces from buffer at page {page_num + 1}")
                text_buffer = ""
                page_refs = []
        
        if text_buffer.strip():
            chunks = self._create_text_chunks(text_buffer, os.path.basename(pdf_path), page_refs, "text")
            text_chunks.extend(chunks)
            logger.info(f"Final text chunking for batch {start_page}-{end_page}: {len(chunks)} chunks")
        
        doc.close()
        return text_chunks, image_data

    def _is_valid_image(self, doc: fitz.Document, img: tuple) -> bool:
        """Filter out trivial or non-visible images."""
        try:
            xref = img[0]
            base_image = doc.extract_image(xref)
            if not base_image or len(base_image["image"]) < 1000:
                return False
            return True
        except Exception:
            return False

    def _create_text_chunks(self, text: str, source: str, pages: List[int], chunk_type: str) -> List[Dict[str, Any]]:
        """Create text chunks with metadata."""
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": {"source": source, "pages": pages[:], "type": chunk_type}
                    })
                current_chunk = sentence + " "
        
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": {"source": source, "pages": pages[:], "type": chunk_type}
            })
        
        return chunks


    def process_images(self, pdf_path: str, image_data: List[Tuple[int, Any]]) -> List[Dict[str, Any]]:
        """Extract text from images using Gemini Vision."""
        doc = fitz.open(pdf_path)
        image_chunks = []
        
        for page_num, img in image_data:
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                if len(image_bytes) > self.max_image_size:
                    logger.warning(f"Image on page {page_num + 1} exceeds size limit: {len(image_bytes)} bytes")
                    continue
                
                img_pil = Image.open(io.BytesIO(image_bytes))
                with self._lock:
                    response = self.gemini_model.generate_content(["Extract text from this image:", img_pil])
                extracted_text = response.text if response.text else ""
                logger.info(f"Extracted text from image on page {page_num + 1}: {len(extracted_text)} chars")
                
                if extracted_text.strip():
                    chunks = self._create_text_chunks(
                        f"[Image Text from Page {page_num + 1}]:\n{extracted_text}",
                        os.path.basename(pdf_path),
                        [page_num + 1],
                        "image"
                    )
                    image_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process image on page {page_num + 1}: {e}")
                if "429" in str(e):
                    self._switch_api_key()
        
        doc.close()
        return image_chunks

    def _switch_api_key(self):
        """Switch to the next API key if available."""
        with self._lock:
            self.current_api_key_index = (self.current_api_key_index + 1) % len(self.api_keys)
            self.current_api_key = self.api_keys[self.current_api_key_index]
            genai.configure(api_key=self.current_api_key)
            logger.info(f"Switched to API key {self.current_api_key_index + 1}")

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for text chunks with error handling."""
        try:
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_model.embed_documents(texts)
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            return chunks
        except exceptions.GoogleAPIError as e:
            if "Quota exceeded" in str(e):
                logger.error(f"Quota exceeded for API key {self.current_api_key}: {e}")
                self._switch_api_key()
                return []  # Return empty list to avoid crash; UI will handle
            elif "503" in str(e) or "Timeout" in str(e):
                logger.error(f"Network error while generating embeddings: {e}")
                return []  # Network issue, UI will show message
            else:
                logger.error(f"API error while generating embeddings: {e}")
                return []
        except Exception as e:
            logger.error(f"Unexpected error while generating embeddings: {e}")
            return []