from langchain.document_loaders import (
    DirectoryLoader,
    UnstructuredURLLoader,
    CSVLoader,
    PDFPlumberLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from bs4 import BeautifulSoup
import requests
from typing import List, Dict
import logging
import os
from ..config.settings import (
    GOOGLE_API_KEY,
    CHROMA_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

logger = logging.getLogger(__name__)

class ContentIndexer:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
            task_type="retrieval_document"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def get_loader_for_file(self, file_path: str):
        """Get appropriate loader based on file extension"""
        ext = file_path.lower().split('.')[-1]
        return {
            'pdf': PDFPlumberLoader,
            'csv': CSVLoader,
            'txt': TextLoader,
            'pptx': UnstructuredPowerPointLoader,
            'docx': UnstructuredWordLoader,
            'xlsx': UnstructuredExcelLoader,
        }.get(ext)
        
    def index_documents(self, documents_dir: str):
        """Index documents from the specified directory with improved handling"""
        try:
            documents = []
            
            # Walk through directory and process files based on type
            for root, _, files in os.walk(documents_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    LoaderClass = self.get_loader_for_file(file_path)
                    
                    if LoaderClass:
                        try:
                            loader = LoaderClass(file_path)
                            file_docs = loader.load()
                            # Add metadata
                            for doc in file_docs:
                                doc.metadata.update({
                                    'source': file_path,
                                    'file_type': file_path.split('.')[-1],
                                    'file_name': file
                                })
                            documents.extend(file_docs)
                            logger.info(f"Successfully loaded {file}")
                        except Exception as e:
                            logger.error(f"Error loading {file}: {str(e)}")
                            continue
            
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            
            # Create and update vector store
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=self.embeddings,
                collection_name="documents"
            )
            
            # Batch process documents
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                vectorstore.add_documents(batch)
                vectorstore.persist()
                logger.info(f"Indexed batch of {len(batch)} chunks")
            
            logger.info(f"Successfully indexed {len(texts)} total document chunks")
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise
    
    def index_website(self, urls: List[str]):
        """Index content from website URLs with improved content extraction"""
        try:
            documents = []
            
            for url in urls:
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup.find_all(['script', 'style', 'nav', 'footer']):
                        element.decompose()
                    
                    # Extract content with structure
                    content_parts = []
                    
                    # Get main content areas
                    for main in soup.find_all(['main', 'article', 'div'], class_=['content', 'main-content']):
                        # Headers
                        for header in main.find_all(['h1', 'h2', 'h3']):
                            content_parts.append(f"Header: {header.get_text().strip()}")
                        
                        # Paragraphs
                        for p in main.find_all('p'):
                            content_parts.append(p.get_text().strip())
                        
                        # Lists
                        for ul in main.find_all(['ul', 'ol']):
                            items = [li.get_text().strip() for li in ul.find_all('li')]
                            content_parts.append("List items: " + " | ".join(items))
                    
                    # Join content with proper spacing
                    content = "\n\n".join(filter(None, content_parts))
                    
                    # Split into chunks
                    chunks = self.text_splitter.create_documents(
                        texts=[content],
                        metadatas=[{
                            'url': url,
                            'type': 'website'
                        }]
                    )
                    
                    documents.extend(chunks)
                    logger.info(f"Successfully processed {url}")
                    
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {str(e)}")
                    continue
            
            if documents:
                vectorstore = Chroma(
                    persist_directory=CHROMA_DB_DIR,
                    embedding_function=self.embeddings,
                    collection_name="website"
                )
                
                # Batch process documents
                batch_size = 100
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    vectorstore.add_documents(batch)
                    vectorstore.persist()
                    logger.info(f"Indexed batch of {len(batch)} chunks")
                
                logger.info(f"Successfully indexed {len(documents)} total website chunks")
            else:
                logger.warning("No content was successfully extracted from the provided URLs")
            
        except Exception as e:
            logger.error(f"Error indexing website: {str(e)}")
            raise
