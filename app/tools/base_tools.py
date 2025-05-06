from langchain.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_experimental.sql import SQLDatabaseChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.orm import Session
import google.generativeai as genai
from chromadb import PersistentClient
from typing import Optional, Type
from pydantic import BaseModel
import os
from ..config.settings import (
    GOOGLE_API_KEY,
    CHROMA_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_TEMP
)

# Configure Google API
genai.configure(api_key=GOOGLE_API_KEY)

class DocumentQueryTool(BaseTool):
    name: str = "document_query"
    description: str = "Use this tool to search through college documents for specific information. Examples: academic policies, course catalogs, faculty handbooks, etc."
    
    def __init__(self):
        super().__init__()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
            task_type="retrieval_query"
        )
        
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embeddings,
            collection_name="documents"
        )
    
    def _run(self, query: str) -> str:
        # Get relevant documents
        docs = self.vectorstore.similarity_search_with_score(
            query,
            k=3,
            score_threshold=0.7  # Only return relevant matches
        )
        
        if not docs:
            return "I couldn't find any relevant information in the documents."
        
        # Format results
        results = []
        for doc, score in docs:
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            results.append(f"Source: {source}\nRelevance: {score:.2f}\nContent: {content}\n")
        
        return "\n".join(results)

class WebsiteQueryTool(BaseTool):
    name: str = "website_query"
    description: str = "Use this tool to search through the college website content. Examples: department information, contact details, news, events, etc."
    
    def __init__(self):
        super().__init__()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
            task_type="retrieval_query"
        )
        
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embeddings,
            collection_name="website"
        )
    
    def _run(self, query: str) -> str:
        # Get relevant content
        docs = self.vectorstore.similarity_search_with_score(
            query,
            k=3,
            score_threshold=0.7  # Only return relevant matches
        )
        
        if not docs:
            return "I couldn't find any relevant information on the website."
        
        # Format results
        results = []
        for doc, score in docs:
            url = doc.metadata.get('url', 'Unknown URL')
            content = doc.page_content
            results.append(f"URL: {url}\nRelevance: {score:.2f}\nContent: {content}\n")
        
        return "\n".join(results)

class DatabaseQueryInput(BaseModel):
    query: str

class DatabaseQueryTool(BaseTool):
    name: str = "database_query"
    description: str = "Use this tool to query structured data from the college database. Examples: student records, course schedules, enrollment data, etc."
    args_schema: Type[BaseModel] = DatabaseQueryInput
    
    def __init__(self, db: Session):
        super().__init__()
        self.db = db
        
        # Initialize language model for text-to-SQL        
        self.chain = SQLDatabaseChain.from_llm(
            llm=ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=DEFAULT_TEMP
            ),
            db=db,
            verbose=True
        )
    
    def _run(self, query: str) -> str:
        try:
            # Convert natural language to SQL and execute
            result = self.chain.run(query)
            return result
        except Exception as e:
            return f"Error querying database: {str(e)}"
