# Database
DATABASE_URL = "sqlite:///./college_db.sqlite"

# Vector store settings
CHROMA_DB_DIR = "storage/chroma"
DOCUMENTS_DIR = "storage/documents"

# Google API settings
GOOGLE_API_KEY = ""  # Add your Google API key here

# Other settings
MAX_MEMORY_WINDOW = 4000  # Maximum number of tokens to retain in memory
DEFAULT_TEMP = 0.1  # Temperature for LLM responses
CHUNK_SIZE = 1000  # Size of text chunks for indexing
CHUNK_OVERLAP = 200  # Overlap between consecutive chunks
