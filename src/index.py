import os
import shutil
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils import get_config, get_logger


def create_index(config_file_path: str, log_dir_path) -> None:
    """Index a PDF into Chroma vector store.

    Returns:
        None.
    """

    # ----------------------------------------------------------------------------------------------------------
    # Basic Setup
    # ----------------------------------------------------------------------------------------------------------

    # Get Logger
    logger = get_logger(log_dir_path=log_dir_path)

    # Input PDF path
    pdf_path = get_config("paths.pdf", config_path=config_file_path)

    # Chroma persistence directory
    vector_store_path = get_config("paths.vector_store", config_path=config_file_path)

    # Chroma collection name
    collection_name = get_config("params.index.collection_name", config_path=config_file_path)

    # Chunking configuration
    chunk_size = get_config("params.index.chunking.chunk_size", config_path=config_file_path)
    chunk_overlap = get_config("params.index.chunking.chunk_overlap", config_path=config_file_path)

    # Embedding model name
    embedding_model_name = get_config("params.index.embedding_model_name", config_path=config_file_path)

    # ----------------------------------------------------------------------------------------------------------
    # Core Logic
    # ----------------------------------------------------------------------------------------------------------

    # Load PDF documents
    logger.info("load pdf: start")
    documents = load_pdf(
        pdf_path=pdf_path,
    )
    logger.info("load pdf: end")

    # Split documents into chunks
    logger.info("split documents: start")
    chunks = split_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    logger.info("split documents: end")

    # Initialize embedding model
    logger.info("create emebedding model: start")
    embedding_model = create_embedding_model(
        embedding_model_name=embedding_model_name,
    )
    logger.info("create emebedding model: end")

    # Reset existing vector database if enabled
    logger.info("reset vector database: start")
    reset_vector_database(
        vector_store_path=vector_store_path
    )
    logger.info("reset vector database: end")

    # Create Chroma vector store
    logger.info("create vector store: start")
    vector_store = create_vector_store(
        collection_name=collection_name,
        vector_store_path=vector_store_path,
        embedding_model=embedding_model,
    )
    logger.info("create vector store: end")

    # Add chunks into vector database
    logger.info("add documents to vector store: start")
    add_documents_to_vector_store(
        vector_store=vector_store,
        chunks=chunks,
    )
    logger.info("add documents to vector store: end")

    return


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------


def load_pdf(pdf_path: str) -> List[Document]:
    """Load a PDF into LangChain documents.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List[Document]: Page-level PDF documents.
    """
    # Validate file existence

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Initialize PDF loader
    loader = PyPDFLoader(pdf_path)

    # Load PDF pages as LangChain documents
    return loader.load()


def split_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Split documents into chunks.

    Args:
        documents: Documents to split.
        chunk_size: Maximum chunk size.
        chunk_overlap: Overlap between chunks.

    Returns:
        List[Document]: Chunked documents.
    """
    # Initialize recursive text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split documents into overlapping chunks
    return splitter.split_documents(documents)


def create_embedding_model(
    embedding_model_name: str,
) -> HuggingFaceEmbeddings:
    """Create a Hugging Face embedding model.

    Args:
        embedding_model_name: Hugging Face model name.

    Returns:
        HuggingFaceEmbeddings: Embedding model.
    """
    # Create embedding model with normalized embeddings
    return HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def reset_vector_database(
    vector_store_path: str
) -> None:
    """Reset vector database directory.

    Args:
        vector_store_path: Chroma persistence directory.
    """
    # Delete existing database if overwrite enabled
    shutil.rmtree(vector_store_path, ignore_errors=True)


def create_vector_store(
    collection_name: str,
    vector_store_path: str,
    embedding_model: HuggingFaceEmbeddings,
) -> Chroma:
    """Create a Chroma vector store.

    Args:
        collection_name: Chroma collection name.
        vector_store_path: Chroma persistence directory.
        embedding_model: Embedding model.

    Returns:
        Chroma: Chroma vector store.
    """
    # Initialize persistent Chroma vector store
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=vector_store_path,
    )


def add_documents_to_vector_store(
    vector_store: Chroma,
    chunks: List[Document],
) -> None:
    """Add document chunks to vector store.

    Args:
        vector_store: Chroma vector store.
        chunks: Document chunks to index.
    """
    # Generate stable document IDs
    ids = [str(i) for i in range(len(chunks))]

    # Add chunks and IDs into vector store
    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )


if __name__ == "__main__":

    # create index for the book (pdf)
    create_index(
        config_file_path="config/config.yaml",
        log_dir_path="logs"
    )

