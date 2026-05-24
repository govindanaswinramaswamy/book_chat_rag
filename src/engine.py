from typing import Any, List
import regex as re

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.utils import get_config


# ----------------------------------------------------
# query engine
# ----------------------------------------------------

def query_engine(
    question: str,
    config_file_path: str,
    chat_history: List | None = None
) -> dict[str, Any]:
    """Run the end-to-end RAG query pipeline.

    Args:
        question: User input question.
        config_file_path: Path to YAML config file.
        chat_history: Optional conversation history.

    Returns:
        dict[str, Any]: Structured RAG response, retrieval metadata,
        evaluation checks, and retrieved chunks.
    """

    # -----------------------------------------
    # basic setup
    # -----------------------------------------

    # load retrieval k
    retriever_k = get_config("params.engine.retriever_k", config_path=config_file_path)

    # load reranker model name
    reranker_model_name = get_config("params.engine.reranker_model_name", config_path=config_file_path)

    # load reranker n
    reranker_n = get_config("params.engine.reranker_n", config_path=config_file_path)

    # load embedding model name
    embedding_model_name = get_config("params.index.embedding_model_name", config_path=config_file_path)

    # load vector store path
    vector_store_path = get_config("paths.vector_store", config_path=config_file_path)

    # load index collection name
    collection_name = get_config("params.index.collection_name", config_path=config_file_path)

    # load llm for generation
    llm_model_name = get_config("params.engine.llm.model_name", config_path=config_file_path)
    llm_temperature = get_config("params.engine.llm.temperature", config_path=config_file_path)
    llm = ChatOllama(model=llm_model_name, temperature=llm_temperature)

    # -----------------------------------------
    # rephrase question
    # -----------------------------------------

    formatted_chat_history = format_chat_history(chat_history)

    try:
        # create structured llm
        rephrase_llm = llm.with_structured_output(RephraseQuestion)

        rephrase_prompt = f"""
        Rewrite the latest user question into a concise,
        standalone retrieval question for a RAG system.

        Use the conversation history when needed to clarify
        references or missing context.

        Return only the rewritten question.

        Latest question:
        {question}

        Conversation history:
        {formatted_chat_history}
        """

        # call llm
        rephrased_question = rephrase_llm.invoke(rephrase_prompt)

        # extract response
        retrieval_question = rephrased_question.rephrased_question

    except Exception as e:

        retrieval_question = question

    # -----------------------------------------
    # retrieve chunks
    # -----------------------------------------

    # retrieve chunks
    chunks = retrieve_chunks(
        question=retrieval_question,
        reranker_model_name=reranker_model_name,
        embedding_model_name=embedding_model_name,
        vector_store_path=vector_store_path,
        collection_name=collection_name,
        retriever_k=retriever_k,
        reranker_n=reranker_n
    )

    # -----------------------------------------
    # generate answer
    # -----------------------------------------

    context = build_context(chunks)

    prompt = f"""
    Answer the question using ONLY the retrieved chunks.

    Question:
    {retrieval_question}

    Retrieved chunks:
    {context}

    Rules:
    - Use only information from the chunks.
    - Do not use outside knowledge.
    - Do not guess or invent information.
    - If the answer is not in the chunks, say:
      "I do not have enough relevant information in my knowledge to answer this question."
    - Every factual claim must include a citation.
    - Citation format: [chunk_id, p. page_number]

    Return ONLY valid JSON:

    {{
      "answer": "string",
      "reason": "string"
    }}

    Example:
    {{
      "answer": "The refund period is 30 days from purchase [2, p. 128].",
      "reason": "Chunk 2, page 128 explicitly states that refunds are allowed within 30 days of purchase."
    }}
    """

    try:

        # create structured llm
        answer_llm = llm.with_structured_output(RagResponse)

        # call llm
        response = answer_llm.invoke(prompt)

        # extract answer, reason
        answer = response.answer.strip()
        reason = response.reason.strip()

        # # lightweight guardrails
        # invalid_answer = (
        #     not answer
        #     or not re.search(r"\[\d+\s*,\s*p\.\s*\d+\]", answer)
        #     or "chunk" in answer.lower()
        #     or "retrieved chunk" in answer.lower()
        #     or "source excerpt" in answer.lower()
        # )
        # if invalid_answer:
        #     answer = "I could not generate a reliable answer to this question."
        #     reason = "Guardrail triggered."

        result = {
            "question": question,
            "formatted_chat_history": formatted_chat_history,
            "retrieval_question": retrieval_question,
            "chunks": chunks,
            "context": context,
            "answer": answer,
            "reason": reason
        }

    except Exception as e:

        result = {
            "question": question,
            "formatted_chat_history": formatted_chat_history,
            "retrieval_question": retrieval_question,
            "chunks": chunks,
            "context": context,
            "answer": f"Error generating answer: {e}",
            "reason": f"Error generating reason: {e}"
        }

    return result


# ----------------------------------------------------
# pydantic schema
# ----------------------------------------------------

class RagResponse(BaseModel):
    answer: str = Field(description="Grounded answer with citations")
    reason: str = Field(description="Brief internal explanation of source support")


class RephraseQuestion(BaseModel):
    rephrased_question: str = Field(description="Rephrased question")


# ----------------------------------------------------
# chat history formatter
# ----------------------------------------------------

def format_chat_history(
    chat_history: List | None
) -> str:
    """Format chat history into a readable conversation string.

    Args:
        chat_history: Optional list of chat messages.

    Returns:
        str: Formatted conversation history.
    """
    if not chat_history:
        return ""

    lines = []

    for message in chat_history:
        if message.type == "human":
            role = "User"
        elif message.type == "ai":
            role = "Assistant"
        else:
            role = message.type.capitalize()

        lines.append(f"{role}: {message.content}")

    return "\n".join(lines)


# ----------------------------------------------------
# retrieval
# ----------------------------------------------------

def retrieve_chunks(
    question: str,
    reranker_model_name: str,
    embedding_model_name: str,
    vector_store_path: str,
    collection_name: str,
    retriever_k: int,
    reranker_n: int
) -> list[dict[str, Any]]:
    """Retrieve and rerank relevant chunks from the vector store.

    Args:
        question: Retrieval question.
        reranker_model_name: Hugging Face reranker model name.
        embedding_model_name: Hugging Face embedding model name.
        vector_store_path: Path to persisted Chroma vector store.
        collection_name: Collection name.
        retriever_k: Number of chunks to retrieve before reranking.
        reranker_n: Number of chunks to keep after reranking.

    Returns:
        list[dict[str, Any]]: Retrieved and reranked chunks with
        text, metadata, page number, and rank.
    """

    # load vector store
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=vector_store_path,
        embedding_function=embedding_model,
    )

    # create retriever object
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": retriever_k,
            "filter": {
                "title": "Designing Machine Learning Systems"
            }
        }
    )

    # create reranker object
    reranker_model = HuggingFaceCrossEncoder(model_name=reranker_model_name)
    compressor = CrossEncoderReranker(
        model=reranker_model,
        top_n=reranker_n,
    )

    # create retriever + reranker object
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    # get results of retrieval + reranking
    results = compression_retriever.invoke(question)

    # extract chunks from results
    chunks = []
    for rank, doc in enumerate(results, start=1):
        chunks.append({
            "idx": rank,
            "text": doc.page_content,
            "metadata": doc.metadata,
            "page": doc.metadata["page_label"]
        })

    return chunks


# ----------------------------------------------------
# context builder
# ----------------------------------------------------

def build_context(
    chunks: list[dict[str, Any]]
) -> str:
    """Build formatted retrieval context from retrieved chunks.

    Args:
        chunks: Retrieved chunk dictionaries.

    Returns:
        str: Concatenated context string used for generation.
    """

    # build context parts using each chunk
    context_parts = []
    for chunk in chunks:
        text = (
            f"Chunk idx {chunk['idx']}\n"
            f"Page number {chunk['page']}\n\n"
            f"{chunk['text']}"
        )

        context_parts.append(text)

    # concatenate context parts
    context = "\n\n".join(context_parts)

    return context
