import os
from typing import List, Tuple

import yaml
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.generation.prompt_manager import PromptManager


def _load_settings() -> dict:
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)


def _format_context(chunks: List[Tuple[Document, float]]) -> str:
    """
    Format retrieved chunks into a numbered, source-labelled context block.
    The chunk number shown here maps directly to the [Source: ..., chunk #N]
    citations the LLM will produce.
    """
    parts = []
    for i, (doc, score) in enumerate(chunks, start=1):
        source    = doc.metadata.get("source_file", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        parts.append(
            f"[Chunk {i} | File: {source} | Internal index: #{chunk_idx} | Score: {score:.3f}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


class RAGGenerator:
    """Calls the Groq LLM with retrieved context and returns an answer + source list."""

    def __init__(self):
        settings          = _load_settings()
        self._prompt_mgr  = PromptManager()
        self._llm = ChatGroq(
            model       = settings["llm"]["model"],
            temperature = settings["llm"]["temperature"],
            max_tokens  = settings["llm"]["max_tokens"],
            groq_api_key= os.getenv("GROQ_API_KEY"),
        )

    def generate(
        self,
        query:  str,
        chunks: List[Tuple[Document, float]],
    ) -> dict:
        """
        Generate an answer grounded in the provided chunks.

        Returns:
            answer        — LLM response text with inline citations
            sources       — list of source dicts (file, chunk index, score, full content)
            prompt_version— version string from prompts.yaml
        """
        context = _format_context(chunks)

        messages = [
            SystemMessage(content=self._prompt_mgr.get_system_prompt()),
            HumanMessage(content=self._prompt_mgr.get_answer_prompt(
                context=context, question=query
            )),
        ]

        response = self._llm.invoke(messages)

        return {
            "answer": response.content,
            "sources": [
                {
                    "source_file":  doc.metadata.get("source_file"),
                    "chunk_index":  doc.metadata.get("chunk_index"),
                    "score":        round(score, 4),
                    "content":      doc.page_content,           # full text for RAGAS eval
                    "preview":      doc.page_content[:200],     # short preview for CLI display
                }
                for doc, score in chunks
            ],
            "prompt_version": self._prompt_mgr.version,
        }
