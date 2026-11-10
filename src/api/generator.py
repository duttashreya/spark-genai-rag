from typing import Any, Dict, List


class RAGGenerator:
    """Formats grounded prompts and produces context-aware answers."""

    SYSTEM_PROMPT = (
        "You are a rigorous technical assistant. Answer the user's question ONLY using "
        "the provided context chunks. If the context does not contain the answer, "
        "state: 'I cannot answer this based on the provided context.' Do not extrapolate."
    )

    @staticmethod
    def build_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
        formatted_context = "\n\n".join(
            [
                f"[Source {idx + 1}] (Document: {c['document_id']}, Chunk: {c['chunk_index']}):\n{c['content']}"
                for idx, c in enumerate(contexts)
            ]
        )

        prompt = (
            f"{RAGGenerator.SYSTEM_PROMPT}\n\n"
            f"--- CONTEXT ---\n{formatted_context}\n\n"
            f"--- QUESTION ---\n{query}\n\n"
            f"--- ANSWER ---"
        )
        return prompt

    def generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates answer and structured citations.
        Uses extractive synthesis when external LLM is not attached.
        """
        if not contexts:
            return {
                "answer": "I cannot answer this based on the provided context.",
                "citations": [],
            }

        citations = [
            {
                "citation_id": idx + 1,
                "document_id": c["document_id"],
                "chunk_index": c["chunk_index"],
                "preview": c["content"][:120] + "...",
            }
            for idx, c in enumerate(contexts)
        ]

        # Extract answer synthesis directly grounded by top matching chunks
        primary_chunk = contexts[0]
        answer_text = (
            f"{primary_chunk['content']} "
            f"[Source 1: {primary_chunk['document_id']}]"
        )

        return {
            "answer": answer_text,
            "citations": citations,
        }