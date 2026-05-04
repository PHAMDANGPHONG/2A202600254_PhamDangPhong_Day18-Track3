"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _get_llm_client():
    """Get LLM client — tries Google Gemini first, then OpenAI, then falls back to extractive."""
    api_key = OPENAI_API_KEY

    # Check if it's a Google API key (starts with AIza)
    if api_key and api_key.startswith("AIza"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            return "gemini", genai
        except ImportError:
            pass

    # Try OpenAI
    if api_key and api_key.startswith("sk-"):
        try:
            from openai import OpenAI
            return "openai", OpenAI()
        except ImportError:
            pass

    return "extractive", None


def _llm_generate(prompt: str, system: str = "", max_tokens: int = 200) -> str:
    """Generate text using available LLM."""
    client_type, client = _get_llm_client()

    if client_type == "gemini":
        try:
            model = client.GenerativeModel("gemini-2.0-flash")
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            response = model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            print(f"  ⚠️ Gemini error: {e}")
            return ""

    elif client_type == "openai":
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️ OpenAI error: {e}")
            return ""

    return ""


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    # Try LLM-based summarization first
    result = _llm_generate(
        prompt=text,
        system="Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. Chỉ trả về phần tóm tắt."
    )
    if result:
        return result

    # Fallback: extractive summarization (take first 2 sentences)
    sentences = text.split(". ")
    if len(sentences) >= 2:
        return ". ".join(sentences[:2]) + "."
    return text[:200]


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    result = _llm_generate(
        prompt=text,
        system=f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Mỗi câu hỏi trên 1 dòng. Chỉ trả về câu hỏi, không đánh số."
    )
    if result:
        questions = result.strip().split("\n")
        # Clean up: remove numbering and empty lines
        cleaned = [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()]
        return cleaned[:n_questions]

    # Fallback: generate basic questions from key phrases
    sentences = text.split(". ")
    questions = []
    for s in sentences[:n_questions]:
        s = s.strip()
        if s:
            questions.append(f"Thông tin gì về {s[:50]}?")
    return questions


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    result = _llm_generate(
        prompt=f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
        system="Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."
    )
    if result:
        return f"{result}\n\n{text}"

    # Fallback: simple prepend with document title
    if document_title:
        return f"Trích từ tài liệu: {document_title}.\n\n{text}"
    return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    result = _llm_generate(
        prompt=text,
        system='Trích xuất metadata từ đoạn văn. Trả về JSON hợp lệ: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi"}. Chỉ trả về JSON.'
    )
    if result:
        try:
            # Clean up potential markdown code blocks
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            pass

    # Fallback: basic metadata extraction
    metadata = {
        "topic": "",
        "entities": [],
        "category": "general",
        "language": "vi"
    }

    text_lower = text.lower()
    if any(kw in text_lower for kw in ["nghỉ phép", "nhân viên", "lương", "thưởng", "đánh giá"]):
        metadata["category"] = "hr"
    elif any(kw in text_lower for kw in ["mật khẩu", "vpn", "bảo mật", "email", "internet"]):
        metadata["category"] = "it"
    elif any(kw in text_lower for kw in ["chi phí", "thanh toán", "hóa đơn", "tài sản", "ngân sách"]):
        metadata["category"] = "finance"
    elif any(kw in text_lower for kw in ["quy định", "chính sách", "quy trình"]):
        metadata["category"] = "policy"

    return metadata


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []

    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        summary = ""
        questions = []
        enriched_text = text
        auto_meta = chunk.get("metadata", {}).copy()

        # Apply requested enrichment methods
        if "summary" in methods or "full" in methods:
            summary = summarize_chunk(text)

        if "hyqa" in methods or "full" in methods:
            questions = generate_hypothesis_questions(text)

        if "contextual" in methods or "full" in methods:
            enriched_text = contextual_prepend(text, source)

        if "metadata" in methods or "full" in methods:
            extracted_meta = extract_metadata(text)
            auto_meta = {**auto_meta, **extracted_meta}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata=auto_meta,
            method="+".join(methods),
        ))

        if (i + 1) % 5 == 0:
            print(f"    Enriched {i + 1}/{len(chunks)} chunks...")

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
