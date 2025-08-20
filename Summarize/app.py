import io
import os
import re
import html as html_lib
from typing import List, Optional

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
from openai import OpenAI
from pypdf import PdfReader
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.DEBUG)

# Local summarization deps
import nltk
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer

# Optional segmentation helper
try:
    import wordninja as _wordninja  # type: ignore
except Exception:  # ImportError or others
    _wordninja = None


load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
app.config["UPLOAD_FOLDER"] = os.path.join(
    os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB

# Summarizer mode: "local" or "remote" (OpenAI). Default to local per request.
SUMMARIZER_MODE = os.environ.get("SUMMARIZER_MODE", "local").lower()

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"pdf", "txt"}


def _read_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts: List[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        texts.append(page_text)
    return "\n\n".join(texts)


def _clean_pdf_artifacts(text: str) -> str:
    if not text:
        return ""
    t = text
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Remove spaced-out 'Page' or page headers/footers
    t = re.sub(r"^\s*(P\s*a\s*g\s*e\b.*)$", "", t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r"^\s*Page\s*\d+\s*(of|/)\s*\d+\s*$", "", t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r"^\s*Signature\w*.*$", "", t, flags=re.MULTILINE)
    # De-hyphenate line breaks
    t = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", t)
    # Collapse single newlines within paragraphs to spaces (preserve blank lines)
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    # Normalize excessive spaces
    t = re.sub(r"\s+", " ", t)
    # Restore paragraph breaks roughly
    t = re.sub(r"(\.)\s+(?=[A-Z])", r"\1\n\n", t)
    return t.strip()


def _repair_whitespace(raw: str) -> str:
    if not raw:
        return ""
    text = raw
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"(\D)(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)(\D)", r"\1 \2", text)
    text = text.replace("/", "/ ")
    tokens = []
    for token in re.findall(r"[A-Za-z]+|[^A-Za-z]+", text):
        if (
            _wordninja is not None
            and token.isalpha()
            and token.lower() == token
            and (len(token) > 18 or re.search(r"[a-z]{12,}", token))
        ):
            tokens.extend(_wordninja.split(token))
        else:
            tokens.append(token)
    text = " ".join(tokens)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        slice_text = text[start:end]
        last_period = slice_text.rfind(". ")
        if last_period != -1 and end != length and last_period > int(0.6 * len(slice_text)):
            end = start + last_period + 1
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file.")
    return OpenAI(api_key=api_key)


def _summarize_chunk_remote(client: OpenAI, text: str, model: str = "gpt-4o-mini") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a world-class summarizer. Write concise, factual summaries using clear headings and bullet points where helpful."},
            {"role": "user", "content": f"Summarize the following text. Preserve key facts, entities, data, and definitions.\n\n{text}"},
        ],
        temperature=0.3,
        max_tokens=700,
    )
    return (response.choices[0].message.content or "").strip()


def _summarize_text_remote(text: str) -> str:
    client = _get_openai_client()
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= 6000:
        return _summarize_chunk_remote(client, text)
    chunks = _chunk_text(text, max_chars=6000)
    intermediate_summaries = [_summarize_chunk_remote(
        client, chunk) for chunk in chunks]
    combined = "\n\n".join(intermediate_summaries)
    final_prompt = (
        "Combine the following partial summaries into a single, coherent summary. "
        "Remove duplicates, maintain structure with clear headings, and include 5-10 key bullet points at the end.\n\n" + combined
    )
    return _summarize_chunk_remote(client, final_prompt)


def _jaccard(a: str, b: str) -> float:
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _summarize_text_local(text: str, max_sentences: Optional[int] = None) -> str:
    # Legacy simple LexRank path retained for safety but unused by enhanced path
    _ensure_nltk()
    text = (text or "").strip()
    if not text:
        return ""
    if max_sentences is None:
        approx = max(3, min(12, len(text) // 350))
        max_sentences = approx
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    sentences = summarizer(parser.document, max_sentences)
    return "\n".join(str(s) for s in sentences)


def _summarize_text_local_enhanced(text: str) -> str:
    _ensure_nltk()
    normalized = (text or "").strip()
    if not normalized:
        return ""
    parser = PlaintextParser.from_string(normalized, Tokenizer("english"))

    # Budget sentences by length, slightly higher than legacy for clarity
    budget = max(5, min(15, len(normalized) // 300))

    candidates: List[str] = []
    for SummCls in (LexRankSummarizer, TextRankSummarizer, LsaSummarizer):
        try:
            s = SummCls()(parser.document, budget)
            candidates.extend(str(x) for x in s)
        except Exception:
            continue

    # Deduplicate similar sentences
    unique: List[str] = []
    for s in candidates:
        if all(_jaccard(s, t) < 0.85 for t in unique):
            unique.append(s)

    # Keep order as in original text for readability
    ordering = {s: normalized.find(s) for s in unique}
    ordered = sorted(unique, key=lambda x: ordering.get(x, 10**9))[:budget]
    return "\n".join(ordered)


def _format_summary_with_style(text: str, style: str) -> str:
    _ensure_nltk()
    clean = (text or "").strip()
    if not clean:
        return ""
    try:
        sentences = nltk.sent_tokenize(clean)
    except Exception:
        sentences = [s.strip() for s in clean.split(".") if s.strip()]

    if style == "study":
        key_points_items = "".join(
            f"<li>{s}</li>" for s in sentences[: max(3, min(8, len(sentences)))]
        )
        # Rotate varied question templates
        templates = [
            "Why is {} important?",
            "What problem does {} address?",
            "How does {} work in this context?",
            "What are the key assumptions behind {}?",
            "What are the potential risks or limitations of {}?",
            "Where is {} applied effectively?",
            "How could {} be improved?",
        ]
        q_items = []
        max_q = max(3, min(5, len(sentences)))
        for i, s in enumerate(sentences[:max_q]):
            snippet = s.strip()
            snippet = snippet.replace("\n", " ").strip()
            snippet = snippet[:90].rstrip(" .,:;!?")
            q_items.append(f"<li>{templates[i % len(templates)].format(snippet)}</li>")
        questions_items = "".join(q_items)
        body = (
            "<b>Study Notes</b><br/><br/>"
            "<b>Overview</b><br/>"
            f"{sentences[0] if sentences else clean}<br/><br/>"
            "<b>Key Points</b>"
            f"<ul>{key_points_items}</ul>"
            "<b>Potential Questions</b>"
            f"<ul>{questions_items}</ul>"
        )
    elif style == "abstract":
        body = " ".join(sentences)
        body = body.replace("\n", "<br/>")
    else:  # bullets (default)
        body = "<ul>" + "".join(f"<li>{s}</li>" for s in sentences) + "</ul>"

    word_count = len(re.findall(r"\b\w+\b", body))
    return f"{body}<br/><br/><b>Word count:</b> {word_count}"


@app.route("/", methods=["GET"])
def landing():
    return render_template("landing.html")


@app.route("/app", methods=["GET"])
def app_page():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    input_text = (request.form.get("text") or "").strip()
    uploaded_file = request.files.get("file")
    style = (request.form.get("format") or "bullets").strip().lower()

    if not input_text and (not uploaded_file or uploaded_file.filename == ""):
        flash("Please provide a PDF or paste some text to summarize.")
        return redirect(url_for("app_page"))

    text_to_summarize = input_text

    is_pdf = False
    if uploaded_file and uploaded_file.filename:
        filename = secure_filename(uploaded_file.filename)
        if not _allowed_file(filename):
            flash("Only PDF or TXT files are supported.")
            return redirect(url_for("app_page"))
        file_ext = filename.rsplit(".", 1)[1].lower()
        file_bytes = uploaded_file.read()
        if file_ext == "pdf":
            is_pdf = True
            try:
                text_to_summarize = _read_pdf_text(file_bytes)
                text_to_summarize = _clean_pdf_artifacts(text_to_summarize)
            except Exception as e:
                flash(f"Failed to read PDF: {e}")
                return redirect(url_for("app_page"))
        elif file_ext == "txt":
            text_to_summarize = file_bytes.decode("utf-8", errors="ignore")

    # Repair spacing issues from PDFs or pasted text
    text_to_summarize = _repair_whitespace(text_to_summarize)

    if not text_to_summarize.strip():
        flash("No readable text found in the provided input.")
        return redirect(url_for("app_page"))

    if SUMMARIZER_MODE == "local":
        summary = _summarize_text_local_enhanced(text_to_summarize)
    else:
        try:
            summary = _summarize_text_remote(text_to_summarize)
        except Exception:
            summary = _summarize_text_local_enhanced(text_to_summarize)
            flash("Falling back to offline summarizer.")

    summary = _format_summary_with_style(summary, style)
    return render_template("result.html", summary=summary)


def _html_to_plain_text(html_input: str) -> str:
    if not html_input:
        return ""
    text = html_lib.unescape(html_input)
    text = re.sub(r"(?is)<li\s*>\s*(.*?)\s*</li\s*>", r"• \1\n", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p\s*>", "\n\n", text)
    text = re.sub(r"(?is)<p\s*>", "", text)
    text = re.sub(r"(?is)</?b\s*>", "", text)
    text = re.sub(r"(?is)</?(ul|ol)\s*>", "", text)
    text = re.sub(r"(?is)<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@app.route("/download", methods=["POST"])
def download():
    summary = request.form.get("summary") or ""
    if not summary.strip():
        flash("Nothing to download.")
        return redirect(url_for("app_page"))

    plain_text = _html_to_plain_text(summary)
    pdf_bytes = _build_pdf(plain_text)
    return send_file(
        pdf_bytes,
        as_attachment=True,
        download_name="summary.pdf",
        mimetype="application/pdf",
    )


def _build_pdf(summary_text: str) -> io.BytesIO:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER

    left_margin = 1 * inch
    right_margin = 1 * inch
    top_margin = 1 * inch
    bottom_margin = 1 * inch
    usable_width = width - left_margin - right_margin

    text_obj = c.beginText()
    text_obj.setTextOrigin(left_margin, height - top_margin)
    text_obj.setFont("Times-Roman", 12)

    for line in summary_text.splitlines():
        if not line.strip():
            text_obj.textLine("")
            continue
        words = line.split(" ")
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if c.stringWidth(candidate, "Times-Roman", 12) <= usable_width:
                current = candidate
            else:
                text_obj.textLine(current)
                current = word
        if current:
            text_obj.textLine(current)

    c.drawText(text_obj)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=True)
