 PDF & Text Summarizer (Flask + AI)

A lightweight web app that turns long PDFs or pasted text into clear, structured summaries.
Supports two modes:

Local mode (default) → offline extractive summarization using LexRank, TextRank, and LSA.

Remote mode → OpenAI’s GPT (gpt-4o-mini) for fluent abstractive summaries, with graceful fallback to local if the API fails.

Built with Flask + Jinja2, styled with a modern dark theme, and includes copy-to-clipboard and download as PDF features.

 Features

- Upload a PDF (up to 25 MB) or paste raw text

- Cleans PDF artifacts (page numbers, footers, hyphen breaks, spacing)

- Choose summary style:

- Bullets → quick scanning

- Abstract → concise paragraph

- Study Notes → overview, key points, and recall questions

- Local extractive summarization (LexRank, TextRank, LSA with deduplication)

- Remote abstractive summarization via ChatGPT (optional)

-  Export results to PDF (server-side, ReportLab)

-  Copy summary with one click

- lassmorphism UI with responsive dark theme

 Quickstart
1) Clone and enter
git clone https://github.com/yourname/pdf-text-summarizer.git
cd pdf-text-summarizer

2) Create venv and install deps
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

3) Configure environment
cp .env.example .env
 Edit .env:
 SUMMARIZER_MODE=local        # or: remote
 OPENAI_API_KEY=sk-...        # required for remote mode
 FLASK_SECRET_KEY=yoursecret
 PORT=5002

 4) Run
python app.py
 Open http://localhost:5002


- Tech Stack

Backend: Flask, OpenAI SDK, sumy, NLTK, PyPDF, ReportLab

Frontend: Jinja2 templates, vanilla JS, CSS dark theme

Config: .env with mode + API key

- Roadmap

OCR for scanned PDFs

Multi-document summarization

Export to DOCX/Markdown

Add citations with page references
