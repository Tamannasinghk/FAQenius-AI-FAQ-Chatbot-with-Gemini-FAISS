# 🤖 FAQenius – AI FAQ Chatbot with Gemini & FAISS

**FAQenius** is a smart AI chatbot designed to answer questions based on a custom FAQ dataset. It uses sentence embeddings and vector similarity search (FAISS) to retrieve the most relevant FAQ, and then responds with a helpful, conversational answer using **Google Gemini**.

> 💡 Simple. Fast. Friendly. Just ask and get answers based on real FAQ data.

---

## 🚀 Try It Out

👉 **Live Demo on Hugging Face Spaces**:  
[![Hugging Face Space](https://huggingface.co/spaces/tannu038/InquiroBot)

---

## 🧠 Features

- 🧾 Semantic search over FAQs using **FAISS**
- 🧠 Contextual responses powered by **Google Gemini (via LangChain)**
- 🤗 User-friendly interface with **Gradio**
- 🔍 Returns the exact FAQ context used to answer
- 💬 Handles greetings with a personal touch

---

## 🛠️ Tech Stack

| Technology           | Role                                 |
|----------------------|--------------------------------------|
| SentenceTransformers | Converts text into semantic vectors  |
| FAISS                | Fast approximate similarity search   |
| LangChain            | Manages prompt pipeline with Gemini  |
| Gemini-1.5 Flash     | Generates final conversational answer |
| Gradio               | Clean and shareable chatbot UI       |
| Pandas & NumPy       | Data handling & embedding prep       |

---

## 📁 Dataset Structure

Your `Final_faqs.csv` should include:

| Column    | Description                     |
|-----------|---------------------------------|
| Question  | User query or FAQ title         |
| Answer    | Answer to the question          |

An additional column is created internally:
- `question_plus_answer`: Combined text for embedding

---

## ⚙️ How It Works

1. User submits a query.
2. Bot generates its embedding using `all-MiniLM-L6-v2`.
3. The most relevant FAQ is retrieved using **cosine similarity** via FAISS.
4. That FAQ snippet + the query is sent to Gemini.
5. Gemini responds with a clean, human-friendly answer.
6. Bot shows both the generated answer and the retrieved FAQ.

---

## 📦 Getting Started Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Tamannasinghk/FAQenius-AI-FAQ-Chatbot-with-Gemini-FAISS.git
cd FAQenius-AI-FAQ-Chatbot-with-Gemini-FAISS
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
### 3. final overview

```bash
├── Final_faqs.csv               # Your FAQs dataset
├── app.py                       # Core chatbot logic
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```
---
