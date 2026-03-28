# 🤖 AI Document Chatbot (RAG + LLM)

Chat with your PDF documents using **AI-powered semantic search + LLM reasoning**.

This project is a **Retrieval-Augmented Generation (RAG) system** that allows users to upload one or multiple PDFs and ask intelligent questions with **context-aware responses**.

---

## 🖥 Application Preview

![App Screenshot](screenshot.png)

---

## 🚀 Features

* 📄 Upload **multiple PDF documents**
* 💬 Ask questions in natural language
* 🧠 **LLM-powered answers** (not just retrieval)
* 🔎 Semantic search using embeddings
* ⚡ FAISS vector database for fast retrieval
* 🧾 Shows **sources with page numbers**
* 💭 **Conversational memory** (remembers chat history)
* 📥 **Download chat as PDF**
* ⏱ Response time tracking
* 🎛 Interactive sidebar controls
* 🎨 Clean and modern UI

---

## 🧠 How It Works

PDF → Text Extraction → Chunking → Embeddings → FAISS → Retrieval → LLM → Final Answer

---

## ⚙️ Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* Sentence Transformers
* OpenAI (LLM)
* ReportLab

---

## 📂 Project Structure

ai-document-chatbot
│
├── app.py
├── requirements.txt
├── README.md
├── screenshot.png
│
├── src/
│   ├── pdf_loader.py
│   └── rag_pipeline.py
│
└── .streamlit/
└── config.toml

---

## ▶️ How to Run Locally

1. Clone the repository

```
git clone https://github.com/gopidesipalamanikumar/ai-document-chatbot.git
cd ai-document-chatbot
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Add your OpenAI API Key
   Create `.streamlit/secrets.toml` and add:

```
OPENAI_API_KEY = "your_api_key_here"
```

4. Run the app

```
streamlit run app.py
```

5. Open in browser
   http://localhost:8501

---

## 🌐 Live Demo

👉 *(Add your Streamlit link after deployment)*

---

## 🧠 Key Highlights

* Implemented **Retrieval-Augmented Generation (RAG)** pipeline
* Integrated **LLM for contextual answer generation**
* Designed **multi-document querying system**
* Added **conversation memory for better UX**
* Built an interactive UI with **real-time responses**

---

## 👨‍💻 Author

**Gopidesi Palamani Kumar**

* GitHub: https://github.com/gopidesipalamanikumar
* LinkedIn: https://www.linkedin.com/in/gopidesi-palamanikumar

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
