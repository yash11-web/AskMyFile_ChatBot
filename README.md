# AskMyFile

AskMyFile is a simple **RAG (Retrieval-Augmented Generation) chatbot** built with **Streamlit** and **Gemini API**.  
It allows you to upload a document (PDF, Word, CSV, Excel, PPTX, JSON, or TXT) and then **ask natural language questions** about its contents.  
The app retrieves relevant parts of the document using FAISS embeddings and provides concise answers using Google Gemini.

---

##  Features
- Upload multiple file formats:
  - PDF, DOCX, TXT, CSV, XLSX, PPTX, JSON
- Preview extracted file content before querying
- Chunk text and embed using HuggingFace (`all-MiniLM-L6-v2`)
- Store and search chunks efficiently with FAISS
- Answer questions using **Gemini 2.0 Flash**
- Built with **Streamlit** for an easy-to-use web interface

---

## 🛠️ Tech Stack
- [Streamlit] - UI framework
- [LangChain] - chaining and retrieval
- [Google Gemini API] - LLM backend
- [HuggingFace Embeddings] - vector embeddings
- [FAISS] - vector database
- PyMuPDF, python-docx, python-pptx, pandas - file parsing

---


