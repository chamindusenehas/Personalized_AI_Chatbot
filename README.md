<h1>🧠 RAG Chatbot using LangChain, Hugging Face, and Pinecone. </h1>

This project is a Retrieval-Augmented Generation (RAG) chatbot built with Flask, LangChain, Hugging Face, and Pinecone.
It allows a user to ask natural-language questions and receive intelligent responses generated from stored knowledge.
An admin interface lets you upload text files that are automatically chunked, embedded using Hugging Face sentence transformers, and stored in a Pinecone vector database.

<h2>🚀 Features </h2>

AI-powered responses: Uses a Hugging Face large language model (DeepSeek-R1-0528) for generation.

Knowledge retrieval: Relevant information is fetched from a Pinecone vector store via semantic search.

Admin document upload: Upload text files to expand the chatbot’s knowledge base.

Automatic chunking & embedding: Text is split, embedded, and stored without manual preprocessing.

REST API architecture: Designed to integrate easily with any frontend.

Cross-origin ready: Includes CORS support for browser-based clients.

🧩 Tech Stack
* Backend Framework:	Flask
* AI Integration:	    LangChain + Hugging Face
* Vector Database:      Pinecone
* Embeddings Model:	    sentence-transformers/all-MiniLM-L6-v2
* Frontend Example:	    Vanilla HTML + JS
* Language Model:	    deepseek-ai/DeepSeek-R1-0528
