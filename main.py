from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

_ = load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



model = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
)
llm = ChatHuggingFace(llm=model)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
parser = StrOutputParser()
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
vectorstore = PineconeVectorStore(embedding=embeddings, index=index)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10},search_type="similarity")



def upsert_to_vectorstore(documents,vectorstore):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    vectorstore.add_documents(docs)




template = """
You are an AI assistant that helps peoples to know about the person named in the context. Use the context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. But you can reason to give better answers. Always answer in a concise and clear manner.
dont mention about the context in your answer. atleast don't say that "as per the context" or "according to the context" etc.
strictly remember that you are only allowed to answer questions related to the person named in the context. if the question is not related to that person, politely refuse to answer.
if the provided context is not suffiecient to answer the question, strictly, just say "I don't know".
deeply remember that don't mention about the context I provided in your answer anywhere.

{question}

Context:
{context}

Answer:
"""


prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context":retriever, "question":RunnablePassthrough()}
    | prompt
    | llm
    | parser
)




app = Flask(__name__)
CORS(app)


@app.route("/ask", methods=["POST"])
def ask_ai():
    data = request.get_json()
    question = data.get("prompt")
    if not question:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        response = chain.invoke(question)
        if "</think>" in response:
            response = response.split("</think>")[-1]
        return jsonify({"response": response.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        loader = TextLoader(filepath)
        upsert_to_vectorstore(loader.load(), vectorstore)

        return jsonify({"message": "File uploaded and indexed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(port=5000, debug=True)