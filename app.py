from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
from langchain.memory import ConversationSummaryBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import shutil
import pytesseract 
from PIL import Image
from datetime import datetime
import uuid
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import whisper
# from tiktoken import get_encoding
import tempfile

# Load environment variables
load_dotenv()

app = FastAPI()

# Configuration
UPLOAD_FOLDER = "uploads"
VECTOR_STORE_PATH = "vector_store"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

whisper_model = whisper.load_model("base")
# tokenizer = get_encoding("cl100k_base")
# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    # length_function=lambda text: len(tokenizer.encode(text)),
    add_start_index=True,
)

# Initialize vector store
vector_store = None
try:
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
except:
    vector_store = None

# Document memory to store last processed text
document_memory = {}

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM client 
client = ChatGroq(
    model_name="llama3-70b-8192",
    groq_api_key=os.environ["GROQ_API_KEY"],
    temperature=0.7
)

# Initialize conversation memory with summary capability
memory = ConversationSummaryBufferMemory(
    llm=client,
    memory_key="chat_history",
    max_token_limit=2000,  # Will summarize when this limit is reached
    return_messages=True
)

# Request models
class FileInfo(BaseModel):
    type: str  # 'image' or 'pdf'
    filename: str

class ChatRequest(BaseModel):
    message: str
    files: List[FileInfo] = []
    action: Optional[str] = None  # 'extract' or 'analyze'

# Helper functions
def generate_filename(original_filename):
    ext = original_filename.split('.')[-1]
    unique_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{unique_id}.{ext}"

def clean_extracted_text(text):
    """Clean and normalize extracted text"""
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        page_texts = []  # Store each page's text separately
        
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    cleaned_text = clean_extracted_text(page_text)
                    page_texts.append(f"Page {i+1}:\n{cleaned_text}\n")
            except Exception as page_error:
                print(f"Error processing page {i+1}: {str(page_error)}")
                page_texts.append(f"Page {i+1}: [Could not extract text]\n")
                continue
        
        # Combine all pages with clear separation
        text = "\n".join(page_texts)
        return text
    except Exception as e:
        print(f"PDF processing error: {str(e)}")
        raise Exception(f"PDF processing failed: {str(e)}")

def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return clean_extracted_text(text)
    except Exception as e:
        raise Exception(f"OCR processing failed: {str(e)}")

def update_document_memory(filename, content, file_type, file_size):
    """Update document memory with summarized content"""
    if len(content) > 2000:
        summary = client.invoke([
            {"role": "system", "content": "Summarize this document content concisely"},
            {"role": "user", "content": f"Summarize this document while preserving key information:\n{content[:5000]}"}
        ]).content
    else:
        summary = content
    
    document_memory[filename] = {
        "type": file_type,
        "text": content,
        "summary": summary,
        "size": file_size,
        "timestamp": datetime.now().isoformat()
    }
    return summary

def update_vector_store(text, filename):
    global vector_store
    if not text:
        return
    
    # Create Document objects with metadata
    documents = [Document(page_content=text, metadata={"source": filename})]
    
    # Split the text into chunks
    chunks = text_splitter.split_documents(documents)
    
    # Create new FAISS index or add to existing one
    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        # Extract page_content and metadata for add_texts
        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]
        vector_store.add_texts(texts, metadatas)
    
    # Save the updated vector store
    vector_store.save_local(VECTOR_STORE_PATH)

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix=".webm") as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp.flush()
            
            # Transcribe using Whisper 
            result = whisper_model.transcribe(tmp.name)
            
            return {"text": result["text"]}
            
    except Exception as e:
        raise HTTPException(500, f"Audio transcription failed: {str(e)}")

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Validate file type and size (10MB limit) 
        MAX_SIZE = 10 * 1024 * 1024
        file_size = 0
         
        if file.content_type.startswith('image/'):
            file_type = "image"
        elif file.filename.lower().endswith('.pdf'):
            file_type = "pdf"
        else:
            raise HTTPException(400, "Only PDF or image files allowed")

        # Generate filename
        filename = generate_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save file with size check
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(8192)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_SIZE:
                    os.remove(file_path)
                    raise HTTPException(413, "File too large (max 10MB)")
                buffer.write(chunk)
        
        # Process file
        extracted_text = ""
        try:
            if file_type == "pdf":
                extracted_text = extract_text_from_pdf(file_path)
                update_vector_store(extracted_text, filename)
            elif file_type == "image":
                extracted_text = extract_text_from_image(file_path)
                update_vector_store(extracted_text, filename)
            
            # Store the extracted text in document memory with summary
            doc_summary = update_document_memory(filename, extracted_text, file_type, file_size)
            
        except Exception as e:
            os.remove(file_path)
            raise HTTPException(500, f"Error processing file: {str(e)}")
        
        return JSONResponse({
            "status": "success",
            "file_type": file_type,
            "filename": filename,
            "preview": doc_summary[:500] + "..." if len(doc_summary) > 500 else doc_summary,
            "file_url": f"/uploads/{filename}",
            "size": file_size
        })
        
    except HTTPException:
        raise
    except Exception as e:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(500, f"Upload error: {str(e)}")

# Serve uploaded files
@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

def get_recent_chat_history(count=3):
    """Get recent chat history (last 'count' exchanges)"""
    history = memory.load_memory_variables({})["chat_history"]
    # Each exchange is a pair of human and AI messages
    return history[-(count*2):] if len(history) > (count*2) else history

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        chat_request = ChatRequest(**data)
        
        # Process files and build context
        file_contexts = []
        extracted_texts = []
        
        for file_info in chat_request.files:
            if file_info.filename in document_memory:
                file_data = document_memory[file_info.filename]
                # Use summary for context, full text for analysis
                context_text = file_data["summary"] if chat_request.action != "extract" else file_data["text"]
                extracted_texts.append(file_data["text"])
                
                if file_data["type"] == "pdf":
                    file_contexts.append(f"PDF '{file_info.filename}' summary: {context_text[:1000]}")
                elif file_data["type"] == "image":
                    file_contexts.append(f"Image content: {context_text[:1000]}")
            else:
                # Fall back to file processing if not in memory
                file_path = os.path.join(UPLOAD_FOLDER, file_info.filename)
                if os.path.exists(file_path):
                    try:
                        if file_info.type == "pdf":
                            text = extract_text_from_pdf(file_path)
                            if text:
                                extracted_texts.append(text)
                                doc_summary = update_document_memory(file_info.filename, text, "pdf", os.path.getsize(file_path))
                                file_contexts.append(f"PDF '{file_info.filename}' summary: {doc_summary[:1000]}")
                        elif file_info.type == "image":
                            text = extract_text_from_image(file_path)
                            if text:
                                extracted_texts.append(text)
                                doc_summary = update_document_memory(file_info.filename, text, "image", os.path.getsize(file_path))
                                file_contexts.append(f"Image content: {doc_summary[:1000]}")
                    except Exception as e:
                        print(f"Error processing file: {str(e)}")
                        file_contexts.append(f"Error processing {file_info.type} file")

        # Build final message with context
        final_message = chat_request.message
        if file_contexts:
            final_message = f"{final_message}\n\nDocument context:\n" + "\n".join(file_contexts)
        
        # Get relevant context from vector store
        if vector_store and final_message:
            try:
                docs = vector_store.similarity_search(final_message, k=15)
                context = "\n".join([f"From {doc.metadata['source']}:\n{doc.page_content[:1000]}" 
                                   for doc in docs])
                if context:
                    final_message = f"Relevant document excerpts:\n{context}\n\nQuestion: {final_message}"
            except Exception as e:
                print(f"Vector store error: {str(e)}")

        # Get recent conversation history (last 2-3 exchanges)
        recent_history = get_recent_chat_history(count=3)
        
        # Prepare messages for LLM
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant. Answer conversationally and concisely. "
                          "You only have to answer education related.try to answer in short as possible"
                          
            },
            *[ 
                {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
                for msg in recent_history
            ],
            {"role": "user", "content": final_message},
        ]

        # Get response from LLM
        completion = client.invoke(messages)
        ai_response = completion.content
        
        # Update memory (will automatically summarize if needed)
        memory.save_context(
            {"input": chat_request.message[:1000]},  # Truncate to prevent memory bloat
            {"output": ai_response[:1000]}
        )
        
        return {"response": ai_response}
    
    except Exception as e:
        raise HTTPException(500, f"Error processing chat request: {str(e)}")

@app.post("/clear-memory")
async def clear_memory():
    try:
        memory.clear()
        # Clear document memory
        document_memory.clear()
        # Clear vector store
        global vector_store
        vector_store = None
        if os.path.exists(VECTOR_STORE_PATH):
            shutil.rmtree(VECTOR_STORE_PATH)
            os.makedirs(VECTOR_STORE_PATH)
        # Clear uploads
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        
        return {"status": "Memory and files cleared successfully"}
    except Exception as e:
        raise HTTPException(500, f"Error clearing memory: {str(e)}")

@app.get("/document-memory")
async def get_document_memory():
    return {
        "count": len(document_memory),
        "documents": [
            {
                "filename": filename,
                "type": data["type"],
                "size": data["size"],
                "timestamp": data["timestamp"],
                "text_length": len(data["text"]),
                "summary_length": len(data.get("summary", ""))
            }
            for filename, data in document_memory.items()
        ]
    }

@app.get("/conversation-memory")
async def get_conversation_memory():
    memory_vars = memory.load_memory_variables({})
    return {
        "recent_messages": [
            {"type": "human" if msg.type == "human" else "ai", "content": msg.content}
            for msg in get_recent_chat_history(count=3)
        ],
        "summary": memory_vars.get("history", ""),
        "total_messages": len(memory_vars["chat_history"])
    }

# Serve frontend
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)