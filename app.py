import os
import gradio as gr
import gc
import psutil
import time
import traceback
import threading
from functools import lru_cache
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

import warnings
import logging

# Suppress warnings & logs
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*wrong pointing object.*")
warnings.filterwarnings("ignore", message=".*sqlite.*")
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.db").setLevel(logging.ERROR)


# Intalling dependecies
# !pip install langchain_groq langchain_core langchain_community
# !pip install pypdf
# !pip install chromadb
# !pip install sentence_transformers
# pip install psutil
# pip install -U langchain-groq
# !pip install gradio


# Step 1: Initialize LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key="ENTER YOUR GROQ KEY HERE",
    model_name="llama-3.3-70b-versatile"
)

# ========== DEFAULT MODEL SETUP ==========
# Define default PDF and DB paths (set these to the Grant Thornton files if you want the original fallback)
#PDF_PATH = r"C:\C Drive data\Ratnesh\Data Science\Intern Projects\3. End-to-End Generative AI Report sumarization\PDF for Extraction\AnnualReport_2025_GT.pdf"
#DB_PATH = r"./gt_report_db"

PDF_PATH = None
DB_PATH = r"./gt_report_db"   # only used if no PDF is uploaded

# Each uploaded PDF gets its own unique vector DB path, so embeddings don’t mix
def get_new_db_path(filename):
    import os, time
    base_name = os.path.splitext(os.path.basename(filename))[0]
    timestamp = int(time.time())
    return f"./db_store/{base_name}_{timestamp}"

def process_pdf(pdf_file):
    global DB_PATH

    if pdf_file is None:
        return "⚠️ No PDF uploaded."

    # Create a unique DB path for this upload
    DB_PATH = get_new_db_path(pdf_file.name)

    # Now continue with your existing embedding + vector DB creation
    # Example:
    docs = process_and_chunk_pdf(pdf_file)  # <-- your existing function
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=DB_PATH)
    vectordb.persist()
    return "✅ PDF processed successfully!"


# If you want no default fallback, set PDF_PATH = None or to a path that doesn't exist.
if PDF_PATH and not os.path.exists(DB_PATH) and os.path.exists(PDF_PATH):
    print("⏳ Creating default vector DB...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    texts = splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=DB_PATH)
    vector_db.persist()
else:
    # If DB_PATH exists, use it. If not and no PDF_PATH, still create embeddings object so code doesn't fail.
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    except Exception:
        # Graceful fallback: empty vectorstore (no default)
        vector_db = None

print("✅ Default Vector DB loaded.")

# Step 2: Set up QA Chain for default model (only if vector_db available)
if vector_db is not None:
    retriever = vector_db.as_retriever()
    prompt_template = """You are a PDF summarizer chatbot. Respond thoughtfully to the following question.

Context:
{context}

User: {question}
Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
else:
    # Minimal PROMPT if no default DB exists
    prompt_template = """You are a PDF summarizer chatbot. Respond thoughtfully to the following question.

Context:
{context}

User: {question}
Assistant:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = None


# Global state to store trained model from uploaded PDF
current_vectordb = None
current_retriever = None
current_doc_chunks = None
using_uploaded_pdf = False

# Lock & flag to avoid race conditions between processing & chat
processing_lock = threading.Lock()
processing_in_progress = False


# ========== CRITICAL ADDITION: State Clearing Function ==========
def clear_and_reset_state(remove_disk_persisted_db: bool = False):
    """
    Clear all global state variables when new PDF is uploaded or cleared.
    If remove_disk_persisted_db is True, attempt to remove persistent DB folder (if used).
    """
    global current_vectordb, current_retriever, current_doc_chunks, using_uploaded_pdf

    # Attempt to clean Chroma persistent directory if used for uploaded PDFs (we do not persist uploaded PDFs by default)
    try:
        if current_vectordb is not None:
            # if there is a persist method, flush
            try:
                current_vectordb.persist()
            except Exception:
                pass
    except Exception:
        pass

    current_vectordb = None
    current_retriever = None
    current_doc_chunks = None
    using_uploaded_pdf = False
    gc.collect()
    print("🧹 Cleared all model state")


# Memory monitoring
def monitor_memory():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB


# NOTE: Removed cached_process_pdf / process_pdf_internal usage because it referenced a non-existent function
# and lru_cache could hide processing errors. Keep processing simple and explicit.


# ========== OPTIMIZED PDF PROCESSING ==========
def process_pdf_and_create_model_optimized(pdf_file):
    """
    Processes uploaded PDF and creates an in-memory vectorstore + retriever.
    Uses a lock to avoid races with chat answering.
    """
    global current_vectordb, current_retriever, current_doc_chunks, using_uploaded_pdf, processing_in_progress

    # Acquire processing lock to avoid parallel calls
    if processing_lock.locked():
        # Another process is running; return None to indicate busy
        raise Exception("Another PDF processing is currently in progress. Please wait until it finishes.")

    with processing_lock:
        processing_in_progress = True
        try:
            initial_memory = monitor_memory()
            print(f"📊 Initial memory usage: {initial_memory:.1f} MB")

            file_size = os.path.getsize(pdf_file.name) / (1024 * 1024)  # MB
            if file_size > 40:
                processing_in_progress = False
                return None, None, None

            # Clear in-memory state to avoid mixing retrievers
            clear_and_reset_state()
            print(f"📁 Processing new PDF: {pdf_file.name}")
            gc.collect()

            loader = PyPDFLoader(pdf_file.name)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
            chunks = splitter.split_documents(documents)

            chunks = chunks[:25]  # Limit chunks for performance
            current_doc_chunks = chunks

            embeddings = HuggingFaceBgeEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            # IMPORTANT: create an in-memory Chroma store (do NOT persist to the default DB path)
            vectordb = Chroma.from_documents(chunks, embeddings)
            retriever = vectordb.as_retriever(search_kwargs={"k": 2})

            # Save to globals so chatbot uses this retriever
            current_vectordb = vectordb
            current_retriever = retriever
            using_uploaded_pdf = True

            final_memory = monitor_memory()
            print(f"✅ PDF processed: {len(chunks)} chunks, Memory: {final_memory:.1f} MB")

            processing_in_progress = False
            return vectordb, retriever, chunks

        except Exception as e:
            processing_in_progress = False
            clear_and_reset_state()
            gc.collect()
            raise Exception(f"Error processing PDF: {str(e)}")


# ========== SUMMARY GENERATION ==========
def generate_summary(pdf_file, summary_type="both", progress=gr.Progress()):
    global current_vectordb, current_retriever, current_doc_chunks, using_uploaded_pdf, processing_in_progress

    try:
        # If processing already in progress for some reason, block early
        if processing_in_progress:
            return "⚠️ A PDF is already being processed. Please wait a moment and retry."

        progress(0.1, desc="Starting PDF processing...")
        vectordb, retriever, chunks = process_pdf_and_create_model_optimized(pdf_file)

        if chunks is None:
            return "❌ File too large for fast processing. Please use a smaller PDF."

        progress(0.5, desc="Creating QA chain...")
        qachain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False
        )

        progress(0.7, desc="Generating summary...")
        total_chars = sum(len(chunk.page_content) for chunk in chunks)

        # Define prompts for both summary types
        executive_prompt = """
        Create an EXECUTIVE SUMMARY (1-2 pages) for C-level executives that includes:

        1. OVERVIEW: Brief high-level description of the document's purpose and scope
        2. KEY FINDINGS: Top 3-5 most critical insights that impact business strategy
        3. STRATEGIC IMPLICATIONS: How these findings affect business objectives and competitive position
        4. FINANCIAL IMPACT: Revenue, cost, or profitability implications (if applicable)
        5. RECOMMENDATIONS: 2-3 actionable strategic recommendations with expected outcomes
        6. NEXT STEPS: Immediate actions required from leadership

        Write in executive language, focus on business impact, avoid technical jargon, and keep each section concise but comprehensive. Target 300-500 words total.
        """

        analytical_prompt = """
        Create an ANALYTICAL SUMMARY for analysts and technical teams using bullet-point format:

        • METHODOLOGY: Data sources, analytical techniques, and sample sizes used
        • KEY METRICS: Quantitative findings with specific numbers, percentages, and statistical significance
        • TREND ANALYSIS: Patterns, correlations, and anomalies identified in the data
        • PERFORMANCE INDICATORS: KPIs, benchmarks, and performance against targets
        • TECHNICAL FINDINGS: Detailed analytical insights, model outputs, and data quality assessments
        • LIMITATIONS: Data constraints, assumptions, and potential biases
        • DETAILED RECOMMENDATIONS: Specific, measurable actions with implementation timelines
        • SUPPORTING DATA: References to charts, tables, and additional analysis needed

        Use bullet points throughout, include specific metrics and data points, and maintain technical accuracy. Target 200-400 words with quantifiable insights.
        """

        # Generate summary based on summary_type parameter
        if summary_type == "executive":
            progress(0.9, desc="Generating executive summary...")
            executive_summary = qachain.run(executive_prompt)
            progress(1.0, desc="Complete!")
            return f"""

# EXECUTIVE SUMMARY
#### (For C-Level Leadership)

{executive_summary}

---
*Document processed: {len(chunks)} text chunks, {total_chars:,} characters*
*✅ Model trained and ready for Q&A below*
            """.strip()

        elif summary_type == "analytical":
            progress(0.9, desc="Generating analytical summary...")
            analytical_summary = qachain.run(analytical_prompt)
            progress(1.0, desc="Complete!")
            return f"""

# ANALYTICAL SUMMARY  
#### (For Analysts & Technical Teams - Data-Backed Insights)

{analytical_summary}

---
*Document processed: {len(chunks)} text chunks, {total_chars:,} characters*
*✅ Model trained and ready for Q&A below*
            """.strip()

        else:  # summary_type == "both" (default)
            progress(0.8, desc="Generating executive summary...")
            executive_summary = qachain.run(executive_prompt)
            progress(0.9, desc="Generating analytical summary...")
            analytical_summary = qachain.run(analytical_prompt)

            progress(1.0, desc="Complete!")
            combined_summary = f"""

# EXECUTIVE SUMMARY
#### (For C-Level Leadership)

{executive_summary}

---

# ANALYTICAL SUMMARY  
#### (For Analysts & Technical Teams - Data-Backed Insights)

{analytical_summary}

---

*Document processed: {len(chunks)} text chunks, {total_chars:,} characters*
*✅ Model trained and ready for Q&A below*
            """

            return combined_summary.strip()

    except Exception as e:
        return f"Error processing document: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"


# Modified function call to include summary type
def generate_summary_with_type(pdf_file, summary_type_selected, progress=gr.Progress()):
    """
    Wrapper function to handle Gradio radio button selection and map to summary types.
    """
    if pdf_file is None:
        return "Please upload a PDF file first."

    # Check if file exists and is accessible
    try:
        if not os.path.exists(pdf_file.name):
            return "❌ Error: Uploaded file not found. Please try uploading again."

        file_size = os.path.getsize(pdf_file.name) / (1024 * 1024)  # Size in MB
        if file_size > 50:  # 50MB limit
            return f"❌ Error: File too large ({file_size:.1f}MB). Please use a file smaller than 50MB."

    except Exception as e:
        return f"❌ Error accessing file: {str(e)}"

    # Map radio button values to function parameters
    type_mapping = {
        "Executive Summary (for CXOs)": "executive",
        "Analytical Summary (for Analysts)": "analytical",
        "Both Summaries": "both"
    }

    selected_type = type_mapping.get(summary_type_selected, "both")
    # This will automatically clear old state and process new PDF
    return generate_summary(pdf_file, summary_type=selected_type, progress=progress)


# Timer decorator
def timed_processing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️ Processing completed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper


generate_summary_with_type = timed_processing(generate_summary_with_type)
process_pdf_and_create_model_optimized = timed_processing(process_pdf_and_create_model_optimized)


# ========== MODIFIED CHATBOT RESPONSE ==========
def chatbot_response(user_input):
    global current_retriever, current_doc_chunks, using_uploaded_pdf, processing_in_progress

    try:
        if not user_input.strip():
            return "⚠️ Please enter something."

        # If a PDF processing is underway, inform user and do not answer with stale model
        if processing_in_progress:
            return "⚠️ Currently building knowledge from your uploaded PDF — please wait a few seconds and try again."

        if using_uploaded_pdf and current_retriever is not None:
            qachain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=current_retriever,
                chain_type_kwargs={"prompt": PROMPT}
            )
            response = qachain.run(user_input)
            return response + f"\n\n---\n*✅ Using your uploaded PDF model with {len(current_doc_chunks)} chunks*"
        else:
            if qa_chain is None:
                return "⚠️ No default model available and no PDF uploaded."
            response = qa_chain.run(user_input)
            return response + "\n\n---\n*📋 Using default annual report model*"

    except Exception as e:
        traceback.print_exc()
        return f"❌ Error: {str(e)}"


# Clear state
def clear_model_state():
    clear_and_reset_state()
    return None, "", "📋 Cleared uploaded PDF. Now using default model for Q&A."


# Step 5: Gradio Interface (Professional CSS)
import gradio as gr

custom_css = """
body, .gradio-container {
    background-color: #f4f7fa !important;
    color: #30475e !important;
}

::selection {
    background-color: #0000ff !important;
    color: #fffd8c  !important;
}

#input-textbox textarea {
    background-color: #e9f5fe !important;
    color: #000151 !important;
    font-size: 16px;
    border-radius: 6px;
    border: 1px solid #bfcbd2;
    padding: 12px;
}

#output-textbox textarea {
    background-color: #e9f5fe !important;
    color: #000151  !important;
    font-weight: 500;
    border-radius: 6px;
    border: 1px solid #bfcbd2;
    padding: 12px;
}

.gr-label {
    color: #3f6071 !important;
    font-weight: 600;
}

.main-header {
    font-family: 'Roboto', 'Arial', sans-serif;
    font-weight: 600;
    font-size: 2rem;           /* Reduced font size */
    color: #1f3557 !important;
    letter-spacing: 1.5px;
    margin-bottom: 10px;
    text-align: center !important;
    text-shadow: 1.5px 1.5px 3px rgba(0,0,0,0.1);  /* Subtle shadow for depth */
}

.main-description {
    background: #fdfdfd !important;
    border-radius: 7px;
    padding: 18px 12px;
    color: #4a5a7a !important;
    font-size: 1rem;
    font-family: 'Georgia', serif;
    line-height: 1.6;
    font-style: italic;
    margin-bottom: 30px;
    max-width: 600px;
    margin-left: 0;
    margin-right: auto;
    text-align: left !important;
    white-space: nowrap;          /* Keep in one line */
    overflow: hidden;             /* Hide overflowing text */
    text-overflow: ellipsis;      /* Add ... if text is clipped */
}

.sub-header {
    font-family: 'Roboto', 'Arial', sans-serif;
    font-weight: 600;
    font-size: 1.5rem;           /* Reduced font size */
    color: #30475e !important;
    letter-spacing: 1.2px;
    margin-bottom: 14px;
    text-align: left !important;
}

.sub-description {
    background: #fdfdfd !important;
    border-radius: 5px;
    padding: 10px 12px;          /* Reduced padding */
    color: #4a5a7a !important;
    font-size: 0.9rem;           /* Reduced font size */
    font-family: 'Georgia', serif;
    line-height: 1.4;
    font-style: italic;
    margin-bottom: 20px;
}

.gr-button {
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 1.07rem !important;
    padding: 8px 18px !important;
    background: linear-gradient(90deg, #f7971e 0%, #ffd200 100%) !important;  /* orange to yellow gradient */
    color: #0053a0 !important;   /* deep blue-gray text */
    box-shadow: 0 2px 8px rgba(247, 151, 30, 0.18) !important;
    transition: background 0.3s ease, color 0.3s ease;
}

.gr-button:hover {
    background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%) !important;  /* teal to blue hover */
    color: #fff !important;
    cursor: pointer;
}
#submit-btn {
    background: #0053a0 !important;
    color: #fff !important;
}

#clear-btn {
    background: #0053a0 !important;
    color: #fff !important;
}
#generate-btn {
    background: #0053a0 !important;
    color: #fff !important;
}
#clear-chat-btn {
    background: #0053a0 !important;
    color: #fff !important;
}
/* Custom Footer Styles */
.custom-footer {
    margin-top: 40px;
    padding: 30px 20px 20px 20px;
    background: #0053a0 !important;
    color: #ffffff;
    border-radius: 12px 12px 0 0;
    box-shadow: 0 -4px 20px rgba(31, 53, 87, 0.15);
    text-align: center;
    font-family: 'Roboto', 'Arial', sans-serif;
    position: relative;
    overflow: hidden;
}

.custom-footer::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #f7971e 0%, #ffd200 50%, #43cea2 100%);
}

.footer-content {
    max-width: 800px;
    margin: 0 auto;
}

.footer-divider {
    width: 60%;
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, #ffffff40 50%, transparent 100%);
    margin: 20px auto;
    border: none;
}

.footer-name {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffd200;
    margin-bottom: 8px;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
}

.footer-project {
    font-size: 1.1rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 12px;
    font-style: italic;
}

.footer-copyright {
    font-size: 0.95rem;
    color: #b8c5d6;
    font-weight: 400;
    line-height: 1.4;
}

.footer-year {
    color: #43cea2;
    font-weight: 600;
}

/* Responsive footer */
@media (max-width: 768px) {
    .custom-footer {
        padding: 25px 15px 18px 15px;
        margin-top: 30px;
    }
    
    .footer-name {
        font-size: 1.2rem;
    }
    
    .footer-project {
        font-size: 1rem;
    }
    
    .footer-copyright {
        font-size: 0.9rem;
    }
}
"""

# Fixed Gradio Interface
with gr.Blocks(css=custom_css) as demo:

    gr.Markdown("<div class='main-header'>AI CHATBOT DASHBOARD</div>")
    gr.Markdown("<div class='main-description'>A next generation chatbot interface powered by RetrievalQA and LLaMA 3 via Groq.</div>")

    gr.Markdown("<div class='sub-header'>Annual Report (PDF File) Summarizer</div>")
    gr.Markdown("<div class='sub-description'>A GenAI-powered assistant for extracting financial insights from company annual reports or your own PDFs.</div>")
    
    #gr.Markdown("# PDF Summarizer")
    
    # Summary Type Selection and PDF Upload in PARALLEL (side by side)
    with gr.Row():
        # Left Column: Summary Type Selection
        with gr.Column(scale=2, min_width=400):
            summary_type = gr.Radio(
                choices=["Executive Summary (for CXOs)", "Analytical Summary (for Analysts)", "Both Summaries"],
                value="Both Summaries",
                label="Select Summary Type",
                info="Choose the type of summary you want to generate",
                elem_id="summary-type-radio"
            )
        
        # Right Column: PDF Upload and Action Buttons
        with gr.Column(scale=2, min_width=400):
            pdf_input = gr.File(
                file_types=[".pdf"], 
                label="Upload PDF", 
                height=80,
                elem_id="pdf-upload",
                file_count="single"  # Single file only
            )
            
            # ========== CRITICAL: Add change event handler to clear state when new file is uploaded ==========
            # Clear in-memory state immediately when a new file is selected.
            pdf_input.change(
                fn=lambda _: clear_and_reset_state(),  # Clear state immediately when new file detected
                inputs=[pdf_input],
                outputs=[]
            )
            
            # Action buttons in a row within the right column
            with gr.Row():
                generate_btn = gr.Button("Generate Summary", elem_id="generate-btn", scale=1)
                clear_btn = gr.Button("Clear PDF", elem_id="clear-btn", scale=1)
    
    # Summary Output (full width)
    summary_output = gr.Textbox(
        label="Summary", 
        lines=10,
        max_lines=25,
        show_copy_button=True,
        elem_id="summary-output"
    )
    
    # File validation function with size check
    def validate_and_process(pdf_file, summary_type_selected, progress=gr.Progress()):
        if pdf_file is None:
            return "Please upload a PDF file first."
        
        # Manual file size validation since gr.File doesn't support max_file_size
        try:
            file_size = os.path.getsize(pdf_file.name) / (1024 * 1024)  # MB
            if file_size > 40:  # 40MB limit
                return f"❌ File too large ({file_size:.1f}MB). Please use a file smaller than 40MB."
            
            if not os.path.exists(pdf_file.name):
                return "❌ Error: Uploaded file not found. Please try uploading again."
                
        except Exception as e:
            return f"❌ Error accessing file: {str(e)}"
        
        return generate_summary_with_type(pdf_file, summary_type_selected, progress)
    
    
    # Connect the generate button with both inputs (ONLY ONCE)
    generate_btn.click(
        fn=validate_and_process, 
        inputs=[pdf_input, summary_type], 
        outputs=summary_output,
        show_progress=True
    )
    
    # Modified clear function to also clear model state and update chat output
    def clear_all():
        pdf_clear, summary_clear, chat_clear = clear_model_state()
        return pdf_clear, summary_clear
    
    clear_btn.click(fn=clear_all, inputs=None, outputs=[pdf_input, summary_output])
    
    # Add information about summary types
    with gr.Accordion("Summary Type Information", open=False):
        gr.Markdown("""
        **Executive Summary (for CXOs):**
        - 300-500 words (1-2 pages)
        - Strategic focus with business impact
        - Professional tone for senior leadership
        - Structured format with key findings and recommendations
        
        **Analytical Summary (for Analysts):**  
        - Data-driven content with specific metrics
        - Bullet-point format for easy reference
        - Technical depth and risk analysis
        - Actionable insights with implementation steps
        
        **Both Summaries:**
        - Generates both executive and analytical summaries
        - Clearly separated sections for different audiences
        - Comprehensive coverage for all stakeholders
        """)
    
    # Chat Interface Section# Chat Interface Section
    gr.Markdown("<div class='sub-header'>Annual Report/ PDF File Q&A</div>")
    #gr.Markdown("# Annual Report Q&A:")
    gr.Markdown("### Ask Questions About Your Document:")
    gr.Markdown("*Upload a PDF above to train a custom model, or use the default GT Bharat annual report*")
    with gr.Row():
        with gr.Column(scale=1):
            user_input = gr.Textbox(
                lines=8,
                label="User Query",
                placeholder="Ask about financial status, future growth etc.",
                elem_id="input-textbox"
            )
        with gr.Column(scale=1):
            report_output = gr.Textbox(
                label="Report Insights",
                interactive=True,
                lines=10,
                elem_id="output-textbox"
            )

    ## Action buttons row below both textboxes
    with gr.Row():
        submit_btn = gr.Button("Submit", elem_id="submit-btn")
        clear_chat_btn = gr.Button("Clear Chat", elem_id="clear-chat-btn")  # Renamed to avoid confusion
        
        
    gr.Markdown("### Sample Questions:")    
    
    # Examples below action buttons
    examples_list = [
        "What is the main theme of the document?",
        "What are the main problems and solutions mentioned?",
        "What future actions suggested to meet the challenges?",
        "What is the trend in revenue, margins, and net profit over the past years?",        
        "How efficient is the company in using its assets and equity to generate returns (ROA, ROE)?",
        "What risks does the company face (operational, regulatory, market, financial) and how is management mitigating these?",
        "Which product lines or segments contributed most to revenue this year?"
    ]

    gr.Examples(examples=examples_list, inputs=user_input)

    # Button actions for chat functionality
    submit_btn.click(chatbot_response, user_input, report_output)
    user_input.submit(chatbot_response, user_input, report_output)

    clear_chat_btn.click(lambda: ("", ""), None, [user_input, report_output])
    
    # Custom Footer using HTML component
    import datetime
    current_year = datetime.datetime.now().year
    
    gr.HTML(f"""
    <div class="custom-footer">
        <div class="footer-content">
            <div class="footer-name">Prepared by - Ratnesh Satyarthi</div>
            <div class="footer-project">Project - AI-Powered Annual Report (PDF File) Analysis System</div>
            <hr class="footer-divider">
            <div class="footer-copyright">
                © 2025 - <span class="footer-year">{current_year}</span> Data Science Projects. All rights reserved.
            </div>
        </div>
    </div>
    """)

# For Launching on Local
if __name__ == "__main__":
    # For local development
    demo.launch(
        max_file_size="40mb",        # THIS is where max_file_size belongs
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        show_error=True
    )
