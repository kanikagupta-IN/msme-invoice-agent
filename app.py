import os
import streamlit as st
from ingestion import InvoiceIngestor
from agent import get_invoice_agent_executor

# 1. Page Configuration & Layout Customizations
st.set_page_config(
    page_title="MSME Invoice Procurement Agent",
    page_icon="🧾",
    layout="wide"
)

# Initialize Session States to keep track of chat history and instances
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = get_invoice_agent_executor()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# For your multi-tenant setup, mock a static Workspace ID for the small business owner
TENANT_ID = "msme_retail_shop_1"

# 2. Application Header
st.title("🧾 MSME Invoice Procurement & Insight Agent")
st.caption(f"Connected Workspace: **{TENANT_ID}** | Stack: Local Ollama (Llama 3.2 + Nomic Embeddings)")

# 3. Sidebar Layout for Document Ingestion & File Uploads
with st.sidebar:
    st.header("📥 Inbound Invoice Ingestion")
    st.write("Upload your newly received supplier/wholesaler PDFs to populate your database and vector index instantly.")
    
    uploaded_files = st.file_uploader(
        "Choose supplier invoice PDFs", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process & Index Invoices", use_container_width=True):
            # Create a temporary local data folder if it doesn't exist
            os.makedirs("data/temp_invoices", exist_ok=True)
            
            # Initialize your Day 2 ingestion system class
            ingestor = InvoiceIngestor()
            
            for uploaded_file in uploaded_files:
                temp_path = os.path.join("data/temp_invoices", uploaded_file.name)
                
                # Write file safely to disk so the PDF parsers can ingest it
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner(f"Parsing structured fields & embeddings for {uploaded_file.name}..."):
                    try:
                        ingestor.process_and_save_invoice(tenant_id=TENANT_ID, pdf_path=temp_path)
                        st.success(f"Successfully indexed: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}: {e}")
                    finally:
                        # Clean up the temporary local file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
            
            st.toast("Database and RAG Vector space fully synchronized!", icon="🔄")

# 4. Main Chat Interface Component Area
st.subheader("💬 Ask Your Procurement Agent")
st.write("Run analytical cost evaluations like: *'Which vendors are charging me more for delivery this month?'*")

# Display previous messages in the chat UI container
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Accept conversational user typing inputs
if user_query := st.chat_input("Ask about supplier costs, shipping jumps, or contract fine-print..."):
    
    # Render user query right away
    with st.chat_message("user"):
        st.write(user_query)
    
    # Store query inside memory session array
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # Invoke the agent executor block with standard message constraints
    with st.chat_message("assistant"):
        with st.spinner("Agent analyzing database trends and document context..."):
            try:
                # Package LangChain requirements matching your system variables
                agent_response = st.session_state.agent_executor.invoke({
                    "tenant_id": TENANT_ID,
                    "input": user_query,
                    "chat_history": [] # Pass a list tracking conversation tokens if adding conversational memory later
                })
                
                output_text = agent_response["output"]
                st.write(output_text)
                
                # Append assistant feedback to history memory block
                st.session_state.chat_history.append({"role": "assistant", "content": output_text})
                
            except Exception as e:
                error_msg = f"An execution boundary error occurred inside the agent loop: {str(e)}"
                st.error(error_msg)
