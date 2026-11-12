import json
from datetime import datetime
from pydantic import BaseModel, Field
from pypdf import PdfReader

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database import get_db
from models import Invoice, InvoiceEmbedding

# 1. Define the Structured Pydantic Extraction Schema
class InvoiceExtractionSchema(BaseModel):
    """
    Defines the structural format the LLM must follow when extracting 
    invoice parameters from the raw layout text.
    """
    vendor_name: str = Field(description="The primary name of the company or supplier issuing the invoice.")
    invoice_number: str = Field(description="The unique invoice ID, invoice number, or reference bill code.")
    invoice_date: str = Field(description="The billing date stated on the invoice (preferred format: YYYY-MM-DD).")
    total_amount: float = Field(description="The total gross balance due, including taxes and adjustments.")
    delivery_charges: float = Field(description="Any freight, shipping, transport, delivery, or postal fee line items. Use 0.0 if not listed.")

# 2. Main Extraction and Pipeline class
class InvoiceIngestor:
    def __init__(self):
        # Set up local Ollama models (API costs = ₹0)
        self.llm = ChatOllama(model="llama3.2", temperature=0, format="json") # Forces structured JSON response
        self.embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        
        # Text splitter configurations for the Unstructured RAG Layer
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def extract_raw_text_from_pdf(self, pdf_path: str) -> str:
        """Reads a local PDF file and extracts text page by page."""
        reader = PdfReader(pdf_path)
        raw_text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content + "\n"
        return raw_text
        
    def extract_structured_data(self, text: str) -> InvoiceExtractionSchema:
        """Passes unstructured invoice text to local LLM and parses it into a schema."""
        # Provide an explicit, simple string pattern structure so local LLMs can easily replicate it
        prompt = f"""
        You are an expert financial parsing system. Analyze the following unstructured vendor invoice text.
        Extract the requested fields and return a single valid JSON object following this exact structure:

        {{
            "vendor_name": "Name of the supplier company",
            "invoice_number": "The invoice ID or reference number",
            "invoice_date": "YYYY-MM-DD",
            "total_amount": 0.0,
            "delivery_charges": 0.0
        }}

        Rules:
        1. If no specific delivery, freight, or shipping charge line item is mentioned, set "delivery_charges" to 0.0.
        2. Ensure the "invoice_date" matches the YYYY-MM-DD format strictly.
        3. Return ONLY the raw JSON block. Do not wrap it in markdown block quotes (```json) or include extra conversational text.

        Invoice Text Content:
        \"\"\"{text}\"\"\"
        """
        
        response = self.llm.invoke(prompt)
        
        # Clean up any unexpected markdown tags if the model injected them
        clean_content = response.content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]
        clean_content = clean_content.strip()

        # Parse the clean JSON output back into our safe Pydantic object model
        parsed_json = json.loads(clean_content)
        return InvoiceExtractionSchema.model_validate(parsed_json)

    def process_and_save_invoice(self, tenant_id: str, pdf_path: str):
        """Executes the dual ingestion pipeline: Structured data to relational SQL & Unstructured chunks to Vector store."""
        print(f"Processing document: {pdf_path}...")
        raw_text = self.extract_raw_text_from_pdf(pdf_path)
        
        # Step A: Generate Structured Data via Local LLM
        extracted = self.extract_structured_data(raw_text)
        
        # Normalise strings into correct Python date formats safely
        try:
            clean_date = datetime.strptime(extracted.invoice_date, "%Y-%m-%d").date()
        except ValueError:
            clean_date = datetime.today().date() # Safe fallback on formatting anomalies

        # Step B: Chunk raw context text for the Unstructured RAG Pipeline
        chunks = self.text_splitter.split_text(raw_text)
        
        # Context block using our database helper wrapper
        with get_db() as db:
            # 1. Write the relational invoice records into standard SQL tables
            db_invoice = Invoice(
                tenant_id=tenant_id,
                vendor_name=extracted.vendor_name,
                invoice_number=extracted.invoice_number,
                invoice_date=clean_date,
                total_amount=extracted.total_amount,
                delivery_charges=extracted.delivery_charges
            )
            db.add(db_invoice)
            db.flush() # Flush generates the auto-increment database ID needed for the foreign key reference

            # 2. Write the chunks and corresponding embeddings into pgvector vector tables
            for chunk in chunks:
                vector_embedding = self.embeddings_model.embed_query(chunk)
                db_embedding = InvoiceEmbedding(
                    invoice_id=db_invoice.id,
                    chunk_text=chunk,
                    embedding=vector_embedding
                )
                db.add(db_embedding)
            
            print(f"Success! Relational details & {len(chunks)} vectors stored for {extracted.vendor_name} (ID: {db_invoice.id}).")

if __name__ == "__main__":
    # Test execution trace block (For running standalone on Day 2)
    # Ensure you replace this with a real path to a sample supplier invoice PDF file
    sample_pdf = "sample_supplier_invoice.pdf"
    
    import os
    if os.path.exists(sample_pdf):
        ingestor = InvoiceIngestor()
        ingestor.process_and_save_invoice(tenant_id="msme_store_1", pdf_path=sample_pdf)
    else:
        print(f"Please place a valid sample invoice PDF at '{sample_pdf}' to test the standalone script.")
