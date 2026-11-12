import numpy as np
from sqlalchemy import text
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from database import get_db
from models import InvoiceEmbedding

# 1. Instantiate our Local AI Models
# We use llama3.2 for fast local reasoning/tool calling, and nomic for semantic indexing
llm = ChatOllama(model="llama3.2", temperature=0)
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# 2. TOOL A: Structured Business Metrics Analyzer (SQL Router)
@tool
def query_delivery_trends(tenant_id: str) -> str:
    """
    Executes a structured analytical SQL query to check month-on-month delivery charge 
    trends across all inbound supplier invoices for a specific tenant.
    Use this tool whenever the user asks about delivery costs increasing, comparing suppliers, 
    or financial metrics regarding transport expenses.
    """
    sql_query = text("""
        WITH monthly_delivery AS (
            SELECT 
                vendor_name,
                DATE_TRUNC('month', invoice_date) AS invoice_month,
                AVG(delivery_charges) AS avg_delivery
            FROM invoices
            WHERE tenant_id = :tenant_id
            GROUP BY vendor_name, DATE_TRUNC('month', invoice_date)
        )
        SELECT 
            curr.vendor_name,
            TO_CHAR(curr.invoice_month, 'YYYY-Month') AS current_month,
            curr.avg_delivery AS current_delivery_fee,
            TO_CHAR(prev.invoice_month, 'YYYY-Month') AS previous_month,
            prev.avg_delivery AS previous_delivery_fee,
            (curr.avg_delivery - prev.avg_delivery) AS cost_increase
        FROM monthly_delivery curr
        LEFT JOIN monthly_delivery prev ON 
            curr.vendor_name = prev.vendor_name AND 
            curr.invoice_month = prev.invoice_month + INTERVAL '1 month'
        ORDER BY cost_increase DESC;
    """)
    
    try:
        with get_db() as db:
            result = db.execute(sql_query, {"tenant_id": tenant_id}).fetchall()
            
            if not result:
                return "No structured delivery history or supplier invoices found in the system database for this tenant."
            
            # Format the relational SQL row results neatly for the LLM to digest
            summary = "Structured SQL Report Matrix:\n"
            for row in result:
                summary += (
                    f"- Vendor: '{row.vendor_name}' | Month: {row.current_month} | "
                    f"Current Fee: ₹{row.current_delivery_fee:.2f} | "
                    f"Prev Fee: ₹{row.previous_delivery_fee or 0.0:.2f} | "
                    f"Net Increase: ₹{row.cost_increase or 0.0:.2f}\n"
                )
            return summary
            
    except Exception as e:
        return f"Database analytical query error: {str(e)}"

# 3. TOOL B: Unstructured Fine-Print Vector Lookup (RAG Router)
@tool
def search_unstructured_invoice_context(tenant_id: str, query_text: str) -> str:
    """
    Performs a semantic similarity vector search across all text chunks extracted from uploaded invoice PDFs.
    Use this tool to find contextual text descriptions, return policies, structural footnotes, line item names, 
    or hidden details regarding why a supplier changed their delivery fee policies.
    """
    try:
        # Convert the user query text into a vector matching Nomic's 768 dimensions
        query_vector = embeddings_model.embed_query(query_text)
        
        with get_db() as db:
            # Execute a native pgvector L2 distance / Cosine distance vector search joined against invoices table
            # '<->' signifies Euclidean distance, '<=>' signifies Cosine distance in pgvector
            results = (
                db.query(InvoiceEmbedding)
                .join(InvoiceEmbedding.invoice)
                .filter(text(f"invoices.tenant_id = :tenant_id"))
                .params(tenant_id=tenant_id)
                .order_by(InvoiceEmbedding.embedding.l2_distance(query_vector))
                .limit(4)
                .all()
            )
            
            if not results:
                return "No contextual text chunks matched this search query in the vector store database."
                
            context = "Unstructured Vector RAG Text Snippets:\n"
            for match in results:
                context += f"- [From Invoice ID {match.invoice_id}]: {match.chunk_text}\n"
            return context
            
    except Exception as e:
        return f"Vector store search query error: {str(e)}"

# 4. Agent Initialization Factory Function
def get_invoice_agent_executor():
    """
    Binds the analytical tools and constructs the unified agent pipeline engine.
    """
    # Compile our tool references into an indexed list
    tools = [query_delivery_trends, search_unstructured_invoice_context]
    
    # Establish a clear prompt guiding the agent on operational constraints and tenant logic
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are a highly analytical MSME Procurement & Inbound Logistical Assistant Agent.
            Your purpose is to answer financial questions about supplier invoices for a given 'tenant_id'.
            
            Operational Workflow Rules:
            1. You must ALWAYS explicitly pass the provided 'tenant_id' to every tool function you execute. 
            2. If the user asks about costs increasing, changing, or trends, use the 'query_delivery_trends' tool first.
            3. If a trend tool highlights an anomalous increase for a vendor, immediately use the 'search_unstructured_invoice_context' tool with the vendor's name to see if you can extract context (like fuel surcharges or terms changes) written in the text footnotes.
            4. Synthesize your final answer using simple, straightforward language for the small business owner. State the exact numeric jumps clearly.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Tenant Workspace ID: {tenant_id} \nUser Question: {input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the modern LangChain function calling agent loop
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Standalone Execution Validation Test
if __name__ == "__main__":
    print("🤖 Bootstrapping test instance of MSME Invoice Agent Loop...")
    agent_engine = get_invoice_agent_executor()
    
    # Simulating a dynamic prompt query run by an owner inside tenant workspace 'msme_store_1'
    test_run = agent_engine.invoke({
        "tenant_id": "msme_store_1",
        "input": "Which vendors are charging me more for delivery this month? Tell me why if there are notes.",
        "chat_history": []
    })
    
    print("\n🎯 Final Agent Output Answer:")
    print(test_run["output"])
