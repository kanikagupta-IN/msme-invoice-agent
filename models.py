from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from pgvector.sqlalchemy import Vector # Enables native PostgreSQL vector storage

Base = declarative_base()

class Invoice(Base):
    """
    Stores structured metadata extracted from inbound supplier invoices.
    """
    __tablename__ = 'invoices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(50), nullable=False, index=True) # Essential for multi-tenant isolation
    vendor_name = Column(String(255), nullable=False, index=True)
    invoice_number = Column(String(100), nullable=False)
    invoice_date = Column(Date, nullable=False, index=True)
    total_amount = Column(Float, nullable=False, default=0.0)
    delivery_charges = Column(Float, nullable=False, default=0.0) # Field used for tracking trends

    # Relationship to cleanly link text embeddings back to their source invoice document
    embeddings = relationship("InvoiceEmbedding", back_populates="invoice", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Invoice(vendor='{self.vendor_name}', date={self.invoice_date}, delivery=₹{self.delivery_charges})>"


class InvoiceEmbedding(Base):
    """
    Stores the unstructured raw text chunks and their corresponding mathematical vectors.
    """
    __tablename__ = 'invoice_embeddings'

    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_id = Column(Integer, ForeignKey('invoices.id', ondelete='CASCADE'), nullable=False)
    
    # Stores the raw text segment extracted from the PDF (e.g., specific terms or items)
    chunk_text = Column(Text, nullable=False)
    
    # 1536 dimensions matches OpenAI's 'text-embedding-3-small' or 'text-embedding-ada-002'
    # Change to 384 or 1024 if using local HuggingFace/Ollama embedding models (like bge-small)
    embedding = Column(Vector(768), nullable=False) 

    # Reverse relationship linkage
    invoice = relationship("Invoice", back_populates="embeddings")

    def __repr__(self):
        return f"<InvoiceEmbedding(id={self.id}, invoice_id={self.invoice_id})>"


# 🔌 Optional Database Initialisation Utility (For your Day 1 script)
def init_db(database_url="postgresql://postgres:postgres@localhost:5432/msme_vector_db"):
    """
    Connects to PostgreSQL, activates the pgvector extension, and provisions tables.
    """
    engine = create_engine(database_url)
    
    # Enable the extension in your database manually before creating tables
    with engine.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
    Base.metadata.create_all(engine)
    print("🚀 Database tables configured successfully with pgvector extension enabled.")
    
    return sessionmaker(bind=engine)
