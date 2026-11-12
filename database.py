import os
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError

# 1. Define your Database Connection URL
# Replace with your actual credentials or use environment variables
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5433/msme_vector_db"
)

# 2. Initialize the SQLAlchemy Engine
# pool_pre_ping=True automatically tests connections before executing queries to prevent dropouts
engine = create_engine(
    DATABASE_URL, 
    pool_size=10, 
    max_overflow=20, 
    pool_pre_ping=True
)

# 3. Create a Thread-Safe Session Factory
SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine
)

# 4. Context Manager Utility for clean session management
@contextmanager
def get_db():
    """
    Provides a transactional database context. 
    Automatically commits changes if successful, rolls back on errors, 
    and closes the session when finished.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit() # Safely save additions made during the block
    except SQLAlchemyError as e:
        db.rollback() # Undo database changes if any operation fails
        print(f"❌ Database error encountered: {e}")
        raise
    finally:
        db.close() # Always return the connection to the pool

# 5. Database Table Initialization Routine (For Day 1 Setup)
def create_tables_and_extensions():
    """
    Connects to the default 'postgres' database to ensure 'msme_vector_db' exists.
    Then boots up the pgvector extension exists on your PostgreSQL instance
    and boots up your 'invoices' and 'invoice_embeddings' tables.
    """
    import psycopg2 # Use the raw driver to bypass SQLAlchemy's transaction blocks for DB creation
    from urllib.parse import urlparse
    from models import Base 
    
    # 1. Parse your connection string to extract credentials safely
    parsed_url = urlparse(DATABASE_URL)
    username = parsed_url.username or "postgres"
    password = parsed_url.password or "postgres"
    host = parsed_url.hostname or "localhost"
    port = parsed_url.port or 5432
    target_db = parsed_url.path.lstrip('/') # e.g., 'msme_vector_db'

    print(f"Verifying existence of target database: '{target_db}'...")
    
    # 2. Connect to the default 'postgres' database system namespace first
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=username,
            password=password,
            dbname="postgres" # Land on the default factory system database
        )
        conn.autocommit = True # PostgreSQL forbids database creation inside explicit transactions
        cursor = conn.cursor()
        
        # Check if your custom database exists
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s;", (target_db,))
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Database '{target_db}' not found. Creating it now...")
            cursor.execute(f'CREATE DATABASE "{target_db}";')
            print(f"Database '{target_db}' successfully created!")
        else:
            print(f"Database '{target_db}' already exists.")
            
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Failed to verify/provision base database layer: {e}")
        return

    # 3. Now that the database safely exists, let SQLAlchemy run its migrations
    try:
        with engine.connect() as conn:
            # Enable the vector extensions inside your newly made database namespace
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
            print("Verified/Enabled 'pgvector' extension.")
            
        # Build your invoices and vector tables automatically
        Base.metadata.create_all(bind=engine)
        print("Application structural database tables successfully provisioned!")
        
    except Exception as e:
        print(f"Structural schema migration initialization failed: {e}")


if __name__ == "__main__":
    # Running this file directly initializes your local database structures instantly
    print("Initializing MSME Database setup...")
    create_tables_and_extensions()
