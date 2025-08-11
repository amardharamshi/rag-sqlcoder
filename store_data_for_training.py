import json
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # Or OpenAIEmbeddings
from langchain_core.documents import Document
import os

SCHEMA_JSON_FILE = "db.json"
SQL_EXAMPLES_FILE = "sql_query_examples.csv"
VECTOR_STORE_PATH = "./chroma_rag_sql_store"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Or another suitable model

def chunk_schema(schema_file):
    print(f"Loading and chunking schema from {schema_file}...")
    chunks = []
    try:
        with open(schema_file, "r") as file:
            schema_data = json.load(file)

        for table_name, table_data in schema_data.get("tables", {}).items():
            table_desc = table_data.get('description', 'N/A')
            columns_str = "\n".join([f"    - {cname}: {ctype}" for cname, ctype in table_data.get('columns', {}).items()])
            content = f"Table: {table_name}\nDescription: {table_desc}\nColumns:\n{columns_str}"
            chunks.append(Document(
                page_content=content,
                metadata={"type": "schema", "table_name": table_name}
            ))
        print(f"Created {len(chunks)} schema chunks.")
        return chunks
    except Exception as e:
        print(f"Error chunking schema: {e}")
        return []

def load_sql_examples(examples_file):
    print(f"Loading SQL examples from {examples_file}...")
    examples = []
    try:
        df = pd.read_csv(examples_file)
        for _, row in df.iterrows():
            nl_question = row['natural_language_question']
            sql_query = row['sql_query']
            examples.append(Document(
                page_content=nl_question, # Embed the question
                metadata={"type": "sql_example", "sql": sql_query}
            ))
        print(f"Loaded {len(examples)} SQL examples.")
        return examples
    except Exception as e:
        print(f"Error loading SQL examples: {e}")
        return []

def setup_vector_store():
    schema_docs = chunk_schema(SCHEMA_JSON_FILE)
    example_docs = load_sql_examples(SQL_EXAMPLES_FILE)
    all_docs = schema_docs + example_docs

    if not all_docs:
        print("No documents found to create vector store. Exiting.")
        return

    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"Creating/loading vector store at: {VECTOR_STORE_PATH}")
    vector_store = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    print("Vector store setup complete and persisted.")

setup_vector_store()