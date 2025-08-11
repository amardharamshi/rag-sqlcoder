import os
import json
import pandas as pd
from typing import TypedDict, Annotated, Sequence, List, Optional
import operator
import gc
import sqlparse
from transformers import BitsAndBytesConfig

from sqlalchemy import create_engine, text
import pymysql

from langchain_openai import ChatOpenAI 
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()

# Schema file
SCHEMA_JSON_FILE = "db.json"
VECTOR_STORE_PATH = "./chroma_rag_sql_store" # IF NOT EXISTS, RUN store_data_for_training.py TO CREATE
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_SUMMARY = "gpt-4o-2024-08-06"  # Using GPT-4o for summarization
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "")
DB_PORT = os.getenv("DB_PORT", 3306)
DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

SQLCODER_MODEL_NAME = "defog/sqlcoder-7b-2"  # CHANGE MODEL BASED ON YOUR GPU MEMORY[sqlcoder-15b, sqlcoder-34b-alpha, sqlcoder-70b-aplha]
# DEVICE = "mps" if torch.backends.mps.is_available() elif "mps" else "cpu"  
DEVICE = "cuda"
print(f"Using device: {DEVICE}")
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.1

sqlcoder_tokenizer = None
sqlcoder_model = None

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables (still needed for summarization).")
if not os.path.exists(VECTOR_STORE_PATH):
     raise FileNotFoundError(f"Vector store not found at {VECTOR_STORE_PATH}. Run the setup script first.")

# SQLCoder Model Setup Begins
print(f"Loading SQLCoder model on device: {DEVICE}")
sqlcoder_tokenizer = None
sqlcoder_model = None

def load_sqlcoder_model():
    global sqlcoder_tokenizer, sqlcoder_model
    
    if sqlcoder_tokenizer is None:
        sqlcoder_tokenizer = AutoTokenizer.from_pretrained(SQLCODER_MODEL_NAME)
    
    if sqlcoder_model is None:
        print("Loading SQLCoder with optimized 4-bit config...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        sqlcoder_model = AutoModelForCausalLM.from_pretrained(
            SQLCODER_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        sqlcoder_model.eval()  # Set to evaluation mode
    return sqlcoder_tokenizer, sqlcoder_model

def unload_sqlcoder_model():
    """Clear GPU memory after use."""
    global sqlcoder_tokenizer, sqlcoder_model
    if sqlcoder_model:
        del sqlcoder_model
        sqlcoder_model = None
    if sqlcoder_tokenizer:
        del sqlcoder_tokenizer
        sqlcoder_tokenizer = None
    torch.cuda.empty_cache()

class RAGSQLState(TypedDict):
    question: str
    db_uri: str
    retrieved_schema_context: Optional[str]
    retrieved_sql_examples: Optional[str]
    sql_query: Optional[str]
    sql_result: Optional[str]
    summary: Optional[str]
    error: Optional[str]

def format_retrieved_docs(docs: List[Document], doc_type: str) -> str:
    """Formats retrieved documents into a string for the prompt."""
    if not docs:
        return "N/A"
    if doc_type == "schema":
        return "\n\n".join([doc.page_content for doc in docs])
    elif doc_type == "sql_example":
        return "\n\n".join([
            f"Example Question: {doc.page_content}\nExample SQL: {doc.metadata.get('sql', 'N/A')}"
            for doc in docs
        ])
    return "Invalid type"

def retrieve_context_node(state: RAGSQLState) -> RAGSQLState:
    """Retrieves relevant schema and SQL examples from the vector store."""
    print("--- Node: Retrieve Context ---")
    question = state["question"]
    error_msg = None
    schema_context_str = "N/A"
    sql_examples_str = "N/A"

    try:
        print(f"Initializing embeddings and vector store from {VECTOR_STORE_PATH}...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)

        print(f"Retrieving documents for question: {question}")
        retrieved_schema_docs = vector_store.similarity_search(question, k=5, filter={'type': 'schema'})
        retrieved_example_docs = vector_store.similarity_search(question, k=3, filter={'type': 'sql_example'})

        print(f"Retrieved {len(retrieved_schema_docs)} schema chunks.")
        print(f"Retrieved {len(retrieved_example_docs)} SQL examples.")

        schema_context_str = format_retrieved_docs(retrieved_schema_docs, "schema")
        sql_examples_str = format_retrieved_docs(retrieved_example_docs, "sql_example")

    except Exception as e:
        error_msg = f"Error during context retrieval: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()

    return {
        **state,
        "retrieved_schema_context": schema_context_str,
        "retrieved_sql_examples": sql_examples_str,
        "error": error_msg
    }

def generate_sql_node(state: RAGSQLState) -> RAGSQLState:
    """Generates SQL using SQLCoder with RAG context."""
    print("--- Node: Generate SQL (SQLCoder + RAG) ---")
    question = state["question"]
    schema_context = state.get("retrieved_schema_context", "N/A")
    sql_examples = state.get("retrieved_sql_examples", "N/A")

    if state.get("error"):
        print("Skipping SQL generation due to retrieval error.")
        return state

    try:
        tokenizer, model = load_sqlcoder_model()
        
        prompt = f"""### Task
Generate a MySQL query to answer the question below by learning from the provided examples.

### Database Schema
{schema_context}

### Example SQL Queries
{sql_examples}

### Critical Instructions
1. Study the retrieved examples closely - they were selected because they're relevant to the current question
2. Adapt the query structure from examples that most closely match the current question's needs
3. Use the JOIN patterns from examples when they would provide more comprehensive information

4. Use the same table aliases as in examples where appropriate (e.g., ord, fin, ret)
5. Only use tables and columns that exist in the provided schema
6. Match the style and approach of the examples - they demonstrate best practices for this database
7. DO NOT HALLUCINATE table or column names - use only what exists in the schema

### Question
{question}

### MySQL Query
```sql
"""

        print(f"Generating SQL with SQLCoder prompt (showing first 500 chars):\n{prompt[:500]}...")

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_sql = generated_text.strip()

        if not generated_sql or not any(
            kw in generated_sql.upper() 
            for kw in ["SELECT", "FROM", "WHERE", "JOIN"]
        ):
            lines = [line for line in generated_text.split("\n") 
                    if not line.startswith("#") and line.strip()]
            generated_sql = lines[0] if lines else ""
            
            if not generated_sql:
                return {**state, "error": "Model returned empty output"}

        print(f"Generated SQL:\n{generated_sql}")
        
        unload_sqlcoder_model()
        
        return {**state, "sql_query": generated_sql, "error": None}

    except Exception as e:
        error_msg = f"Error during SQL generation with SQLCoder: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        unload_sqlcoder_model() 
        return {**state, "error": error_msg, "sql_query": None}
    finally:
        unload_sqlcoder_model()
        gc.collect()

def validate_sql_node(state: RAGSQLState) -> RAGSQLState:
    """Validates the generated SQL for syntax and safety rules."""
    print("--- Node: Validate SQL ---")
    sql_query = state.get("sql_query")
    error_msg = state.get("error") 

    if error_msg:
        print("Skipping validation due to previous error.")
        return state
    if not sql_query:
        print("No SQL query to validate.")
        return {**state, "error": "No SQL query was generated for validation."}

    print(f"Validating SQL: {sql_query}")

    # 1. Syntax Check using sqlparse
    try:
        parsed = sqlparse.parse(sql_query)
        if not parsed or (hasattr(parsed[0], 'tokens') and any(t.ttype is sqlparse.tokens.Error for t in parsed[0].flatten())):
             error_msg = f"SQL Syntax Error detected by sqlparse."
             print(error_msg)
             return {**state, "error": error_msg}
        print("SQLParse basic syntax check passed.")
    except Exception as e:
        error_msg = f"Error during sqlparse validation: {e}"
        print(error_msg)
        return {**state, "error": error_msg}

    # 2. Rule-Based Safety Checks
    disallowed_keywords = ["DELETE", "UPDATE", "DROP", "TRUNCATE", "GRANT", "REVOKE", "INSERT", "ALTER"]
    tokens = [t.value.upper() for t in parsed[0].flatten() if t.ttype in sqlparse.tokens.Keyword]
    found_disallowed = [kw for kw in disallowed_keywords if kw in tokens]

    if found_disallowed:
        error_msg = f"Safety Error: Disallowed keyword(s) found: {', '.join(found_disallowed)}"
        print(error_msg)
        return {**state, "error": error_msg}
    print("Safety keyword check passed.")

    print("SQL validation successful.")
    return state

def execute_sql_node(state: RAGSQLState) -> RAGSQLState:
    """Executes the generated SQL query."""
    print("--- Node: Execute SQL ---")
    sql_query = state.get("sql_query")
    db_uri = state["db_uri"]

    if state.get("error"):
        print("Skipping SQL execution due to previous error.")
        return state
    if not sql_query:
        print("No SQL query to execute.")
        return {**state, "sql_result": None, "error": "No SQL query was generated."}

    print(f"Executing SQL:\n{sql_query}")
    engine = None
    try:
        engine = create_engine(db_uri)
        with engine.connect() as connection:
            limited_query = sql_query
            df = pd.read_sql_query(text(limited_query), connection)

        print(f"Query executed successfully, {len(df)} rows returned.")

        if df.empty:
            result_string = "Query executed successfully, but returned no results."
        else:
            result_string = df.head(20).to_markdown(index=False)
            result_string += f"\n\n(Showing top {min(len(df), 20)} of {len(df)} rows fetched)"

        print(f"Result Preview (Markdown Head):\n{result_string[:500]}...")
        return {**state, "sql_result": result_string, "error": None}

    except Exception as e:
        error_msg = f"SQL Execution Error: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {**state, "sql_result": None, "error": error_msg}
    finally:
         if engine:
             engine.dispose()
         gc.collect()

def summarize_node(state: RAGSQLState) -> RAGSQLState:
    """Summarizes the SQL results."""
    print("--- Node: Summarize Results ---")
    sql_result = state.get("sql_result")
    question = state["question"]

    if state.get("error"):
         print("Skipping summarization due to previous error.")
         return state
    if not sql_result or "no results" in sql_result.lower():
        print("No results to summarize.")
        return {**state, "summary": "No results found for the query."}

    print("Summarizing results...")
    llm = ChatOpenAI(model=MODEL_SUMMARY, temperature=0, api_key=OPENAI_API_KEY)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs = [Document(page_content=sql_result)]
    split_docs = text_splitter.split_documents(docs)

    try:
        summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = summary_chain.invoke(split_docs).get("output_text", "Error: Could not generate summary.")

        print(f"Generated Summary:\n{summary}")
        return {**state, "summary": summary}

    except Exception as e:
        error_msg = f"Error during summarization: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {**state, "summary": error_msg}
    finally:
         gc.collect()

def should_generate_sql(state: RAGSQLState) -> str:
    """Checks if context retrieval was successful."""
    print("--- Condition: Should Generate SQL? ---")
    if state.get("error"):
        print("Decision: Error during retrieval. Routing to END.")
        return "end_run"
    else:
        print("Decision: Context retrieved successfully. Proceeding to SQL generation.")
        return "generate_sql"

def should_execute_sql(state: RAGSQLState) -> str:
    """Determines if SQL execution should proceed *after validation*."""
    print("--- Condition: Should Execute SQL (Post-Validation)? ---")
    if state.get("error") or not state.get("sql_query"):
        print(f"Decision: Error encountered ('{state.get('error')}') or no SQL query. Routing to END.")
        return "end_run"
    else:
        print("Decision: SQL query validated successfully. Proceeding to execution.")
        return "execute_sql"

def should_summarize(state: RAGSQLState) -> str:
    """Determines if summarization should proceed."""
    print("--- Condition: Should Summarize? ---")
    if state.get("error") or not state.get("sql_result") or "no results" in state.get("sql_result","").lower():
        print("Decision: No results to summarize or error occurred. Routing to END.")
        return "end_run"
    else:
        print("Decision: Results found. Proceeding to summarization.")
        return "summarize_results"

print("\n--- Building RAG SQL LangGraph Workflow ---")
workflow = StateGraph(RAGSQLState)

workflow.add_node("retrieve_context", retrieve_context_node)
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("validate_sql", validate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("summarize_results", summarize_node)

workflow.set_entry_point("retrieve_context")

workflow.add_conditional_edges(
    "retrieve_context",
    should_generate_sql,
    {"generate_sql": "generate_sql",
     "end_run": END}
)
workflow.add_edge("generate_sql", "validate_sql")

workflow.add_conditional_edges(
    "validate_sql",
    should_execute_sql,
    {"execute_sql": "execute_sql",
     "end_run": END}
)
workflow.add_conditional_edges(
    "execute_sql",
    should_summarize,
    {"summarize_results": "summarize_results",
     "end_run": END}
)
workflow.add_edge("summarize_results", END)

app = workflow.compile()
print("Workflow compiled successfully.")

def run_pipeline():
    print("\n=== LangGraph RAG SQL Query & Summarization Pipeline (SQLCoder) ===")

    while True:
        print("\n" + "=" * 50)
        try:
            user_question = input("Enter your question about the database (or 'exit' to quit):\n> ")
            if user_question.lower() in ["exit", "quit", "q"]:
                break
            if not user_question.strip():
                 print("Please enter a question.")
                 continue

            initial_state = RAGSQLState(
                question=user_question,
                db_uri=DATABASE_URI,
                retrieved_schema_context=None,
                retrieved_sql_examples=None,
                sql_query=None,
                sql_result=None,
                summary=None,
                error=None
            )

            print("\n--- Running Workflow ---")
            final_state = app.invoke(initial_state, {"recursion_limit": 10})
            print("\n--- Workflow Complete ---")

            print("\nRetrieved Schema Context (Top Chunks):")
            print(final_state.get('retrieved_schema_context', 'N/A')[:1000]+"...") if len(final_state.get('retrieved_schema_context', 'N/A')) > 1000 else print(final_state.get('retrieved_schema_context', 'N/A'))

            print("\nRetrieved SQL Examples (Top Examples):")
            print(final_state.get('retrieved_sql_examples', 'N/A')[:1000]+"...") if len(final_state.get('retrieved_sql_examples', 'N/A')) > 1000 else print(final_state.get('retrieved_sql_examples', 'N/A'))

            print("\nGenerated SQL:")
            print(final_state.get('sql_query', 'N/A'))

            print("\nQuery Result Preview:")
            print(final_state.get('sql_result', 'N/A'))

            print("\nSummary:")
            print(final_state.get('summary', 'N/A'))

            if final_state.get("error"):
                 print(f"\n--- Error Encountered during run ---\n{final_state['error']}")

            gc.collect()

        except KeyboardInterrupt:
            print("\nExiting program...")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred in the main loop: {e}")
            import traceback
            traceback.print_exc()
            gc.collect()

    print("\nPipeline finished.")

if __name__ == "__main__":
    run_pipeline()