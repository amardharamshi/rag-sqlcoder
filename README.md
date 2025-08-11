# RAG SQL Query Generator 

**Turn natural language questions into safe, optimized SQL queries — and get instant results with summaries.**  
This project combines the power of **Retrieval-Augmented Generation (RAG)**, **SQLCoder-7B**, and **LangGraph** to make working with databases as easy as having a conversation.

---

##  Features

- **Ask in plain English** → Get valid, executable SQL queries.
- **Retrieval-Augmented Generation** for context-aware query building.
- **SQL validation & safety checks** before execution.
- **Automatic execution** on a MySQL/MariaDB database.
- **Result summarization** with GPT-4o for quick insights.
- **Customizable schema retrieval** from your own database.
- **GPU-friendly** for faster model inference (recommended).

---

##  How It Works

1. **Retrieve Context**  
   Pulls relevant schema details and past SQL examples from a vector store.

2. **Generate SQL**  
   Uses SQLCoder-7B to craft the best-fit SQL query.

3. **Validate**  
   Checks syntax and ensures no unsafe commands will run.

4. **Execute**  
   Runs the query against your configured MySQL database.

5. **Summarize**  
   GPT-4o condenses results into a short, human-readable answer.

---

##  Requirements

- Python **3.8+**
- MySQL / MariaDB database
- NVIDIA GPU (**recommended** — ~10GB VRAM for SQLCoder-7B)
- OpenAI API key (for summarization)

See [`requirements.txt`](./requirements.txt) for full dependencies.

---

##  Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-sql-generator.git
cd rag-sql-generator

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt
