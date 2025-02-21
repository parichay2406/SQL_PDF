import streamlit as st
import pandas as pd
import sqlite3
import json
from datetime import datetime
from io import StringIO
import os
from dotenv import load_dotenv
import requests
from pypdf import PdfReader
import io
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from together import Together
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import deepdoctection as dd
import tempfile
import re
from unstract.llmwhisperer import LLMWhispererClientV2

# Load environment variables from .env file
load_dotenv()

# Configuration
GOOGLE_PROJECT_ID = os.getenv('GOOGLE_PROJECT_ID')
GOOGLE_LOCATION = os.getenv('GOOGLE_LOCATION', 'us')
GOOGLE_PROCESSOR_ID = os.getenv('GOOGLE_PROCESSOR_ID')

# Initialize Lambda API key check
if 'LAMBDA_API_KEY' not in os.environ:
    st.error("Please set your LAMBDA_API_KEY in the .env file")
    st.stop()

# Add Extractuous API configuration after other API configurations
EXTRACTUOUS_API_KEY = os.getenv('EXTRACTUOUS_API_KEY', 'extr_2UexUJzZZzDLxSzcDVf8htfKJoiA3q7JHpjn16JiotyBHg')
EXTRACTUOUS_API_URL = "https://api.extractous.com/v1/extract"

# Add LLM Whisperer configuration after Extractuous config
LLMWHISPERER_API_KEY = os.getenv('LLMWHISPERER_API_KEY', 'kXGTWL0B2_sB3kb1SysX_ziPDlRfj3RqN3U6E5s4CRw')
LLMWHISPERER_BASE_URL = "https://llmwhisperer-api.us-central.unstract.com/api/v2"

# Add Together AI configuration
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY', '595787d329aeb19c7c39a6fa5a63945b765cfb47803f3ce475601740e54ecbc0')
TOGETHER_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

def test_lambda_api():
    """Test Lambda API connection"""
    try:
        LAMBDA_API_KEY = os.getenv("LAMBDA_API_KEY")
        if not LAMBDA_API_KEY:
            st.error("LAMBDA_API_KEY not found in environment variables")
            return False
            
        headers = {
            "Authorization": f"Bearer {LAMBDA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Simplified test request with shorter timeout
        data = {
            "model": "llama3.1-405b-instruct-fp8",
            "messages": [
                {
                    "role": "user",
                    "content": "Test"
                }
            ],
            "temperature": 0.7,
            "max_tokens": 10,  # Reduced tokens for faster response
            "stream": False
        }
        
        # Increased timeout and added retries
        session = requests.Session()
        retries = requests.adapters.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))
        
        response = session.post(
            "https://api.lambdalabs.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30  # Increased timeout
        )
        
        if response.status_code == 200:
            st.success("✅ Successfully connected to Lambda Labs API")
            return True
        else:
            st.error(f"❌ API connection failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        st.error("❌ API connection timed out. Please try again.")
        return False
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API connection failed: {str(e)}")
        return False
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return False

def init_llm():
    """Initialize LLM with Lambda API"""
    if test_lambda_api():
        LAMBDA_API_KEY = os.getenv("LAMBDA_API_KEY")
        
        def llm_wrapper(prompt):
            headers = {
                "Authorization": f"Bearer {LAMBDA_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama3.1-405b-instruct-fp8",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful warehouse data analyst."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
                "stream": False
            }
            
            try:
                # Use session with retries
                session = requests.Session()
                retries = requests.adapters.Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[500, 502, 503, 504]
                )
                session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))
                
                response = session.post(
                    "https://api.lambdalabs.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30  # Increased timeout
                )
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    return None
                    
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
                return None
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {str(e)}")
                return None
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                return None
                
        return llm_wrapper
    return None

def init_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = False
    if 'pdf_store' not in st.session_state:
        st.session_state['pdf_store'] = {}
    if 'dataframes' not in st.session_state:
        st.session_state['dataframes'] = {}
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []
    if 'llm_wrapper' not in st.session_state:
        st.session_state['llm_wrapper'] = init_llm()

# Initialize session state at startup
init_session_state()

def load_schema():
    try:
        # First load the schema file
        with open('schema.txt', 'r') as file:
            schema = file.read()
            
        # Debug output
        st.write("Debug - Loaded files:", [f.name for f in st.session_state.uploaded_files])
        
        # Now get sample data from each table
        conn = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES)
        sample_data = []
        
        for file in st.session_state.uploaded_files:
            # Debug output
            st.write(f"Debug - Processing file: {file.name}")
            
            table_name = file.name.replace('.csv', '').lower()
            
            # Use cached dataframe if available, otherwise read and cache it
            if table_name not in st.session_state.dataframes:
                df = pd.read_csv(file)
                st.session_state.dataframes[table_name] = df
            else:
                df = st.session_state.dataframes[table_name]
            
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            # Get first row as sample
            if len(df) > 0:
                sample = df.iloc[0].to_dict()
                sample_data.append(f"\nSample row from {table_name}:\n{json.dumps(sample, indent=2)}")
        
        # Combine schema and sample data
        full_schema = schema + "\n\nSAMPLE DATA:" + "\n".join(sample_data)
        
        return full_schema
    except FileNotFoundError:
        st.error("schema.txt file not found!")
        return None
    except Exception as e:
        st.error(f"Error loading schema: {str(e)}")
        return None

def init_db():
    """Initialize SQLite database with schema"""
    conn = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES)
    schema = load_schema()
    if schema:
        try:
            conn.executescript(schema)
            return conn
        except sqlite3.Error as e:
            st.error(f"Error initializing database: {e}")
            return None

def upload_data(conn, uploaded_files):
    """Upload CSV files to SQLite database"""
    try:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                try:
                    # Read the CSV file into a pandas DataFrame
                    df = pd.read_csv(file)
                    
                    # Get table name from file name (remove .csv and lowercase)
                    table_name = file.name.replace('.csv', '').lower()
                    
                    # Save to SQLite
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                    
                    # Add to session state
                    st.session_state.uploaded_files.add(file.name)
                    
                except Exception as e:
                    st.error(f"Error uploading {file.name}: {e}")
                    return False
        return True
    except Exception as e:
        st.error(f"Error in upload_data: {e}")
        return False

def get_db_context() -> str:
    """Get database schema and metadata context"""
    context = []
    
    # Check if schema has been uploaded
    if 'schema_content' not in st.session_state:
        return "Please upload schema.txt file first"
    
    # Add uploaded schema template
    context.append(f"Base Schema Template:\n{st.session_state['schema_content']}\n")
    
    # Add actual sensor table mappings
    context.append("\nActual Sensor Tables:")
    for table_name, df in st.session_state['dataframes'].items():
        if '_R' in table_name:  # Weight sensor tables (e.g., A_R1)
            room = table_name[0]
            rack = table_name.split('_R')[1]
            context.append(f"- {table_name}: Weight sensor for Room {room}, Rack {rack}")
            context.append(f"  Follows weight_sensors template with columns: {', '.join(df.columns)}")
        elif '_temp' in table_name.lower():  # Temperature sensor tables
            room = table_name[0]
            context.append(f"- {table_name}: Temperature sensor for Room {room}")
            context.append(f"  Follows temperature_sensors template with columns: {', '.join(df.columns)}")
    
    return "\n".join(context)

def get_follow_up_question(user_prompt, schema):
    try:
        llm = init_llm()
        if not llm:
            return None
            
        # Get current DB context
        db_context = get_db_context()
        if not db_context:
            return None
            
        system_message = f"""You are a warehouse data analyst. Based on the user's question and the current database state:
        1. If the question is clear enough, respond with 'CLEAR: <answer>' where <answer> is the actual answer to their question
        2. If you need clarification, ask ONE relevant follow-up question
        
        {db_context}"""
        
        response = llm(
            system_message + "\n\nUser Question: " + user_prompt + "\n\nResponse:"
        )
        
        # Check if it's a CLEAR response
        if response.upper().startswith('CLEAR:'):
            return None  # Skip follow-up
        
        return response
    except Exception as e:
        st.error(f"Error getting follow-up question: {str(e)}")
        return None

def determine_data_requirement(conversation_history, schema):
    try:
        llm = init_llm()
        if not llm:
            return None
            
        # Convert conversation history to string format
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in conversation_history
        ])
            
        system_message = """Analyze if the user's query requires:
        1. Only static data (rooms, products, placements)
        2. Both static and sensor data (temperature, weight readings)
        Respond with ONLY either 'static' or 'static + sensory'."""
        
        response = llm(
            system_message + "\n\nConversation:\n" + conversation_text + "\n\nData requirement:"
        )
        
        return response
    except Exception as e:
        st.error(f"Error determining data requirement: {e}")
        return None

def init_hf_client():
    token = os.getenv("HF_API_TOKEN")
    if not token:
        st.error("HF_API_TOKEN not found in environment variables")
        st.stop()
    return InferenceClient(token=token)

def test_inference_access():
    client = InferenceClient(token=os.getenv("HF_API_TOKEN"))
    try:
        # Using a smaller DeepSeek model
        response = client.text_generation(
            "SELECT * FROM test",  # Simple test prompt
            model="deepseek-ai/deepseek-coder-6.7b-instruct",
            max_new_tokens=50
        )
        st.success("Successfully connected to DeepSeek!")
        st.write("Test response:", response)
    except Exception as e:
        st.error(f"Connection failed: {str(e)}")
        st.error("Error details:", str(e))

def validate_and_fix_sql(generated_sql, schema):
    try:
        client = init_hf_client()
        prompt = f"""You are a SQL expert. Review and optimize this SQL query.
        Return ONLY a JSON object with NO additional text.
        
        Schema:
        {schema}
        
        SQL:
        {generated_sql}
        
        Required JSON format:
        {{
            "optimized_sql": "your optimized sql query here",
            "changes": ["change 1", "change 2"]
        }}"""
        
        response = client.text_generation(
            prompt,
            model="deepseek-ai/deepseek-coder-1.5b",
            max_new_tokens=500,
            temperature=0
        )
        
        # Debug: Print raw response
        st.write("Debug - Raw response:", response)
        
        try:
            # Extract JSON from response
            if '{' in response:
                json_str = response[response.find('{'):response.rfind('}')+1]
                validation_result = json.loads(json_str)
                
                # Validate structure
                if not isinstance(validation_result, dict):
                    raise ValueError("Response is not a dictionary")
                if 'optimized_sql' not in validation_result:
                    raise ValueError("Response missing optimized_sql field")
                    
                return validation_result
                
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error: {str(e)}")
            st.error(f"Received content: {response}")
            return None
            
    except Exception as e:
        st.error(f"Validation error: {str(e)}")
        return None

def generate_sql_and_explanation(conversation_history: List[Dict], schema: str) -> Optional[Dict]:
    """Generate SQL query from natural language"""
    try:
        # Extract room and rack information from question
        question = conversation_history[-1]['content']
        
        # Determine which sensor table to query
        room_match = re.search(r'room\s+([A-Z])', question, re.IGNORECASE)
        rack_match = re.search(r'rack\s+(\d+)', question, re.IGNORECASE)
        
        if room_match and rack_match:
            room = room_match.group(1).upper()
            rack = rack_match.group(1)
            table_name = f"{room}_R{rack}"
            
            # Check if table exists
            if table_name in st.session_state.dataframes:
                query = f"""
                SELECT timestamp, weight
                FROM {table_name}
                WHERE timestamp >= '2025-02-15 00:00:00'
                ORDER BY timestamp DESC
                LIMIT 1
                """
                
                return {
                    'sql': query,
                    'explanation': f'Querying weight sensor data from {table_name} (Room {room}, Rack {rack})'
                }
            else:
                return {
                    'sql': f'-- Table {table_name} not found in database',
                    'explanation': f'No sensor data available for Room {room}, Rack {rack}'
                }
        
        return {
            'sql': '-- Could not determine room and rack from question',
            'explanation': 'Please specify both room and rack number in your question'
        }
        
    except Exception as e:
        st.error(f"Error generating SQL: {str(e)}")
        return None

def execute_query(query: str) -> Optional[pd.DataFrame]:
    """Execute SQL query on the loaded dataframes"""
    try:
        import sqlite3
        
        # Create in-memory SQLite database
        conn = sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES)
        
        # Load all dataframes into SQLite
        for table_name, df in st.session_state.dataframes.items():
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Execute query
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result
        
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return None

def generate_final_answer(question: str, sql_result: Optional[Dict], pdf_results: Optional[List[str]]) -> str:
    """Generate final answer combining SQL and PDF results"""
    answer_parts = []
    
    if sql_result:
        answer_parts.append("Database Results:")
        answer_parts.append(sql_result.get('explanation', 'No explanation available'))
    
    if pdf_results:
        answer_parts.append("\nDocument Results:")
        for i, result in enumerate(pdf_results, 1):
            answer_parts.append(f"{i}. {result}")
    
    if not answer_parts:
        return "I couldn't find any relevant information to answer your question."
    
    return "\n".join(answer_parts)

def show_custom_query_sidebar():
    """Simple sidebar for custom SQL queries"""
    with st.sidebar:
        st.header("🔍 Custom SQL Query")
        
        # Use a different key for sidebar state
        if 'sidebar_query' not in st.session_state:
            st.session_state.sidebar_query = ''
        
        # Query input with its own session state
        query = st.text_area(
            "SQL query:", 
            value=st.session_state.sidebar_query,
            key="sidebar_query_input",
            height=150
        )
        
        # Execute button with its own key
        if st.button("Execute Query", key="sidebar_execute"):
            if not st.session_state.uploaded_files:
                st.error("Please upload files first")
                return
                
            if query:
                result, error = execute_custom_query(query)
                if error:
                    st.error(error)
                elif result is not None:
                    st.dataframe(result)
            else:
                st.warning("Please enter a query")

def extract_text_from_pdf(pdf_file):
    """Alternative PDF text extraction"""
    reader = PdfReader(io.BytesIO(pdf_file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def process_pdf(pdf_file) -> Dict:
    """Initial PDF processing using PyPDF and spaCy"""
    # Extract text using PyPDF
    text = extract_text_from_pdf(pdf_file)
    
    # Extract global metadata using spaCy
    doc = st.session_state.nlp(text)
    global_metadata = {
        "doc_id": pdf_file.name,
        "entities": [ent.text for ent in doc.ents],
        "key_terms": [token.text for token in doc if token.is_stop == False and token.is_punct == False],
        "file_type": "pdf"
    }
    
    # Create chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?"]
    )
    
    chunks = text_splitter.split_text(text)
    
    # Process chunks and get embeddings
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        # Get chunk embedding
        response = st.session_state.together_client.embeddings.create(
            model="BAAI/bge-large-en-v1.5",
            input=chunk
        )
        embedding = response.data[0].embedding
        
        # Get chunk metadata
        chunk_doc = st.session_state.nlp(chunk)
        chunk_metadata = {
            "chunk_id": i,
            "local_entities": [ent.text for ent in chunk_doc.ents],
            "local_terms": [token.text for token in chunk_doc if token.is_stop == False and token.is_punct == False]
        }
        
        processed_chunks.append({
            "text": chunk,
            "embedding": embedding,
            "metadata": chunk_metadata
        })
    
    return {
        "global_metadata": global_metadata,
        "chunks": processed_chunks
    }

def update_faiss_index(processed_pdfs: Dict):
    """Update FAISS index with all PDF chunks"""
    # Get dimensionality from first embedding
    first_embedding = next(iter(processed_pdfs.values()))["chunks"][0]["embedding"]
    dim = len(first_embedding)
    
    # Initialize FAISS index
    index = faiss.IndexFlatL2(dim)
    
    # Add all embeddings
    all_embeddings = []
    for pdf in processed_pdfs.values():
        for chunk in pdf["chunks"]:
            all_embeddings.append(chunk["embedding"])
    
    if all_embeddings:
        index.add(np.array(all_embeddings))
    
    return index

def get_embedding(text: str) -> List[float]:
    """Get embedding for text using Together API"""
    try:
        response = st.session_state.together_client.embeddings.create(
            model="BAAI/bge-large-en-v1.5",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return []

def calculate_metadata_relevance(metadata: Dict, query: str) -> float:
    """Calculate relevance score based on metadata"""
    try:
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Get entities and terms from metadata
        entities = metadata.get("local_entities", [])
        terms = metadata.get("local_terms", [])
        
        # Count matches
        entity_matches = sum(1 for entity in entities if entity.lower() in query_lower)
        term_matches = sum(1 for term in terms if term.lower() in query_lower)
        
        # Calculate score (simple average)
        total_items = len(entities) + len(terms)
        if total_items == 0:
            return 0.0
            
        return (entity_matches + term_matches) / total_items
        
    except Exception as e:
        st.error(f"Error calculating metadata relevance: {str(e)}")
        return 0.0

def search_pdf_content(query: str, top_k: int = 3) -> List[str]:
    """Search PDF content using metadata and content"""
    try:
        relevant_chunks = []
        
        # First, search through document-level metadata
        for doc_name, doc_data in st.session_state.pdf_store.items():
            doc_metadata = doc_data["document_level"]
            
            # Check if query terms match document-level metadata
            if any(term.lower() in query.lower() for term in doc_metadata["key_entities"]):
                # If match found, search through page-level metadata
                for page in doc_data["pages"]:
                    # Check page-level metadata match
                    if any(term.lower() in query.lower() for term in page["key_entities"]):
                        # Add relevant text chunks from this page
                        relevant_chunks.extend(page["text_chunks"])
                        
                        # Add relevant tables if query suggests table interest
                        if "table" in query.lower() or "data" in query.lower():
                            for table in page["tables"]:
                                relevant_chunks.append({
                                    "text": f"Table data: {str(table['data'])}",
                                    "score": 1.0
                                })
        
        # Sort chunks by relevance and return top_k
        sorted_chunks = sorted(relevant_chunks, key=lambda x: x.get("score", 0), reverse=True)
        return [chunk["text"] for chunk in sorted_chunks[:top_k]]
        
    except Exception as e:
        st.error(f"Error searching PDF content: {str(e)}")
        return []

def handle_user_question(user_question: str):
    """Handle user questions and generate responses"""
    try:
        # Initialize LLM if not already done
        if 'llm_wrapper' not in st.session_state:
            st.session_state['llm_wrapper'] = init_llm()
        
        if not st.session_state['llm_wrapper']:
            st.error("Failed to initialize LLM")
            return
            
        # Determine search type
        search_type = determine_search_type(user_question)
        st.write(f"🔎 Search Type: {search_type.upper()}")
        
        final_answer = ""
        
        if search_type in ['sql', 'both']:
            # Get database schema and sample data
            schema_with_samples = get_db_context()
            st.write("Available Schema and Sample Data:")
            st.code(schema_with_samples, language='sql')
            
            # Generate SQL query with more context
            sql_prompt = f"""You are a SQL expert. Given this database schema and sample data:
            {schema_with_samples}
            
            Generate a SQL query to answer this question: "{user_question}"
            Consider table relationships and join conditions carefully.
            Return ONLY the executable SQL query without any markdown formatting or explanation.
            The query should be a plain SQL statement.
            
            If the question cannot be answered using SQL, return:
            -- The provided question does not relate to the given database schema.
            """
            
            sql_query = st.session_state['llm_wrapper'](sql_prompt)
            
            # Clean up the response to get just the SQL query
            sql_query = sql_query.strip()
            if sql_query.startswith('```sql'):
                sql_query = sql_query[6:]
            if sql_query.endswith('```'):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            if sql_query and not sql_query.startswith('--'):
                # Show the generated SQL
                st.write("🔍 Generated SQL Query:")
                st.code(sql_query, language='sql')
                
                # Execute the query
                result_df = execute_query(sql_query)
                
                if result_df is not None and not result_df.empty:
                    st.write("📊 Query Results:")
                    st.dataframe(result_df)
                    
                    # Generate SQL explanation with more context
                    explanation_prompt = f"""Given this database query and its results, provide a detailed explanation:
                    
                    Question: {user_question}
                    
                    SQL Query:
                    {sql_query}
                    
                    Results:
                    {result_df.to_string()}
                    
                    Please explain:
                    1. What the query is doing
                    2. What the results mean
                    3. The answer to the original question
                    """
                    
                    sql_explanation = st.session_state['llm_wrapper'](explanation_prompt)
                    final_answer += f"\n\nDatabase Analysis:\n{sql_explanation}"
                else:
                    st.warning("Query returned no results. Please try rephrasing your question.")
        
        if search_type in ['pdf', 'both']:
            # Search PDF content
            pdf_results = search_pdf_content(user_question)
            if pdf_results:
                st.write("📄 Relevant PDF Content Found")
                
                # Generate PDF explanation with more context
                pdf_context = "\n".join(pdf_results)
                pdf_prompt = f"""Based on these document excerpts, provide a detailed answer:
                
                Question: {user_question}
                
                Relevant Document Content:
                {pdf_context}
                
                Please provide:
                1. A direct answer to the question
                2. Supporting information from the documents
                3. Any relevant context
                """
                
                pdf_explanation = st.session_state['llm_wrapper'](pdf_prompt)
                final_answer += f"\n\nDocument Analysis:\n{pdf_explanation}"
        
        # Display final combined answer in a text box
        if final_answer:
            st.text_area("💡 Complete Analysis:", final_answer, height=200)
        else:
            st.warning("No relevant information found in the available sources.")
            
    except Exception as e:
        st.error(f"Error handling question: {str(e)}")
        st.error(f"Debug - Exception details: {type(e).__name__}: {str(e)}")

def determine_search_type(query: str) -> str:
    """Determine if query needs SQL, PDF, or both based on content"""
    try:
        # Check available data sources
        has_pdfs = len(st.session_state['pdf_store']) > 0
        has_sql = len(st.session_state['dataframes']) > 0

        if not has_pdfs and not has_sql:
            return "none"

        # Load schema.txt for SQL matching
        with open('schema.txt', 'r') as f:
            schema = f.read()

        # Check for SQL relevance (look for table and column names in query)
        sql_relevance = 0
        for table_name in st.session_state['dataframes'].keys():
            if table_name.lower() in query.lower():
                sql_relevance += 2
            # Check column names
            df = st.session_state['dataframes'][table_name]
            for col in df.columns:
                if col.lower() in query.lower():
                    sql_relevance += 1

        # Check for PDF relevance
        pdf_relevance = 0
        for pdf_name, pdf_data in st.session_state['pdf_store'].items():
            # Search through the JSON content for relevant terms
            json_str = str(pdf_data).lower()
            query_terms = query.lower().split()
            for term in query_terms:
                if term in json_str:
                    pdf_relevance += 1

        # Make decision based on relevance scores
        if sql_relevance > 0 and pdf_relevance > 0:
            return "both"
        elif sql_relevance > pdf_relevance:
            return "sql"
        elif pdf_relevance > sql_relevance:
            return "pdf"
        else:
            return "pdf"  # Default to PDF if no clear winner

    except Exception as e:
        st.error(f"Error in search type determination: {str(e)}")
        return "none"

def handle_sql_query(query: str) -> str:
    """Process SQL-related query using LLM"""
    try:
        # Get schema and sample data
        schema_context = get_db_context()
        
        # Send to LLM for SQL generation and explanation
        prompt = f"""Given this database schema and sample data:
        {schema_context}
        
        Generate a response for this question: {query}
        Include both the SQL query and a natural language explanation."""
        
        response = st.session_state['llm_wrapper'](prompt)
        return response
    except Exception as e:
        st.error(f"Error in SQL query handling: {str(e)}")
        return None

def handle_pdf_query(query: str) -> str:
    """Process PDF-related query using LLM"""
    try:
        # Prepare PDF content for LLM
        pdf_context = ""
        for pdf_name, pdf_data in st.session_state['pdf_store'].items():
            pdf_context += f"\nContent from {pdf_name}:\n{json.dumps(pdf_data, indent=2)}\n"
        
        # Send to LLM for analysis
        prompt = f"""Given this PDF content:
        {pdf_context}
        
        Answer this question: {query}
        Provide specific references to the documents where you found the information."""
        
        response = st.session_state['llm_wrapper'](prompt)
        return response
    except Exception as e:
        st.error(f"Error in PDF query handling: {str(e)}")
        return None

def process_pdf_with_extractuous(file) -> dict:
    """Process PDF with Extractuous API and return JSON"""
    try:
        files = {
            'file': file,
            'config': (None, json.dumps({"strategy": "FAST_WITH_OCR"}))
        }
        headers = {'X-Api-Key': EXTRACTUOUS_API_KEY}
        
        response = requests.post(EXTRACTUOUS_API_URL, headers=headers, files=files)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"PDF Processing Error: {str(e)}")
        return None

def get_data_context() -> str:
    """Get schema and sample data context"""
    context = ""
    
    # Read schema from session state instead of file
    if 'schema_content' in st.session_state:
        context += f"Schema:\n{st.session_state['schema_content']}\n\n"
    else:
        return "Schema not available. Please upload schema.txt"
    
    # Add sample data from dataframes
    context += "Sample Data:\n"
    for table_name, df in st.session_state['dataframes'].items():
        if not df.empty:
            context += f"\nTable: {table_name}\n"
            context += df.head(2).to_string() + "\n"
    
    return context

def determine_query_type(query: str) -> str:
    """Use LLM to determine if query needs SQL, PDF, or both"""
    try:
        # Prepare context for LLM
        data_context = get_data_context()
        pdf_context = json.dumps(st.session_state['pdf_store'], indent=2)
        
        prompt = f"""Determine if this query requires SQL database access, PDF document access, or both.
        
        Available Data:
        {data_context}
        
        Available PDFs:
        {pdf_context}
        
        Query: {query}
        
        Return ONLY one of these words: SQL, PDF, or BOTH"""
        
        response = st.session_state['llm_wrapper'](prompt)
        return response.strip().upper()
    except Exception as e:
        st.error(f"Error in query type determination: {str(e)}")
        return "PDF"  # Default to PDF if error

def get_sql_answer(query: str) -> str:
    """Generate and execute SQL query, then return answer"""
    try:
        # Get schema and sample data for context
        data_context = get_data_context()
        
        # Generate SQL query with strict formatting instructions
        sql_generation_prompt = f"""Given this schema and sample data:
        {data_context}
        
        Generate a SQL query to answer: {query}
        
        IMPORTANT: Return ONLY the raw SQL query without any formatting, markdown, or backticks.
        The query should start directly with SELECT and end with semicolon."""
        
        generated_sql = st.session_state['llm_wrapper'](sql_generation_prompt)
        
        # Clean up any markdown formatting that might have been included
        generated_sql = generated_sql.strip()
        generated_sql = generated_sql.replace('```sql', '').replace('```', '')
        
        # Execute the query
        conn = sqlite3.connect(':memory:')
        
        # Load dataframes into SQLite
        for table_name, df in st.session_state['dataframes'].items():
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        try:
            # Execute and get results
            results_df = pd.read_sql_query(generated_sql, conn)
            
            # Show the actual query and results
            st.write("🔍 Generated SQL:")
            st.code(generated_sql, language='sql')
            st.write("📊 Query Results:")
            st.dataframe(results_df)
            
            # Get interpretation of results
            interpretation_prompt = f"""Based on these query results, provide a clear answer to the original question.
            
            Question: {query}
            Results: {results_df.to_string()}
            
            Provide a direct, factual answer based only on the data shown."""
            
            return st.session_state['llm_wrapper'](interpretation_prompt)
            
        except sqlite3.Error as e:
            return f"SQL Error: {str(e)}"
        finally:
            conn.close()
            
    except Exception as e:
        return f"Error: {str(e)}"

def process_pdf_with_llmwhisperer(file) -> dict:
    """Process handwritten PDF with LLM Whisperer API and return JSON"""
    try:
        # Initialize client
        client = LLMWhispererClientV2(
            base_url=LLMWHISPERER_BASE_URL,
            api_key=LLMWHISPERER_API_KEY
        )
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Process with LLM Whisperer
            result = client.whisper(
                file_path=tmp_path,
                wait_for_completion=True,
                wait_timeout=200
            )
            return result
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        st.error(f"Error processing handwritten PDF: {str(e)}")
        return None

def get_pdf_answer(query: str) -> str:
    """Get PDF-based answer from LLM including both Extractuous and LLM Whisperer results"""
    # Combine both normal and handwritten content
    combined_content = {}
    
    # Add Extractuous content
    for filename, content in st.session_state['pdf_store'].items():
        if 'extractuous' in content:
            combined_content[f"{filename} (machine-printed)"] = content['extractuous']
    
    # Add LLM Whisperer content
    for filename, content in st.session_state['pdf_store'].items():
        if 'llmwhisperer' in content:
            combined_content[f"{filename} (handwritten)"] = content['llmwhisperer']
    
    pdf_context = json.dumps(combined_content, indent=2)
    prompt = f"""Given these PDF contents (including both machine-printed and handwritten documents):
    {pdf_context}
    
    Answer this question: {query}
    Consider both machine-printed and handwritten content in your answer."""
    
    return st.session_state['llm_wrapper'](prompt)

def get_final_answer(query: str, sql_answer: str = None, pdf_answer: str = None) -> str:
    """Generate final combined answer"""
    try:
        prompt = f"""Original Question: {query}

        Available Information:
        - SQL Analysis: {sql_answer if sql_answer else 'Not Available'}
        - PDF Analysis: {pdf_answer if pdf_answer else 'Not Available'}

        Provide a clear, professional answer using only standard characters and punctuation.
        Format the response in plain text without any special formatting or unicode characters."""
        
        answer = st.session_state['llm_wrapper'](prompt)
        
        # Clean up any special characters or formatting
        answer = answer.encode('ascii', 'ignore').decode('ascii')
        answer = answer.replace('```', '').strip()
        
        return answer
        
    except Exception as e:
        return f"Error generating final answer: {str(e)}"

def analyze_query_complexity(query: str) -> dict:
    """Use LLAMA to determine if query is complex"""
    try:
        prompt = f"""You are a query analyzer. Given this query, determine if it's complex or simple.
        
        Query: {query}
        
        A query is complex if it:
        1. Contains multiple questions
        2. Requires information from multiple sources
        3. Has dependencies between parts
        
        Respond with EXACTLY this JSON format (no other text):
        {{
            "is_complex": true/false,
            "reason": "brief explanation"
        }}
        
        Example response:
        {{
            "is_complex": false,
            "reason": "Single question about room data"
        }}"""
        
        # Get raw response from LLAMA
        response = st.session_state['llm_wrapper'](prompt)
        
        # Clean the response to extract just the JSON part
        json_str = response
        if '```json' in response:
            json_str = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            json_str = response.split('```')[1].split('```')[0]
            
        # Strip any extra whitespace or newlines
        json_str = json_str.strip()
        
        # Parse the JSON
        result = json.loads(json_str)
        
        # Validate required fields
        if not isinstance(result, dict) or 'is_complex' not in result or 'reason' not in result:
            raise ValueError("Invalid response format")
            
        return result
        
    except Exception as e:
        st.error(f"Error in complexity analysis: {str(e)}")
        return {
            "is_complex": False,
            "reason": "Error in analysis, treating as simple query"
        }

def decompose_complex_query(query: str) -> dict:
    """Use DeepSeek to break down complex queries"""
    try:
        # Initialize Together client with API key
        client = Together(api_key=TOGETHER_API_KEY)
        prompt = f"""Break down this complex query into parts:
        Query: {query}
        
        Return a JSON with this structure:
        {{
            "sub_queries": [
                {{"query": "sub-question 1", "order": 1}},
                {{"query": "sub-question 2", "order": 2}}
            ],
            "execution_type": "sequential" or "parallel",
            "dependencies": [] // for sequential queries
        }}"""

        response = client.chat.completions.create(
            model=TOGETHER_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract and parse JSON from response
        result = response.choices[0].message.content
        
        # Clean and parse JSON similar to complexity analyzer
        json_str = result
        if '```json' in result:
            json_str = result.split('```json')[1].split('```')[0]
        elif '```' in result:
            json_str = result.split('```')[1].split('```')[0]
            
        json_str = json_str.strip()
        parsed_result = json.loads(json_str)
        
        # Validate structure
        if not isinstance(parsed_result, dict) or 'sub_queries' not in parsed_result or 'execution_type' not in parsed_result:
            raise ValueError("Invalid response format")
            
        return parsed_result
        
    except Exception as e:
        st.error(f"Error in query decomposition: {str(e)}")
        return {
            "sub_queries": [{"query": query, "order": 1}],
            "execution_type": "sequential",
            "dependencies": []
        }

def process_query_parts(decomposed_query: dict) -> str:
    """Process sub-queries based on execution type"""
    try:
        results = []
        
        if decomposed_query['execution_type'] == 'sequential':
            # Process queries in order
            for sub_query in sorted(decomposed_query['sub_queries'], key=lambda x: x['order']):
                query_type = determine_query_type(sub_query['query'])
                st.write(f"🔍 Processing Sub-Query: {sub_query['query']}")
                st.write(f"📊 Query Type: {query_type}")
                
                sql_answer = None
                pdf_answer = None
                
                if query_type in ['SQL', 'BOTH']:
                    sql_answer = get_sql_answer(sub_query['query'])
                if query_type in ['PDF', 'BOTH']:
                    pdf_answer = get_pdf_answer(sub_query['query'])
                
                result = get_final_answer(sub_query['query'], sql_answer, pdf_answer)
                results.append(result)
                
        else:  # parallel
            # Process all queries simultaneously
            for sub_query in decomposed_query['sub_queries']:
                query_type = determine_query_type(sub_query['query'])
                st.write(f"🔍 Processing Sub-Query: {sub_query['query']}")
                st.write(f"📊 Query Type: {query_type}")
                
                sql_answer = None
                pdf_answer = None
                
                if query_type in ['SQL', 'BOTH']:
                    sql_answer = get_sql_answer(sub_query['query'])
                if query_type in ['PDF', 'BOTH']:
                    pdf_answer = get_pdf_answer(sub_query['query'])
                
                result = get_final_answer(sub_query['query'], sql_answer, pdf_answer)
                results.append(result)
        
        # Combine results
        combined_results = "\n".join(results)
        return get_final_combined_answer(combined_results)
    except Exception as e:
        st.error(f"Error processing query parts: {str(e)}")
        return "Error processing query parts"

def get_final_combined_answer(results: str) -> str:
    """Generate final combined answer from all results"""
    try:
        prompt = f"""Combine these results into a clear, coherent answer:
        Results: {results}
        
        Provide a natural, flowing response that addresses all parts of the original question."""
        
        return st.session_state['llm_wrapper'](prompt)
    except Exception as e:
        st.error(f"Error combining results: {str(e)}")
        return "Error generating final answer"

def main():
    st.title("🏭 Warehouse Data Analysis Assistant")
    
    # Add clear button
    if st.button("Clear Processed Files"):
        st.session_state['pdf_store'] = {}
        st.session_state['dataframes'] = {}
        st.session_state.pop('schema_content', None)
        st.success("Cleared all processed files")
    
    # Initialize session state
    if 'pdf_store' not in st.session_state:
        st.session_state['pdf_store'] = {}
    if 'dataframes' not in st.session_state:
        st.session_state['dataframes'] = {}
    
    # Combined file upload section
    uploaded_files = st.file_uploader(
        "Upload files (schema.txt required)",
        type=['txt', 'csv', 'pdf'],
        accept_multiple_files=True,
        help="Please ensure schema.txt is included in your uploads"
    )
    
    # Check for schema.txt in uploads
    schema_file = next((f for f in uploaded_files if f.name == 'schema.txt'), None)
    if schema_file:
        if 'schema_content' not in st.session_state:
            schema_content = schema_file.getvalue().decode()
            st.session_state['schema_content'] = schema_content
            st.success("✅ Schema file loaded successfully")
    
    if uploaded_files and not schema_file and 'schema_content' not in st.session_state:
        st.error("Please include schema.txt in your uploads")
        st.stop()
    
    # Process other files if schema exists
    if 'schema_content' in st.session_state:
        # Handle PDF files
        pdf_files = [f for f in uploaded_files if f.name.endswith('.pdf')]
        if pdf_files:
            handwritten_files = st.multiselect(
                "Select handwritten PDFs",
                [f.name for f in pdf_files],
                help="Choose which PDF files contain handwritten content"
            )
        
        # Process all uploaded files
        for file in uploaded_files:
            if file.name == 'schema.txt':
                continue  # Skip schema file as it's already processed
                
            if file.name.endswith('.csv'):
                if file.name not in st.session_state['dataframes']:
                    df = pd.read_csv(file)
                    st.session_state['dataframes'][file.name.replace('.csv', '').lower()] = df
                    st.success(f"✅ CSV Processed: {file.name}")
            
            elif file.name.endswith('.pdf'):
                needs_processing = (
                    file.name not in st.session_state['pdf_store'] or 
                    not st.session_state['pdf_store'][file.name]
                )
                
                if needs_processing:
                    st.session_state['pdf_store'][file.name] = {}
                    
                    # Process with appropriate method based on handwritten selection
                    if file.name in handwritten_files:
                        json_output = process_pdf_with_llmwhisperer(file)
                        if json_output:
                            st.session_state['pdf_store'][file.name]['llmwhisperer'] = json_output
                            st.success(f"✅ Handwritten PDF Processed: {file.name}")
                    else:
                        json_output = process_pdf_with_extractuous(file)
                        if json_output:
                            st.session_state['pdf_store'][file.name]['extractuous'] = json_output
                            st.success(f"✅ Machine-printed PDF Processed: {file.name}")

    # Query input - only show if schema is loaded
    if 'schema_content' in st.session_state:
        user_query = st.text_input("❓ What would you like to know?")
        
        if user_query:
            # Initialize final_answer
            final_answer = None
            
            # Step 1: Analyze complexity
            st.write("🤔 Analyzing query complexity...")
            complexity_analysis = analyze_query_complexity(user_query)
            st.write(f"📊 Complexity Analysis: {complexity_analysis['reason']}")
            
            try:
                if complexity_analysis['is_complex']:
                    # Step 2: Decompose complex query
                    st.write("🔄 Breaking down complex query...")
                    decomposed = decompose_complex_query(user_query)
                    
                    if decomposed:  # Add null check
                        st.write("📋 Query Breakdown:", decomposed.get('execution_type', 'SEQUENTIAL').upper())
                        # Step 3: Process parts
                        final_answer = process_query_parts(decomposed)
                    else:
                        # Fallback to simple query processing if decomposition fails
                        query_type = determine_query_type(user_query)
                        st.write(f"🔍 Query Type: {query_type}")
                        
                        sql_answer = None
                        pdf_answer = None
                        
                        if query_type in ['SQL', 'BOTH']:
                            sql_answer = get_sql_answer(user_query)
                        if query_type in ['PDF', 'BOTH']:
                            pdf_answer = get_pdf_answer(user_query)
                        
                        final_answer = get_final_answer(user_query, sql_answer, pdf_answer)
                else:
                    # Process simple query
                    query_type = determine_query_type(user_query)
                    st.write(f"🔍 Query Type: {query_type}")
                    
                    sql_answer = None
                    pdf_answer = None
                    
                    if query_type in ['SQL', 'BOTH']:
                        sql_answer = get_sql_answer(user_query)
                    if query_type in ['PDF', 'BOTH']:
                        pdf_answer = get_pdf_answer(user_query)
                    
                    final_answer = get_final_answer(user_query, sql_answer, pdf_answer)
                
                # Display final answer if we have one
                if final_answer:
                    st.write("💡 Final Answer:", final_answer)
                else:
                    st.error("Unable to generate an answer. Please try rephrasing your question.")
                    
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.error("Please try rephrasing your question or contact support if the issue persists.")

if __name__ == "__main__":
    main()
