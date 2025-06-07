# from flask import Flask, request, render_template, redirect, url_for, flash
# import os
# import pandas as pd
# import pickle
# import numpy as np
# from datetime import datetime
# import logging
# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, Document
# from llama_index.llms.openai import OpenAI
# import openai
# from dotenv import load_dotenv
# import json

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables from .env file
# load_dotenv()

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['DATA_FOLDER'] = 'data'
# app.config['PKL_FILE'] = 'learned_data.pkl'
# app.config['INDEX_STORAGE'] = 'index_storage'
# app.config['METADATA_FILE'] = 'data_metadata.json'
# app.config['ALLOWED_EXTENSIONS'] = {'xlsx'}
# app.secret_key = 'your_secret_key'

# # Ensure directories exist
# for folder in [app.config['UPLOAD_FOLDER'], app.config['DATA_FOLDER'], app.config['INDEX_STORAGE']]:
#     if not os.path.exists(folder):
#         os.makedirs(folder)

# # Set up LlamaIndex with OpenAI
# openai.api_key = os.getenv('OPENAI_API_KEY')
# Settings.llm = OpenAI(model="gpt-4", temperature=0.7)

# class DataManager:
#     """Manages persistent data storage and learning"""
    
#     def __init__(self, pkl_file, metadata_file):
#         self.pkl_file = pkl_file
#         self.metadata_file = metadata_file
#         self.data = self.load_learned_data()
#         self.metadata = self.load_metadata()
    
#     def load_learned_data(self):
#         """Load the learned data from pkl file"""
#         if os.path.exists(self.pkl_file):
#             try:
#                 with open(self.pkl_file, 'rb') as f:
#                     return pickle.load(f)
#             except Exception as e:
#                 logger.error(f"Error loading pkl file: {e}")
#                 return pd.DataFrame()
#         return pd.DataFrame()
    
#     def load_metadata(self):
#         """Load metadata about processed files"""
#         if os.path.exists(self.metadata_file):
#             try:
#                 with open(self.metadata_file, 'r') as f:
#                     return json.load(f)
#             except Exception as e:
#                 logger.error(f"Error loading metadata: {e}")
#                 return {}
#         return {}
    
#     def save_learned_data(self):
#         """Save the learned data to pkl file"""
#         try:
#             with open(self.pkl_file, 'wb') as f:
#                 pickle.dump(self.data, f)
#             logger.info(f"Saved learned data with {len(self.data)} records")
#         except Exception as e:
#             logger.error(f"Error saving pkl file: {e}")
    
#     def save_metadata(self):
#         """Save metadata about processed files"""
#         try:
#             with open(self.metadata_file, 'w') as f:
#                 json.dump(self.metadata, f, indent=2)
#         except Exception as e:
#             logger.error(f"Error saving metadata: {e}")
    
#     def add_new_data(self, df, filename):
#         """Add new data to the existing learned data"""
#         try:
#             # Add source file info to the dataframe
#             df['_source_file'] = filename
#             df['_added_date'] = datetime.now().isoformat()
            
#             if self.data.empty:
#                 self.data = df.copy()
#             else:
#                 # Combine with existing data
#                 self.data = pd.concat([self.data, df], ignore_index=True, sort=False)
            
#             # Update metadata
#             self.metadata[filename] = {
#                 'added_date': datetime.now().isoformat(),
#                 'rows': len(df),
#                 'columns': list(df.columns)
#             }
            
#             # Save both data and metadata
#             self.save_learned_data()
#             self.save_metadata()
            
#             logger.info(f"Added {len(df)} records from {filename}. Total records: {len(self.data)}")
#             return True
#         except Exception as e:
#             logger.error(f"Error adding new data: {e}")
#             return False
    
#     def get_data_summary(self):
#         """Get summary of learned data"""
#         if self.data.empty:
#             return "No data learned yet."
        
#         summary = {
#             'total_records': len(self.data),
#             'columns': list(self.data.columns),
#             'files_processed': len(self.metadata),
#             'date_range': {
#                 'first_added': min([meta['added_date'] for meta in self.metadata.values()]) if self.metadata else None,
#                 'last_added': max([meta['added_date'] for meta in self.metadata.values()]) if self.metadata else None
#             }
#         }
#         return summary

# # Initialize data manager
# data_manager = DataManager(app.config['PKL_FILE'], app.config['METADATA_FILE'])

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def safe_column_detection(df):
#     """Safely detect columns with proper error handling"""
#     try:
#         state_col = None
#         product_col = None
#         sales_col = None
#         quantity_col = None
        
#         # Convert all column names to string first to avoid the 'int' object error
#         column_mapping = {}
#         for col in df.columns:
#             col_str = str(col).lower()
#             column_mapping[col] = col_str
        
#         # Detect columns based on string content
#         for original_col, col_lower in column_mapping.items():
#             if any(keyword in col_lower for keyword in ['state', 'region', 'location', 'city']):
#                 state_col = original_col
#             elif any(keyword in col_lower for keyword in ['product', 'item', 'sku', 'name']):
#                 product_col = original_col
#             elif any(keyword in col_lower for keyword in ['sales', 'revenue', 'amount', 'value']):
#                 sales_col = original_col
#             elif any(keyword in col_lower for keyword in ['quantity', 'qty', 'units', 'count']):
#                 quantity_col = original_col
        
#         return {
#             'state': state_col,
#             'product': product_col,
#             'sales': sales_col,
#             'quantity': quantity_col,
#             'detected_columns': list(column_mapping.keys())
#         }
#     except Exception as e:
#         logger.error(f"Error in column detection: {e}")
#         return None

# def analyze_top_products_by_state(df):
#     """Analyze data to find top performing products by state"""
#     try:
#         cols = safe_column_detection(df)
#         if not cols:
#             return "Error: Could not safely detect columns"
        
#         state_col = cols['state']
#         product_col = cols['product']
#         sales_col = cols['sales']
#         quantity_col = cols['quantity']
        
#         # Use sales if available, otherwise quantity, otherwise first numeric column
#         metric_col = sales_col or quantity_col
#         if not metric_col:
#             # Find first numeric column
#             numeric_cols = df.select_dtypes(include=[np.number]).columns
#             if len(numeric_cols) > 0:
#                 metric_col = numeric_cols[0]
#             else:
#                 return "Error: No numeric columns found for analysis"
        
#         if not all([state_col, product_col, metric_col]):
#             return f"Error: Could not identify required columns. Detected: State={state_col}, Product={product_col}, Metric={metric_col}"
        
#         # Clean data - remove null values
#         analysis_df = df[[state_col, product_col, metric_col]].dropna()
        
#         # Convert metric column to numeric
#         analysis_df[metric_col] = pd.to_numeric(analysis_df[metric_col], errors='coerce')
#         analysis_df = analysis_df.dropna()
        
#         if analysis_df.empty:
#             return "Error: No valid data found after cleaning"
        
#         # Group by state and product, sum the metric
#         grouped = analysis_df.groupby([state_col, product_col])[metric_col].sum().reset_index()
        
#         # Find top product per state
#         top_products = grouped.loc[grouped.groupby(state_col)[metric_col].idxmax()]
        
#         return {
#             'data': top_products,
#             'columns': {
#                 'state': state_col,
#                 'product': product_col,
#                 'metric': metric_col
#             }
#         }
#     except Exception as e:
#         logger.error(f"Error in analysis: {e}")
#         return f"Error analyzing data: {str(e)}"

# def create_vector_index_from_dataframe(df):
#     """Create a vector index from dataframe for semantic search"""
#     try:
#         documents = []
        
#         # Create documents from each row
#         for idx, row in df.iterrows():
#             # Convert row to text representation
#             text_content = f"Record {idx}: " + ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
#             doc = Document(text=text_content, metadata={"row_id": idx})
#             documents.append(doc)
        
#         # Create index
#         index = VectorStoreIndex.from_documents(documents)
        
#         # Persist the index
#         index.storage_context.persist(persist_dir=app.config['INDEX_STORAGE'])
        
#         return index
#     except Exception as e:
#         logger.error(f"Error creating vector index: {e}")
#         return None

# def load_or_create_index():
#     """Load existing index or create new one from learned data"""
#     try:
#         # Try to load existing index
#         if os.path.exists(os.path.join(app.config['INDEX_STORAGE'], 'docstore.json')):
#             from llama_index.core import StorageContext, load_index_from_storage
#             storage_context = StorageContext.from_defaults(persist_dir=app.config['INDEX_STORAGE'])
#             index = load_index_from_storage(storage_context)
#             logger.info("Loaded existing vector index")
#             return index
#         else:
#             # Create new index from learned data
#             if not data_manager.data.empty:
#                 index = create_vector_index_from_dataframe(data_manager.data)
#                 if index:
#                     logger.info("Created new vector index from learned data")
#                     return index
#             logger.warning("No data available to create index")
#             return None
#     except Exception as e:
#         logger.error(f"Error loading/creating index: {e}")
#         return None

# def format_analysis_result(result, question):
#     """Format analysis result into HTML"""
#     try:
#         if isinstance(result, str):
#             return f"<div class='error'>{result}</div>"
        
#         if isinstance(result, dict) and 'data' in result:
#             df = result['data']
#             cols = result['columns']
            
#             html = f"""
#             <div style="font-family: Arial, sans-serif; padding: 20px;">
#                 <h2 style="color: #2c3e50;">Analysis Results</h2>
#                 <p><strong>Question:</strong> {question}</p>
#                 <p><strong>Analysis:</strong> Top performing products by {cols['state']} based on {cols['metric']}</p>
                
#                 <table style="width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
#                     <thead>
#                         <tr style="background-color: #3498db; color: white;">
#                             <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">{cols['state']}</th>
#                             <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">Top Product</th>
#                             <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">{cols['metric']}</th>
#                         </tr>
#                     </thead>
#                     <tbody>
#             """
            
#             for _, row in df.iterrows():
#                 html += f"""
#                         <tr style="border-bottom: 1px solid #eee;">
#                             <td style="padding: 10px; border: 1px solid #ddd;">{row[cols['state']]}</td>
#                             <td style="padding: 10px; border: 1px solid #ddd;">{row[cols['product']]}</td>
#                             <td style="padding: 10px; border: 1px solid #ddd;">{row[cols['metric']]:,.2f}</td>
#                         </tr>
#                 """
            
#             html += """
#                     </tbody>
#                 </table>
#                 <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
#                     <small><em>Data analyzed from learned dataset with {} total records</em></small>
#                 </div>
#             </div>
#             """.format(len(data_manager.data))
            
#             return html
        
#         return f"<div class='error'>Unexpected result format</div>"
        
#     except Exception as e:
#         logger.error(f"Error formatting result: {e}")
#         return f"<div class='error'>Error formatting result: {str(e)}</div>"

# def ask_openai_with_context(question, context_data=None):
#     """Ask OpenAI with context from learned data"""
#     try:
#         # Prepare context
#         context = ""
#         if context_data is None and not data_manager.data.empty:
#             # Use summary of learned data as context
#             summary = data_manager.get_data_summary()
#             context = f"Available data summary: {summary}"
#         elif context_data:
#             context = str(context_data)
        
#         messages = [
#             {
#                 "role": "system", 
#                 "content": """You are a business data analyst. You have access to a learned dataset from uploaded Excel files. 
#                 Provide your answer in HTML format with proper styling to make it attractive and presentable. 
#                 You can include tables, charts descriptions, and insights. 
#                 If you don't have sufficient data, provide general business insights."""
#             },
#             {
#                 "role": "user", 
#                 "content": f"Context: {context}\n\nQuestion: {question}"
#             }
#         ]
        
#         response = openai.chat.completions.create(
#             model="gpt-4",
#             messages=messages,
#             max_tokens=4000,
#             temperature=0.7,
#         )
        
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         logger.error(f"Error with OpenAI: {e}")
#         return f"<div class='error'>Error processing with AI: {str(e)}</div>"

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         files = request.files.getlist('files')
#         if not files or files[0].filename == '':
#             flash('No files selected')
#             return redirect(request.url)
        
#         processed_files = 0
#         errors = []
        
#         for file in files:
#             if file and allowed_file(file.filename):
#                 try:
#                     # Read Excel file directly
#                     df = pd.read_excel(file, engine='openpyxl')
                    
#                     # Add to learned data
#                     if data_manager.add_new_data(df, file.filename):
#                         processed_files += 1
#                     else:
#                         errors.append(f"Failed to process {file.filename}")
                        
#                 except Exception as e:
#                     errors.append(f"Error reading {file.filename}: {str(e)}")
#             else:
#                 errors.append(f"Invalid file format: {file.filename}")
        
#         # Update vector index with new data
#         if processed_files > 0:
#             try:
#                 create_vector_index_from_dataframe(data_manager.data)
#                 flash(f'Successfully processed {processed_files} files and updated knowledge base!')
#             except Exception as e:
#                 flash(f'Files processed but index update failed: {str(e)}')
        
#         if errors:
#             for error in errors:
#                 flash(error)
        
#         return render_template('ask_question.html', data_summary=data_manager.get_data_summary())
    
#     # Show current data summary on GET request
#     summary = data_manager.get_data_summary()
#     return render_template('index.html', data_summary=summary)

# @app.route('/ask', methods=['POST'])
# def ask():
#     question = request.form.get('question')
    
#     if not question or question.strip() == '':
#         flash('Please enter a valid question.')
#         return redirect(url_for('index'))
    
#     # Check if we have learned data
#     if data_manager.data.empty:
#         flash('Please upload some files first to build the knowledge base.')
#         return redirect(url_for('index'))
    
#     try:
#         # Handle specific analysis questions
#         if any(phrase in question.lower() for phrase in ['top performing product', 'best product', 'highest selling']):
#             result = analyze_top_products_by_state(data_manager.data)
#             answer = format_analysis_result(result, question)
#             return render_template('answer.html', question=question, answer=answer)
        
#         # For other questions, use vector search + OpenAI
#         index = load_or_create_index()
#         if index:
#             query_engine = index.as_query_engine(similarity_top_k=5)
#             response = query_engine.query(question)
            
#             # Enhance with OpenAI
#             enhanced_answer = ask_openai_with_context(question, str(response))
#             return render_template('answer.html', question=question, answer=enhanced_answer)
#         else:
#             # Fallback to direct OpenAI with data summary
#             answer = ask_openai_with_context(question)
#             return render_template('answer.html', question=question, answer=answer)
            
#     except Exception as e:
#         logger.error(f"Error processing question: {e}")
#         flash(f"Error processing question: {str(e)}")
#         return redirect(url_for('index'))

# @app.route('/data-summary')
# def data_summary():
#     """Route to view current data summary"""
#     summary = data_manager.get_data_summary()
#     return render_template('data_summary.html', summary=summary)

# @app.route('/reset-data', methods=['POST'])
# def reset_data():
#     """Route to reset all learned data"""
#     try:
#         # Remove pkl file
#         if os.path.exists(app.config['PKL_FILE']):
#             os.remove(app.config['PKL_FILE'])
        
#         # Remove metadata file
#         if os.path.exists(app.config['METADATA_FILE']):
#             os.remove(app.config['METADATA_FILE'])
        
#         # Remove index storage
#         import shutil
#         if os.path.exists(app.config['INDEX_STORAGE']):
#             shutil.rmtree(app.config['INDEX_STORAGE'])
#             os.makedirs(app.config['INDEX_STORAGE'])
        
#         # Reset data manager
#         global data_manager
#         data_manager = DataManager(app.config['PKL_FILE'], app.config['METADATA_FILE'])
        
#         flash('All learned data has been reset successfully!')
#     except Exception as e:
#         flash(f'Error resetting data: {str(e)}')
    
#     return redirect(url_for('index'))

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import os
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import logging
import json
import shutil
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# --- LlamaIndex & Google AI Imports ---
from llama_index.core import VectorStoreIndex, Settings, Document, StorageContext, load_index_from_storage
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import google.generativeai as genai

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Flask App Configuration ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FOLDER'] = 'data'
app.config['PKL_FILE'] = os.path.join(app.config['DATA_FOLDER'], 'learned_data.pkl')
app.config['METADATA_FILE'] = os.path.join(app.config['DATA_FOLDER'], 'data_metadata.json')
app.config['INDEX_STORAGE'] = 'index_storage'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'xls', 'csv'}
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'a-strong-default-secret-key')

# Ensure all necessary directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['DATA_FOLDER'], app.config['INDEX_STORAGE']]:
    os.makedirs(folder, exist_ok=True)

# --- AI Configuration (LlamaIndex Settings) ---
def configure_ai():
    """Configure AI services with proper error handling"""
    try:
        # Get API key from environment - try both possible names
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("Neither GOOGLE_API_KEY nor GEMINI_API_KEY environment variable is set")
            return False
        
        # Remove any quotes or whitespace
        api_key = api_key.strip().strip('"').strip("'")
        
        if not api_key:
            logger.error("API key is empty after cleaning")
            return False
        
        logger.info(f"Configuring AI with API key: {api_key[:10]}...")
        
        # Configure the underlying Google SDK
        genai.configure(api_key=api_key)

        # Set the LLM to Google GenAI for generation and chat
        Settings.llm = GoogleGenAI(
            model_name="gemini-1.5-pro-latest", 
            temperature=0.7,
            api_key=api_key
        )
        
        # Set the embedding model to Google's embedding model
        Settings.embed_model = GoogleGenAIEmbedding(
            model_name="text-embedding-004",
            api_key=api_key,
            embed_batch_size=100
        )
        
        logger.info("Successfully configured Google AI services for LlamaIndex.")
        return True

    except Exception as e:
        logger.error(f"FATAL: Failed to configure Google AI. Error: {e}")
        Settings.llm = None
        Settings.embed_model = None
        return False

# Configure AI services
ai_configured = configure_ai()

class DataManager:
    """Manages persistent data storage and learning"""
    
    def __init__(self, pkl_file, metadata_file):
        self.pkl_file = pkl_file
        self.metadata_file = metadata_file
        self.data = self.load_learned_data()
        self.metadata = self.load_metadata()
    
    def load_learned_data(self):
        if os.path.exists(self.pkl_file):
            try:
                with open(self.pkl_file, 'rb') as f: 
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading pkl file: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def load_metadata(self):
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f: 
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                # If metadata is corrupted, create backup and start fresh
                if os.path.exists(self.metadata_file):
                    backup_file = f"{self.metadata_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.copy2(self.metadata_file, backup_file)
                    logger.info(f"Corrupted metadata backed up to {backup_file}")
                return {}
        return {}
    
    def save_learned_data(self):
        try:
            # Create backup before saving
            if os.path.exists(self.pkl_file):
                backup_file = f"{self.pkl_file}.backup"
                shutil.copy2(self.pkl_file, backup_file)
            
            with open(self.pkl_file, 'wb') as f: 
                pickle.dump(self.data, f)
            logger.info(f"Saved learned data with {len(self.data)} records")
            return True
        except Exception as e:
            logger.error(f"Error saving pkl file: {e}")
            return False
    
    def save_metadata(self):
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f: 
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            return False
    
    def add_new_data(self, df, filename):
        try:
            # Clean the dataframe
            df = df.copy()
            
            # Reset index to ensure clean integer indexing
            df = df.reset_index(drop=True)
            
            # Add metadata columns
            df['_source_file'] = filename
            now_iso = datetime.now().isoformat()
            df['_added_date'] = now_iso
            
            # Handle NaN values
            df = df.fillna('')
            
            # Convert problematic columns to string to avoid type issues
            for col in df.columns:
                if col not in ['_source_file', '_added_date']:
                    try:
                        # Convert to string, handling any special cases
                        df[col] = df[col].astype(str)
                    except Exception as e:
                        logger.warning(f"Could not convert column {col} to string: {e}")
                        df[col] = df[col].fillna('').astype(str)
            
            if self.data.empty:
                self.data = df.copy()
            else:
                # Reset index for both dataframes before concatenation
                self.data = self.data.reset_index(drop=True)
                df = df.reset_index(drop=True)
                self.data = pd.concat([self.data, df], ignore_index=True)
            
            # Ensure final dataframe has clean index
            self.data = self.data.reset_index(drop=True)
            
            self.metadata[filename] = {
                'added_date': now_iso,
                'rows': len(df),
                'columns': [str(col) for col in df.columns],  # Ensure column names are strings
                'file_size': os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], filename)) if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)) else 0
            }
            
            if self.save_learned_data() and self.save_metadata():
                logger.info(f"Added {len(df)} records from {filename}. Total records: {len(self.data)}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error adding new data: {e}")
            return False
    
    def get_data_summary(self):
        """Fixed method with proper error handling"""
        if self.data.empty: 
            return {
                'total_records': 0,
                'columns': [],
                'files_processed': 0,
                'date_range': {'first_added': None, 'last_added': None},
                'sample_data': None
            }
        
        try:
            all_dates = [meta.get('added_date') for meta in self.metadata.values() if meta.get('added_date')]
            
            # Get sample data (first 3 rows)
            sample_data = None
            if len(self.data) > 0:
                sample_df = self.data.head(3).copy()
                
                # Fix: Ensure all column names are strings and filter out internal columns
                display_columns = []
                for col in sample_df.columns:
                    col_str = str(col)
                    if not col_str.startswith('_'):
                        display_columns.append(col)
                
                if display_columns:
                    try:
                        # Use .loc to safely select columns and convert to dict
                        sample_data = sample_df.loc[:, display_columns].fillna('').to_dict('records')
                    except Exception as e:
                        logger.warning(f"Error creating sample data: {e}")
                        # Fallback: just get the first few non-internal columns
                        try:
                            safe_columns = [col for col in sample_df.columns if not str(col).startswith('_')][:5]  # First 5 columns
                            if safe_columns:
                                sample_data = sample_df[safe_columns].fillna('').head(3).to_dict('records')
                        except:
                            sample_data = None

            # Fix: Ensure all column names are converted to strings and filter safely
            all_columns = []
            for col in self.data.columns:
                col_str = str(col)
                if not col_str.startswith('_'):
                    all_columns.append(col_str)

            return {
                'total_records': len(self.data),
                'columns': all_columns,
                'files_processed': len(self.metadata),
                'date_range': {
                    'first_added': min(all_dates) if all_dates else None,
                    'last_added': max(all_dates) if all_dates else None
                },
                'sample_data': sample_data,
                'file_details': self.metadata
            }
        except Exception as e:
            logger.error(f"Error in get_data_summary: {e}")
            return {
                'total_records': len(self.data) if not self.data.empty else 0,
                'columns': [],
                'files_processed': len(self.metadata),
                'date_range': {'first_added': None, 'last_added': None},
                'sample_data': None,
                'file_details': self.metadata
            }

# --- Global Instances ---
data_manager = DataManager(app.config['PKL_FILE'], app.config['METADATA_FILE'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def read_file(file_path, filename):
    """Read various file formats"""
    try:
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        if file_extension in ['xlsx', 'xls']:
            return pd.read_excel(file_path, engine='openpyxl' if file_extension == 'xlsx' else None)
        elif file_extension == 'csv':
            # Try different encodings for CSV
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not read CSV file with any supported encoding")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}")
        raise

def create_vector_index_from_dataframe(df):
    """Create vector index from dataframe with improved error handling"""
    try:
        if not ai_configured or Settings.embed_model is None:
            logger.error("Embedding model not configured. Cannot create vector index.")
            return None
            
        # Convert all dataframe content to string to prevent embedding errors
        df_str = df.astype(str)
        
        # Create more meaningful document content
        documents = []
        for idx, row in df_str.iterrows():
            # Skip internal columns - ensure column names are strings
            data_cols = {str(k): v for k, v in row.items() if not str(k).startswith('_')}
            
            if data_cols:  # Only create document if there's actual data
                text_content = "Record data: " + ", ".join([f"{col}: {val}" for col, val in data_cols.items() if val and str(val) != 'nan'])
                
                documents.append(Document(
                    text=text_content,
                    metadata={
                        "row_id": idx, 
                        "source_file": row.get('_source_file', 'N/A'),
                        "added_date": row.get('_added_date', 'N/A')
                    }
                ))
        
        if not documents:
            logger.warning("No documents were generated from the dataframe to be indexed.")
            return None
            
        logger.info(f"Creating vector index with {len(documents)} documents...")
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=app.config['INDEX_STORAGE'])
        logger.info(f"Successfully created and persisted vector index with {len(documents)} documents.")
        return index
        
    except Exception as e:
        logger.error(f"Error creating vector index: {e}", exc_info=True)
        return None

def load_or_create_index():
    """Load existing index or create new one"""
    try:
        if not ai_configured or Settings.embed_model is None:
            logger.error("Embedding model not configured. Cannot load/create index.")
            return None
            
        storage_path = app.config['INDEX_STORAGE']
        
        # Try to load existing index
        if os.path.exists(os.path.join(storage_path, 'docstore.json')):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=storage_path)
                index = load_index_from_storage(storage_context)
                logger.info("Loaded existing vector index from storage.")
                return index
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new one.")
        
        # Create new index if we have data
        if not data_manager.data.empty:
            logger.info("Creating a new vector index from learned data.")
            return create_vector_index_from_dataframe(data_manager.data)
        else:
            logger.warning("No data available to create an index.")
            return None
            
    except Exception as e:
        logger.error(f"Error loading/creating index: {e}", exc_info=True)
        return None

def ask_gemini_with_context(question, context_data=None):
    """Ask Gemini with context from learned data"""
    try:
        if not ai_configured or Settings.llm is None:
            return "<div class='alert alert-danger'>AI service is not properly configured. Please check your GOOGLE_API_KEY.</div>"
        
        # Prepare context
        context = ""
        if context_data is None and not data_manager.data.empty:
            summary = data_manager.get_data_summary()
            context = f"""
            Data Summary:
            - Total Records: {summary['total_records']}
            - Columns: {', '.join(summary['columns'])}
            - Files Processed: {summary['files_processed']}
            - Date Range: {summary['date_range']['first_added']} to {summary['date_range']['last_added']}
            """
            
            # Add sample data if available
            if summary.get('sample_data'):
                context += f"\n\nSample Data (first 3 records):\n{json.dumps(summary['sample_data'], indent=2)}"
                
        elif context_data:
            context = str(context_data)
        
        prompt = f"""
        You are an expert business data analyst. Answer the user's question based on the provided context.
        
        **IMPORTANT**: Format your response as clean, well-structured HTML suitable for web display. Use:
        - <h3> for section headings
        - <table class="table table-striped"> for data tables
        - <ul> and <li> for lists
        - <p> for paragraphs
        - <strong> for emphasis
        - <div class="alert alert-info"> for important notes
        
        **Context Data:**
        {context}

        **User Question:**
        {question}
        
        **Instructions:**
        1. If the context contains specific data, analyze it and provide insights
        2. Create tables or charts when presenting numerical data
        3. If the question cannot be answered with the available data, explain what's missing
        4. Provide actionable business insights when possible
        5. Keep the response professional and easy to understand
        """
        
        # Use the configured LLM through Settings
        response = Settings.llm.complete(prompt)
        
        # Clean the response
        cleaned_response = str(response).strip()
        
        # Remove any markdown code blocks if present
        if cleaned_response.startswith("```html"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
            
        return cleaned_response.strip()
        
    except Exception as e:
        logger.error(f"Error with Gemini API: {e}", exc_info=True)
        return f"""
        <div class='alert alert-danger'>
            <h4>Error Processing Request</h4>
            <p>There was an error processing your request with the AI service.</p>
            <p><strong>Error:</strong> {str(e)}</p>
            <p>Please check your API configuration and try again.</p>
        </div>
        """

# Add favicon route to prevent 404 errors
@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests to prevent 404 errors"""
    return '', 204  # No Content response

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            flash('No files were selected.', 'warning')
            return redirect(request.url)
        
        processed_files, errors = 0, []
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # Save file temporarily
                    filename = file.filename
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    
                    # Read and process file
                    df = read_file(file_path, filename)
                    
                    if df.empty:
                        errors.append(f"File {filename} is empty or could not be read")
                        continue
                    
                    if data_manager.add_new_data(df, filename):
                        processed_files += 1
                        logger.info(f"Successfully processed {filename} with {len(df)} records")
                    else:
                        errors.append(f"Failed to process {filename}")
                        
                except Exception as e:
                    errors.append(f"Error reading {file.filename}: {str(e)}")
                    logger.error(f"Error reading file {file.filename}: {e}", exc_info=True)
                finally:
                    # Clean up temporary file
                    if 'file_path' in locals() and os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except:
                            pass
                            
            elif file.filename:
                errors.append(f"Invalid file format for {file.filename}. Allowed formats: .xlsx, .xls, .csv")
        
        # Update vector index if files were processed
        if processed_files > 0:
            flash(f'Successfully processed {processed_files} file(s). Updating knowledge base...', 'success')
            try:
                if ai_configured:
                    index = create_vector_index_from_dataframe(data_manager.data)
                    if index:
                        flash('Knowledge base updated successfully!', 'success')
                    else:
                        flash('Files processed but knowledge base update failed. You can still ask questions based on data summary.', 'warning')
                else:
                    flash('Files processed but AI features are disabled. Configure GOOGLE_API_KEY to enable AI features.', 'warning')
            except Exception as e:
                logger.error(f"Error updating knowledge base: {e}")
                flash('Files processed but knowledge base update failed. You can still ask questions based on data summary.', 'warning')
        
        # Display errors
        for error in errors: 
            flash(error, 'danger')
        
        return redirect(url_for('index'))
    
    return render_template('index.html', data_summary=data_manager.get_data_summary(), ai_configured=ai_configured)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question')
    if not question or not question.strip():
        flash('Please enter a valid question.', 'warning')
        return redirect(url_for('index'))
    
    if data_manager.data.empty:
        flash('Please upload data files first to build the knowledge base.', 'warning')
        return redirect(url_for('index'))
    
    if not ai_configured:
        flash('AI features are disabled. Please configure your GOOGLE_API_KEY.', 'danger')
        return redirect(url_for('index'))
    
    try:
        # Try to use vector search first
        index = load_or_create_index()
        context_data = None
        
        if index and Settings.llm:
            try:
                query_engine = index.as_query_engine(
                    similarity_top_k=5, 
                    response_mode="compact"
                )
                response = query_engine.query(question)
                context_data = str(response)
                logger.info("Used vector search for context")
            except Exception as e:
                logger.warning(f"Vector search failed, using data summary: {e}")
                context_data = None
        
        # Generate answer using Gemini
        answer = ask_gemini_with_context(question, context_data)
        
        return render_template('answer.html', 
                             question=question, 
                             answer=answer,
                             data_summary=data_manager.get_data_summary())
            
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        flash(f"An unexpected error occurred: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/data-preview')
def data_preview():
    """Show a preview of the loaded data"""
    if data_manager.data.empty:
        flash('No data available. Please upload files first.', 'warning')
        return redirect(url_for('index'))
    
    try:
        # Get first 50 rows and non-internal columns
        preview_data = data_manager.data.head(50).copy()
        
        # Fix: Safely get display columns
        display_columns = []
        for col in preview_data.columns:
            if not str(col).startswith('_'):
                display_columns.append(col)
        
        if display_columns:
            # Use .loc for safe column selection
            preview_data = preview_data.loc[:, display_columns].fillna('')
        
        return render_template('data_preview.html', 
                             data=preview_data.to_html(classes='table table-striped table-hover', table_id='dataTable'),
                             data_summary=data_manager.get_data_summary())
    except Exception as e:
        logger.error(f"Error in data preview: {e}")
        flash(f'Error displaying data preview: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/data-summary')
def data_summary():
    """Alias for data_preview to handle template compatibility"""
    return redirect(url_for('data_preview'))

@app.route('/reset-data', methods=['POST'])
def reset_data():
    try:
        # Remove data and metadata files
        files_to_remove = [app.config['PKL_FILE'], app.config['METADATA_FILE']]
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed {file_path}")
        
        # Remove vector index storage
        if os.path.exists(app.config['INDEX_STORAGE']):
            shutil.rmtree(app.config['INDEX_STORAGE'])
            logger.info(f"Removed index storage directory")
        
        # Recreate directories
        for folder in [app.config['INDEX_STORAGE']]:
            os.makedirs(folder, exist_ok=True)
        
        # Re-initialize the data manager to clear in-memory data
        global data_manager
        data_manager = DataManager(app.config['PKL_FILE'], app.config['METADATA_FILE'])
        
        flash('All learned data and the knowledge base have been successfully reset!', 'success')
        logger.info("Data reset completed successfully")
        
    except Exception as e:
        flash(f'Error resetting data: {str(e)}', 'danger')
        logger.error(f"Error during data reset: {e}", exc_info=True)
    
    return redirect(url_for('index'))

# Improved error handlers that don't require templates
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors without requiring a template"""
    logger.info(f"404 error: {request.url}")
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found.',
        'status_code': 404
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors without requiring a template"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred on the server.',
        'status_code': 500
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({
        'error': 'Unexpected Error',
        'message': 'An unexpected error occurred.',
        'status_code': 500
    }), 500

if __name__ == '__main__':
    # Check configuration and provide helpful feedback
    if not ai_configured:
        logger.warning("AI services not properly configured. Some features may not work.")
        print("\n" + "="*50)
        print("⚠️  WARNING: AI services not properly configured!")
        print("Please ensure your GOOGLE_API_KEY is set correctly in your .env file.")
        print("The app will run but AI features will be disabled.")
        print("="*50 + "\n")
    else:
        logger.info("Application starting with full AI capabilities")
        print("\n" + "="*50)
        print("✅ AI services configured successfully!")
        print("🚀 Starting Flask application...")
        print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)