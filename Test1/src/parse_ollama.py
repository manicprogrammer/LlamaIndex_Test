
# pip install llama-index-embeddings-huggingface
# pip install llama-index-llms-ollama
# pip install llama-index-core llama-parse llama-index-readers-file

import os
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv


load_dotenv()

# set LlamaParse for markdown output and auto_mode only parsing page 8
parser = LlamaParse(
    result_type="markdown", 
    auto_mode=True,
    auto_mode_trigger_on_image_in_page=True,
    auto_mode_trigger_on_table_in_page=True,
    target_pages="8",
    verbose=True
)

file_name = "Amazon_nova_technical_report.pdf"
extra_info = {"file_name": file_name}

with open(f"../../sample_docs/{file_name}", "rb") as file_to_parse:
    # LlamaParse will cache a parsed document 48 hours if the parse parameters are not changed
    # thus not incuring additional parse cost if you run this multiple times for testing purposes
    # see the history tab in the LlamaParse dashboard for the project to confirm that 
    # credits used = 0 for subsequent runs
    # 
    # must provide extra_info with file_name key when passing file object
    documents = parser.load_data(file_to_parse, extra_info=extra_info)
    # to manually check the output uncomment the below
    #print(documents[0].text)

# set the default embeddings and llm so that it doesn't have to be passed around
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="llama3.1:latest", request_timeout=120.0)

# index the parsed documents using the default embedding model
index = VectorStoreIndex.from_documents(documents)

# generate a query engine for the index using the default llm
query_engine = index.as_query_engine()

# provide the query and output the results
query = "What is the latency in seconds for Nova Micro?"
response = query_engine.query(query)
print(response)

