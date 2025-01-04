
import os
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv


load_dotenv()

parser = LlamaParse(result_type="markdown", verbose=True)

file_name = "ssi-page-5.pdf"
extra_info = {"file_name": file_name}

with open(f"../../sample_docs/{file_name}", "rb") as file_to_parse:
    # must provide extra_info with file_name key when passing file object
    documents = parser.load_data(file_to_parse, extra_info=extra_info)
    # to manually check the output uncomment the below
    # print(documents[0].text)

# index the parsed documents
index = VectorStoreIndex.from_documents(documents)

# generate a query engine for the index
query_engine = index.as_query_engine()

# provide the query and output the results
query = "what are the principles of SSI?"
response = query_engine.query(query)
print(response)

