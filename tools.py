# tools
import os
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from utils import *

# llamaindex
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, SummaryIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.core.tools import FunctionTool,QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition

db_folder = './mixed_data' + '/db'
os.makedirs(db_folder, exist_ok=True)
try:
    os.remove(db_folder + '/vs')
except:
    pass

class DocumentToolsGenerator:
    """
    Document processing and tool generation for vector search and summarization.
    """

    def __init__(self, file_path: str):
        """
        Initializes the DocumentToolsGenerator with the given file path and document ID.

        Args:
            file_path (str): The path to the document file to be processed.
            doc_id (str): The identifier for the document.
        """
        self.file_path = file_path
        self.doc_id = file_path
        self.vector_index = None

    def data_ingestion(self, chunk_size: int = 1024, chunk_overlap: int = 64) -> List[BaseNode]:
        """
        Loads and splits the document into chunks for processing.

        Args:
            chunk_size (int): The size of each chunk. Default is 1024.
            chunk_overlap (int): The overlap between chunks. Default is 64.

        Returns:
            List[BaseNode]: A list of document nodes created from the chunks.
        """
        documents = SimpleDirectoryReader(input_files=[self.file_path]).load_data()
        sentence_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = sentence_splitter.get_nodes_from_documents(documents=documents)

        return nodes

    def vector_query(self, query: str) -> str:
        """
        Performs a vector search over the entire document using the specified query.

        Args:
            query (str): The query string to be embedded for the search.

        Returns:
            str: The search response.
        """
        # Import the necessary components
        # from llama_index.core.postprocessor import SentenceTransformerRerank

        # # Initialize the rerank model
        # rerank = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=10)

        # Configure the query engine to search over the entire document
        query_engine = self.vector_index.as_query_engine(
            response_mode='refine',
        )

        # Execute the query
        response = query_engine.query(query)

        # Return the response or handle empty response
        return response

    def tool_generator(self, nodes: List[BaseNode], vector_store_path: str = db_folder, db_name: str = 'vs') -> Tuple[FunctionTool, QueryEngineTool, FunctionTool]:
        """
        Generates and returns tools for vector search, document summarization, and file saving.

        Args:
            nodes (List[BaseNode]): The list of nodes generated from the document.
            vector_store_path (str): The path to store the vector index. Default is './data/db'.
            db_name (str): The name of the vector store database. Default is 'vs'.

        Returns:
            Tuple[FunctionTool, QueryEngineTool, FunctionTool]: A tuple containing the vector query tool, summary query tool, and file saving tool.
        """
        if not os.path.exists(db_name):
            self.vector_index = VectorStoreIndex(nodes=nodes)
            self.vector_index.storage_context.vector_store.persist(persist_path=f'{vector_store_path}/{db_name}')
        else:
            self.vector_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=f'{vector_store_path}'))

        summary_index = SummaryIndex(nodes=nodes)

        vector_query_tool = FunctionTool.from_defaults(
            name="vector_search_tool", 
            fn=self.vector_query,
            description="Useful for searching specific facts in a document"
        )

        summary_query = summary_index.as_query_engine(response_mode="tree_summarize")
        summary_query_tool = QueryEngineTool.from_defaults(
            name="summary_query_tool",
            query_engine=summary_query,
            description="Useful for summarizing an entire video. DO NOT USE if you have specified questions over the videos."
        )

        return vector_query_tool, summary_query_tool
