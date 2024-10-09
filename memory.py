import ollama
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import psycopg
from psycopg.rows import dict_row
from models import *
from utils import DB_PARAMS


class MemoryAgent:
    def __init__(self):
        self.client = chromadb.Client()
        # self.system_prompt = {
        #     'You are an AI assistant that has memory of every conversation you have ever had with this user. '
        #     'On every prompt from the user, the system has checked for any relevant messages you have ever had with the user. '
        #     'If any embedded previous conversations are attached, use them for context to responding to the user, '
        #     'if the context is relevant and useful to responding. If the recalled conversations are irrelevant, '
        #     'disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations. '
        #     'Just use any useful data from the previous conversations and respond normally as an intelligent AI assistant.'
        # }
        # self.convo = [{'role': 'system', 'content': self.system_prompt}]
        self.DB_PARAMS = DB_PARAMS

    # def stream_response(self, prompt):
    #     self.convo.append({'role': 'user', 'content': prompt})
    #     response = ''
    #     stream = ollama.chat(model=selected_model, messages=self.convo, stream=True)
    #     print(f'ASSISTANT:')

    #     for chunk in stream:
    #         content = chunk['message']['content']
    #         response += content
    #         print(content, end='', flush=True)

    #     print('\n')
    #     self.store_response(prompt=prompt, response=response)
    #     self.convo.append({'role': 'assistant', 'content': response})

    def connect_db(self):
        conn = psycopg.connect(**self.DB_PARAMS)
        return conn

    def fetch_conversations(self):
        conn = self.connect_db()
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute('SELECT * FROM conversations')
            conversations = cursor.fetchall()
        conn.close()
        return conversations

    def create_vector_db(self, conversations):
        vector_db_name = 'conversations'

        try:
            self.client.delete_collection(name=vector_db_name)
        except ValueError:
            pass

        vector_db = self.client.create_collection(name=vector_db_name)

        for c in conversations:
            serialized_convo = f"prompt: {c['prompt']} response: {c['response']}"
            response = ollama.embeddings(model='nomic-embed-text', prompt=serialized_convo)
            embedding = response['embedding']

            vector_db.add(
                ids=[str(c['id'])],
                embeddings=[embedding],
                documents=[serialized_convo]
            )

    def retrieve_embeddings(self, prompt):
        response = ollama.embeddings(model=selected_embed_model_memory, prompt=prompt)
        prompt_embedding = response['embedding']

        vector_db = self.client.get_collection(name='conversations')
        results = vector_db.query(query_embeddings=[prompt_embedding], n_results=1)
        best_embedding = results['documents'][0][0]
        
        return best_embedding

def recall(prompt):

    memory_agent = MemoryAgent()
    conversations = memory_agent.fetch_conversations()
    memory_agent.create_vector_db(conversations=conversations)
    # print(memory_agent.fetch_conversations())

    context = memory_agent.retrieve_embeddings(prompt=prompt)

    prompt = f'USER PROMPT: {prompt} \n CONTEXT FROM PREVIOUS CONVERSATIONS: {context}'
    
    return prompt

def store_response(prompt, response):
    memory_agent = MemoryAgent()
    conn = memory_agent.connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            'INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)'
            , (prompt, response)
        )
        conn.commit()
    conn.close()
    print(f' Memory state: {memory_agent.fetch_conversations()}')
