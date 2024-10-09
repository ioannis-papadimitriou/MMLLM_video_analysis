model_options = {
    "Llava-1.6": "llava:7b-v1.6-mistral-q6_K",
    "Llava-llama3": "llava-llama3:8b-v1.1-fp16",
    "Llama-3.1": "llama3.1:8b-instruct-q6_K",
}

embed_model_options = {
    "multilingual_baai": "BAAI/bge-m3", # ~4X more compute needed
    "english_baai": "BAAI/bge-base-en-v1.5"
}

# Only parameters to adjust
selected_model_name = "Llava-1.6"
selected_embed_model_name = "english_baai"
selected_embed_model_memory = 'nomic-embed-text:latest'
temperature = 0.1
context_window = 4096 # llama3.1 goes up to 128k but you'll need a big boat to float that

selected_model = model_options[selected_model_name]
selected_embed_model = embed_model_options[selected_embed_model_name]
base_url="http://127.0.0.1:11434"