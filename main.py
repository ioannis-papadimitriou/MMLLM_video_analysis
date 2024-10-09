import os
import gc
import re
import time
import speech_recognition as sr
import qdrant_client
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from moviepy.editor import VideoFileClip
from pytubefix import YouTube
from llama_index.core import SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.multi_modal_llms.ollama import OllamaMultiModal
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response.notebook_utils import display_source_node
from models import *
from prompts import *

# Set up paths
output_video_path = "./video_data/"
output_folder = "./mixed_data/"
output_audio_path = "./mixed_data/output_audio.wav"
filepath = os.path.join(output_video_path, "input_vid.mp4")
Path(output_folder).mkdir(parents=True, exist_ok=True)

embed_model = HuggingFaceEmbedding(model_name=selected_embed_model)
Settings.embed_model = embed_model

# Ensure necessary directories exist
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Streamlit setup for chat-based interface
st.title("YouTube Video Analyzer")

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

# Function to validate YouTube URL
def is_valid_youtube_url(url):
    youtube_regex = re.compile(r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/.+')
    return youtube_regex.match(url)

# Function to download video
def download_video(url, output_path):
    yt = YouTube(url)
    metadata = {"Author": yt.author, "Title": yt.title, "Views": yt.views}
    yt.streams.get_highest_resolution().download(output_path=output_path, filename="input_vid.mp4")
    return metadata

# Function to extract images from video
def video_to_images(video_path, output_folder):
    clip = VideoFileClip(video_path)
    clip.write_images_sequence(os.path.join(output_folder, "frame%04d.png"), fps=0.2)

# Function to extract audio from video
def video_to_audio(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_audio_path)

# Convert audio to text using Whisper
def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)
    with audio as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_whisper(audio_data)
        except sr.UnknownValueError:
            st.write("Speech recognition could not understand the audio.")
        except sr.RequestError as e:
            st.write(f"Could not request results from service; {e}")
    return text

def clear_old_data(folder_path):
    for file in Path(folder_path).glob("*"):
        file.unlink()  # Removes the file

def reset_qdrant():
    import shutil
    db_path = Path("qdrant_mm_db")
    if db_path.exists() and db_path.is_dir():
        shutil.rmtree(db_path)  # This will remove the directory and all its contents
            
# Setup index and metadata
# Use st.cache_resource to cache the index creation
@st.cache_resource
def setup_index(url):
    if not is_valid_youtube_url(url):
        st.write("Invalid URL")
        return None, None
    else:
        # Clear old data
        clear_old_data(output_folder)
        reset_qdrant()
        
        metadata_vid = download_video(url, output_video_path)
        video_to_images(filepath, output_folder)
        video_to_audio(filepath, output_audio_path)
        text_data = audio_to_text(output_audio_path)

        with open(output_folder + "output_text.txt", "w") as file:
            file.write(text_data)

        os.remove(output_audio_path)  # Clean up the audio file

        # Create a local Qdrant vector store
        client = qdrant_client.QdrantClient(path="qdrant_mm_db")

        text_store = QdrantVectorStore(client=client, collection_name="text_collection")
        image_store = QdrantVectorStore(client=client, collection_name="image_collection")
        storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)

        documents = SimpleDirectoryReader(output_folder).load_data()
        index = MultiModalVectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            image_embed_model=ClipEmbedding()
        )
        return index, metadata_vid

# Streamlit UI for querying and displaying results
def display_response(response):
    # Display retrieved text content
    for text_node in response.metadata["text_nodes"]:
        st.write(display_source_node(text_node, source_length=200))
    
    # Display retrieved images
    image_paths = [n.metadata["file_path"] for n in response.metadata["image_nodes"]]
    plot_images(image_paths)

# Function to plot images using matplotlib
def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break
    st.pyplot(plt)

def response_generator(response):
    for word in response.split():
        yield word + ' '
        time.sleep(0.01)

if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    video_url = st.text_input("Insert YouTube video URL:", "")

if video_url:
    index, metadata_vid = setup_index(video_url)

    if prompt := st.chat_input("Enter your query:"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        llm = OllamaMultiModal(model=selected_model, temperature=temperature)
        qa_tmpl = PromptTemplate(qa_tmpl_str)
        query_engine = index.as_query_engine(llm=llm, text_qa_template=qa_tmpl)

        response = query_engine.query(prompt)
        # display_response(response)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(str(response)))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


# Button to reset the session
if st.sidebar.button("Reset Conversation"):
    st.session_state.clear()  # Clear the session state, removing all stored data and messages
    st.rerun()  # Refresh the page to reflect the cleared state

# Add a disclaimer in the sidebar
st.caption(":red[Disclaimer: The responses from the AI are generated based on the data provided and general knowledge. They might not always be accurate or applicable to your specific circumstances. Please verify the information independently.]")
