# YouTube Video Analyzer with Multimodal AI

This project implements a Streamlit web application that analyzes YouTube videos using multimodal AI models. The app extracts video frames, audio, and text data from a given YouTube video and allows you to perform queries over the multimodal data using a Local Qdrant vector store and LLaMA-based models.

## Features

- **YouTube Video Analysis**: Download and process YouTube videos, extracting images, audio, and transcribed text.
- **Multimodal AI Queries**: Perform multimodal queries across text and images using `OllamaMultiModal` LLM.
- **Vector Store Indexing**: Store and retrieve multimodal data using a local Qdrant vector store for efficient querying.
- **Streamlit Interface**: Simple and interactive web interface for video URL input, queries, and results display.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Caching and Resetting](#caching-and-resetting)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/youtube-video-analyzer.git
    cd youtube-video-analyzer
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up [Qdrant](https://qdrant.tech/), a local vector store for storing embeddings. You can either use the server version or stick with the embedded store.

4. Install [Streamlit](https://streamlit.io/) for the web-based interface:
    ```bash
    pip install streamlit
    ```

5. (Optional) Download the models used for embeddings and multimodal AI queries if not installed automatically.

## Usage

1. To run the app, simply execute:
    ```bash
    streamlit run app.py
    ```

2. Enter the YouTube video URL in the input field on the left sidebar.

3. The app will:
    - Download the video.
    - Extract frames and save them as images.
    - Extract audio and convert it to text using Whisper.
    - Store all data in a local Qdrant vector store.

4. Once indexed, enter your query to search across the text and image embeddings for relevant results.

## Architecture

### Components:

- **`YouTube Video Download & Processing`**: Download video content using `pytube`, then process it to extract frames and audio.
- **`Multimodal Embeddings`**: Use HuggingFace and CLIP models to create embeddings for both the text and image data.
- **`Qdrant Vector Store`**: A local vector store is used to store and retrieve embeddings efficiently.
- **`LLaMA-based Multimodal AI Model`**: LLaMA is used for generating responses to queries over the indexed data.
- **`Streamlit Interface`**: Interactive web app to accept YouTube URLs, enter queries, and view results.

### Data Flow:
1. **Download YouTube Video**: The video is downloaded using `pytube`.
2. **Extract Frames and Audio**: Frames are saved as PNG files, and the audio is extracted and transcribed using `Whisper`.
3. **Indexing**: Both text and image data are indexed and stored in Qdrant.
4. **Querying**: Users can submit text-based queries to retrieve relevant data from the indexed video.

## Caching and Resetting

- **Caching**: Index creation is cached using `st.cache_resource` to avoid reprocessing the same video multiple times.
- **Resetting**: The Qdrant store and cached resources can be reset by:
    ```python
    shutil.rmtree("qdrant_mm_db")  # Deletes the Qdrant database
    st.session_state.clear()  # Clears the session state and cache
    ```

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for any improvements, bug fixes, or feature requests.

### Steps to Contribute:
1. Fork this repository.
2. Create a new branch for your feature.
3. Make your changes and push to your branch.
4. Submit a pull request and describe your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
