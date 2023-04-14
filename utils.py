import openai
import logging
import sys
import time

from config import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_pinecone_id_for_file_chunk(session_id, filename, chunk_index):
    return str(session_id+"-!"+filename+"-!"+str(chunk_index))

def get_embedding(text, engine):
    return openai.Engine(id=engine).embeddings(input=[text])["data"][0]["embedding"]

def get_embeddings(text_array, engine):
    # Parameters for exponential backoff
    max_retries = 5 # Maximum number of retries
    base_delay = 1 # Base delay in seconds
    factor = 2 # Factor to multiply the delay by after each retry
    while True:
        try:
            return openai.Engine(id=engine).embeddings(input=text_array)["data"]
        except Exception as e:
            if max_retries > 0:
                logging.info(f"Request failed. Retrying in {base_delay} seconds.")
                time.sleep(base_delay)
                max_retries -= 1
                base_delay *= factor
            else:
                raise e

# Create embeddings for a text using a tokenizer and an OpenAI engine
def create_embeddings_for_text(text, tokenizer):
    """Return a list of tuples (text_chunk, embedding) and an average embedding for a text."""
    token_chunks = list(chunks(text, TEXT_EMBEDDING_CHUNK_SIZE, tokenizer))
    text_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]

    # Split text_chunks into shorter arrays of max length 10
    text_chunks_arrays = [text_chunks[i:i+MAX_TEXTS_TO_EMBED_BATCH_SIZE] for i in range(0, len(text_chunks), MAX_TEXTS_TO_EMBED_BATCH_SIZE)]

    # Call get_embeddings for each shorter array and combine the results
    embeddings = []
    for text_chunks_array in text_chunks_arrays:
        embeddings_response = get_embeddings(text_chunks_array, EMBEDDINGS_MODEL)
        embeddings.extend([embedding["embedding"] for embedding in embeddings_response])

    text_embeddings = list(zip(text_chunks, embeddings))

    return (text_embeddings)

# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j