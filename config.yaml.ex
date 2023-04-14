# ----- PINECONE CONFIG -----
PINECONE_API_KEY: ""
PINECONE_INDEX: "" # dimensions: 1536, metric: cosine similarity
PINECONE_ENV: ""

# ----- SERVER PORT ----
SERVER_PORT: "8080"

# ---- OPENAI CONFIG -----
OPENAI_API_KEY: ""
EMBEDDINGS_MODEL: "text-embedding-ada-002"
GENERATIVE_MODEL: "gpt-4" # use gpt-4 for better results
EMBEDDING_DIMENSIONS: 1536
TEXT_EMBEDDING_CHUNK_SIZE: 200
# This is the minimum cosine similarity score that a file must have with the search query to be considered relevant
# This is an arbitrary value, and you should vary/ remove this depending on the diversity of your dataset
COSINE_SIM_THRESHOLD: 0.7
MAX_TEXTS_TO_EMBED_BATCH_SIZE: 100
MAX_PINECONE_VECTORS_TO_UPSERT_PATCH_SIZE: 100
