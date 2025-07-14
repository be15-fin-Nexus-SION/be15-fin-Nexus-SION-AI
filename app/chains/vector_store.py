from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

vectorstore = Qdrant(
    client=client,
    collection_name="fp_examples",
    embeddings=embedding
)