from app.chains.vector_store import vectorstore

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8, "k": 1}
)

# retriever 정보 확인
print(f"Retriever: {retriever}")
