import os
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    temperature=0.2,
    max_output_tokens=2048,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
