import os
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.8,
    max_output_tokens=2048,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
