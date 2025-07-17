from langchain.chains import RetrievalQA, LLMChain
from app.chains.retriever import retriever
from app.chains.llm import llm
from app.prompts.prompt_builder import prompt_template
from app.prompts.prompt_fallback import prompt_fallback

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

fallback_chain = LLMChain(
    llm=llm,
    prompt=prompt_fallback
)
