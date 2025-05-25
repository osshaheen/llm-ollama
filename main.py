from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import sys

model = OllamaLLM(model="llama3.2")

template = """
You are an exeprt in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# while True:
    # print("\n\n-------------------------------")
question = sys.argv[1]
    # print("\n\n")
    # if question == "q":
        # break
    
reviews = retriever.invoke(question)
result = chain.invoke({"reviews": reviews, "question": question})
print(result)