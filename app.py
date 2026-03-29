from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

template = ChatPromptTemplate.from_messages([
    ("system", "You are a precise math calculator. Provide only the numerical answer or a very brief step-by-step solution."),
    ("human", "{question}")
])

model = OllamaLLM(model="gemma2:2b")
llm_chain = template | model

print("--- Math AI Assistant Loaded (Type 'exit' to quit) ---")

while True:
    user_input = input("Enter a math question: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    
    response = llm_chain.invoke({"question": user_input})
    print(f"Result: {response}\n")