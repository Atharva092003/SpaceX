import os
from dotenv import load_dotenv
from google import genai
from src.vectorization import Vectorizer
from src.retriver import mainretriever

vc= Vectorizer()
vector_db= vc.hfembedder()
mnr= mainretriever()

load_dotenv()
api_key = os.getenv("GENAI_API_KEY")
client=genai.Client(api_key=api_key)   

while True:
    question= input("\nAsk Any Question: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Stopped.")
        break
    context= mnr.Topkretriever(question, vector_db, 3) 
    sysprompt= f'''
    You are a helpful assistant
    Use the context provided to answer the user's question
    You have to provide answer in depth
    Context: {context}
    User Question: {question}
    you have to give response in points.
    '''

    response = client.models.generate_content(
    model= "gemini-3.1-flash-lite",
    contents=sysprompt,
    )

    print("\nGemini Response:\n")
    print(response.text)




        
        
    