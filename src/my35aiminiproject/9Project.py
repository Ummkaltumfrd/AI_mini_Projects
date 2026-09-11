"""
THE 9TH PROJECT : Swappable	Model	Config
Learn:
1.abstrct the model from the code(env file)

"""
import os

import ollama
from dotenv import load_dotenv

load_dotenv()
client=ollama.Client()
model=os.environ.get("ai_model")

response = client.generate(
  model=model,
  prompt="Explain recursion simply"
  )

print("=========================")
print("THE result")
print("=========================")

print("model: ",model)
print("response: ",response.response)