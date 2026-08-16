
"""
  THE 2ND PROJECT : Streaming	Chatbot
Learn:
 1.Streaming with ollama
 2.prompt not hardcoded
 3.KeyboardInterrupt exception (try/except)
"""
import os

import ollama
from dotenv import load_dotenv

load_dotenv()

client=ollama.Client()
model =os.environ.get('ai_model')

while True:
  prompt=input("\n You:")

  if prompt.lower() in ["exit","quit"]:
    print("Bye!")
    break

  print("Bot:",end="")

  response = client.generate(
   model=model,
   prompt=prompt,
   stream=True
  )

  try:
   for chunck in response:
    print(chunck.response,end="")

  except KeyboardInterrupt:
    print("\n [END THE RESPONSE]")
    continue

  print()