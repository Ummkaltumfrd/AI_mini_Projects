import os
import ollama
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()

client =ollama.Client()

class RecipeExactor(BaseModel):
  name: str
  ingrediants: list[str]
  steps: list[str]


model =os.environ.get('ai_model')
#prompt = "Give me a short recipe for pasta with tomato sauce."

filePath=os.environ.get('file_txt_path')

with open(filePath,"r",encoding="utf-8") as f:
 prompt=f.read()

for attempt in range(3):
  try:
    response  = client.generate(
     model=model,
     prompt=prompt,
     format=RecipeExactor.model_json_schema()
     )
    recipe=RecipeExactor.model_validate_json(response.response)
    break
  except Exception:
    print("Invalid output, retrying...")

print("Response from Ollama:")
print(recipe)