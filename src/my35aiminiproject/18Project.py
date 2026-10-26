"""
THE 18TH PROJECT : Agentic	RAG
Learn:
1.use tools (have one function called search_knowledge_base(query)).
2.get search_documents function from 17main.py file without writing everything from 0.
3.for every qst a seperate call (tool calls).
4. every time the calling happening since I'm using llama3.2:1B (lchta for greeting).
5.retrive data from chromadb : collection (company_documents).

"""

import importlib.util
import json
import os
import sys

import ollama
from dotenv import load_dotenv

load_dotenv()


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT17_DIR = os.path.join(PROJECT_DIR, "17Project")
MAIN17_PATH = os.path.join(PROJECT17_DIR, "17main.py")

sys.path.insert(0, PROJECT17_DIR)

spec = importlib.util.spec_from_file_location(
    "main17", MAIN17_PATH
)

if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load: {MAIN17_PATH}")

main17 = importlib.util.module_from_spec(spec)
sys.modules["main17"] = main17
spec.loader.exec_module(main17)

search_documents = main17.search_documents

print("17main database count:", main17.collection.count())

load_dotenv()

model=os.environ.get("ai_model")
client =ollama.Client()

def search_knowledge_base(query:str) -> str:

  results = search_documents(query)
  if not results:
    return "No relevant information found."
  return "\n\n".join(str(result) for result in results)

tools=[{
  "type":"function",
  "function":{
    "name":"search_knowledge_base",
   "description": """
        Search the knowledge base for ONE specific information need.

        IMPORTANT:
        - Each query must contain only ONE question or information need.
        - If the user asks multiple questions in one message, call this tool separately for each question.
        - Never combine different questions into one query.
        - Example: if the user asks about water usage, solar generation,and electricity consumption, make three separate tool calls,one for each topic.
        - Do not use this tool for greetings or casual conversation.
""",
    "parameters":{
      "type":"object",
      "properties":{
        "query":{
          "type":"string",
          "description":"the specific information to searh for."
        }
      },
      "required":["query"]
    }
  }
}]

system_prompt = """
You are an Agentic RAG assistant grounded in a knowledge base.

- For greetings and casual conversation, answer directly without retrieval.
- For document-specific questions, use search_knowledge_base.
- Before searching, identify how many distinct questions the user asked.
- If there are N distinct information needs, make N separate tool calls.
- Each tool call must search for exactly ONE information need.
- Never merge multiple questions into a single search query.
- If the user asks three questions, search for each one separately.
- Inspect the results and answer every question using its own evidence.
- Do not invent facts or confuse similar values.
- If a result does not answer a question, search again with a more
  specific query when needed.
"""

while True:
  prompt = input("You: ")

  if prompt.lower() in ["exit","quit"]:
    print("Bye!")
    break

   
  messages =[
     {"role":"system","content":system_prompt},
     {"role":"user","content":prompt}
    ]

  # Prevent an endless tool-calling loop
  max_steps = 8

  for step in range(max_steps):

   response = client.chat(
    model=model,
    messages=messages,
    tools=tools
   )

   messages.append(response.message)

   if not response.message.tool_calls:
     print("Assistant:",response.message.content)
     break

   for tool_call in response.message.tool_calls:
     name = tool_call.function.name
     args = tool_call.function.arguments

     if name == "search_knowledge_base":
       if isinstance(args, str):
         args = json.loads(args)

       query = args.get("query", "").strip()

         # Separate questions using question marks
       questions = [
          q.strip()
          for q in query.split("?")
          if q.strip()
          ]

          # Search each question separately
       results = []
       try:
        for question in questions:
         print(f"[Retrieval] Searching: {question}")
         
        result = search_knowledge_base(question)
        results.append(
        f"Question: {question}\nResult: {result}"
        )

        result = "\n\n".join(results)
         
       except Exception as exc:
         result = f"Retrieval failed: {exc}"

     else:
       result = f"Unknown tool:{name}"
     
     messages.append({
       "role":"tool",
       "tool_name":name,
       "content": str(result)
     })

     print("Assistante: ",response.message.content)