
"""
  THE 3ND PROJECT : Single-Tool	Agent
Learn:
 1.agent loop (User->1st call LLM -> checking if it call to a function ->call function -> 2nd call LLM -> final result-> )
                                                        |-------------------------------------------------------------|->response )
 2.using single tool
"""
import os

import ollama
from dotenv import load_dotenv

load_dotenv()

client =ollama.Client()
# using 3B (it always calling the tool because small models overuse the available tool.)
model=os.environ.get('ai_model')

# Function 
def Calculator(a,b):
  res= float(a)+float(b)
  return int(res) if res.is_integer() else res
tools= [
  {
    "type" : "function",
    "function": {
      "name" :"Calculator",
      "description": "Adds two numbers together.Only use this tool when the user asks for arithmetic.Pass the exact numbers from the user's question.",
      "parameters":{
        "type" : "object",
        "properties" : {
           "a" : {"type":"number"},
           "b" : {"type":"number"},
        },
        "required" :["a","b"],
        "additionalProperties":False,
      },
      "strict" : True
    },
  }
]

# Function exsicution
def call_function(name,args):
  if name=="Calculator":
    return Calculator(**args)
  raise ValueError(f"Unknown tool:{name}")

system_prompt = """
You are a helpful assistant.

You have access to one tool called Calculator.

IMPORTANT:
- Only call Calculator when the user explicitly asks you to calculate or perform arithmetic.
- For greetings, casual conversation, general questions, or questions that do not require arithmetic, DO NOT call Calculator.
- For example:
  User: "hi" -> answer directly, no tool call.
  User: "what is the capital of France?" -> answer directly, no tool call.
  User: "what is 2 + 1?" -> call Calculator with a=2 and b=1.
"""

while True:
  prompt=input("You: ")

  if prompt.lower() in ["exit","quit"]:
    print("Bye!")
    break

  messages = [
  {"role":"system", "content" : system_prompt},
  {"role":"user", "content" : prompt}
  ]

  response1=client.chat(
    model=model,
    messages=messages,
    tools=tools
  )

  message = response1.message

  if message.tool_calls:

   for tool_call in message.tool_calls:
   
    name = tool_call.function.name
    args = tool_call.function.arguments

    print(f"\n [tool call] {name}({args})")

    try:
     result = call_function(name,args)

    except (TypeError,ValueError) as e:
     result = f"Tool error:{e}"

    print(f"[tool result] {result}")

    messages.append(message)

    messages.append({
     "role":"tool",
     "content":  str(result)
    })

    response2=client.chat(
     model=model,
     messages=messages
     )
    print("Bot:",response2.message.content)
  else:
    print("Bot:",message.content)