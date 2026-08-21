
"""
  THE 5TH PROJECT : Agent	With	Persistent	Memory
Learn:
 1.using a db storage sqlite3 (local file)
 2.the flow of storing and retriving data to be used for future conversation with LLM
 3 resert the agent's memory
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import ollama
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[2]))

from memory import load_memory, reset_memory, save_excute

load_dotenv()

client= ollama.Client()
model = os.environ.get('ai_model')

tools=[
  {
    "type":"function",
    "function":{
      "name":"Calculator",
      "description": "ONLY use this tool for arithmetic addition of two numbers. The user MUST explicitly ask for addition, such as 3 + 4. Never use this tool for general questions, personal questions, unit conversion, dates, or questions that do not require addition.",
      "parameters":{
        "type":"object",
        "properties":{
         "a":{"type":"number"},
         "b":{"type":"number"}
        },
        "required":["a","b"],
        "additionalProperties":False
      },
      "strict":True
    },
  },
    {
    "type":"function",
    "function":{
      "name":"Converter",
      "description":"ONLY use this tool when the user explicitly asks to convert a specific numeric value from centimeters (cm) to feet. The user MUST provide a numeric centimeter value. Never use this tool for general questions, personal questions, greetings, or questions without a numeric cm value"
      "",
      "parameters":{
        "type":"object",
        "properties":{
         "cm":{"type":"number"}
        },
        "required":["cm"],
        "additionalProperties":False
      },
      "strict":True
    },
  },
    {
    "type":"function",
    "function":{
      "name":"DateTool",
      "description":"ONLY use this tool when the user explicitly asks for today's date, today's date and time, or the current date. Never use this tool for general questions or other dates.",
      "parameters":{
        "type":"object",
        "properties":{
        },
        "required":[],
        "additionalProperties":False
      },
      "strict":True
    },
  }
]

def Calculator(a,b):
 num = float(a) + float(b)
 return int(num) if num.is_integer() else num 

def Converter(cm):
  cmttr =float(cm)
  return cmttr*0.0328

def DateTool():
 today=datetime.today()
 return today

def call_function(name,args):
 if name == "Calculator":
  return Calculator(**args)
 elif name == "Converter":
  return Converter(**args)
 elif name == "DateTool":
  return DateTool()
 raise ValueError(f"Unknown tool {name}")



system_prompt="""
You are a helpful assistant with access to exactly three tools:
Calculator, Converter, and DateTool.

Your job is to choose the correct tool based on the user's request and if the user ask you a regular qestion just answer and do not use any tool.

You may:
- call one tool
- call multiple tools
- call no tool


TOOL 1: Calculator

Use Calculator ONLY when the user asks for arithmetic addition.

It adds two numbers together.

Pass the exact numbers from the user's question.

Do NOT use Calculator for:
- unit conversion
- dates
- greetings
- general questions

Example:
User: "What is 3 + 4?"
→ call Calculator with a=3 and b=4.


TOOL 2: Converter

Use Converter ONLY when the user asks to convert centimeters (cm) to feet.

The parameter is called "cm".

Example:
User: "Convert 190 cm to feet."
→ call Converter with cm=190.

Do NOT use Calculator for cm-to-feet conversion.


TOOL 3: DateTool

Use DateTool ONLY when the user asks for today's date or the current date.

DateTool takes no arguments.

Example:
User: "What is today's date?"
→ call DateTool.


NO TOOL

Do NOT call any tool for:
- greetings
- casual conversation
- general knowledge
- questions that do not require a tool

Examples:

User: "Hi"
→ answer directly without calling a tool.

User: "Hello"
→ answer directly without calling a tool.

User: "What is the capital of France?"
→ answer directly without calling a tool.


MULTIPLE TOOLS

If the user asks for multiple independent tasks that require different tools,
call all the required tools.

Example:

User:
"What is today's date and convert 190 cm to feet?"

→ call DateTool
→ call Converter with cm=190


Another example:

User:
"What is today's date and what is 3 + 3?"

→ call DateTool
→ call Calculator with a=3 and b=3


IMPORTANT

Choose tools based on the user's intent.

Do not call a tool just because a number appears in the message.

If no tool is needed, do not call any tool.

Never invent tool results.

After receiving the tool results, answer the user's original question
using all the available tool results.
"""

memories=load_memory()
messages =[ {"role":"system", "content":system_prompt}]
for role,content in memories:
 messages.append({
  "role":role,
  "content":content
 })

while True:
 prompt = input("You: ")

 save_excute("user",prompt)
  
 if prompt.lower() in ["quit","exit"]:
  print("Bye!")
 elif prompt.lower() == "/reset":
  reset_memory()
  print("the reset is done successfuly")
  break

 messages.append(
  {"role":"user",
   "content":prompt}
 )
 response1 = client.chat(
  model=model,
  messages=messages,
  tools=tools
 ) 

 message = response1.message
 
 if message.tool_calls:
 
  for tool_call in message.tool_calls:

   name=tool_call.function.name
   print("name",name)
   args=tool_call.function.arguments
   print("arguments",args)
   try:
    result = call_function(name,args)
    print("result",result)
   except (TypeError,ValueError) as e:
    print(f"[tool error]: {e}")
    break

   messages.append(message)

   messages.append({
    "role":"tool",
    "content": str(result)
   })


  response2=client.chat(
     model=model,
     messages=messages
    )

  print("Bot: ",response2.message.content)
  save_excute("assistant",response2.message.content)
 else:
  print("Bot: ",message.content)
  save_excute("assistant",message.content)