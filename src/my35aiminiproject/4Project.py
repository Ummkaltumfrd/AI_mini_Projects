
"""
  THE 4TH PROJECT : Multi-Tool	Router
Learn:
  1.using multiple tool.(tool routing)
  2.no tool call (in the code it does not work correctly 100%).
  3.the model excute multiple tools at time.
"""
import os
from datetime import datetime

import ollama
from dotenv import load_dotenv

load_dotenv()

client= ollama.Client()
model = os.environ.get('ai_model')

tools=[
  {
    "type":"function",
    "function":{
      "name":"Calculator",
      "description":"Adds two numbers together.Only use this tool when the user asks for arithmetic.Pass the   exact numbers from the user's question.",
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
      "description":"Transform and convert the centimeters (cm) to feet",
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
      "description":"Get the todays date",
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

Your job is to choose the correct tool based on the user's request.

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
while True:
 prompt = input("You: ")

 if prompt.lower() in ["quit","exit"]:
  print("Bye!")
  break

 messages=[
  {"role":"system", "content":system_prompt},
  {"role":"user", "content":prompt}
 ]
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
 else:
  print("Bot: ",message.content)