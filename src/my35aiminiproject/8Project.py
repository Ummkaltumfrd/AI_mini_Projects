"""
  THE 8TH PROJECT : Model	Size Trade-off	Benchmark
Learn:
 1.model : llma3.2:1B => faster (26 tokens/sec) and make more mistakes
 2.model : llma3.2:3B => slower (15 tokens/sec) and make more accurante

  when the model small it is faster(generate tokens faster) and give false responses
    and the bigger models slower and give more accurate responses.
"""

"""From project 7"""
import json
import os
import re
from datetime import datetime

import ollama
import psutil
from dotenv import load_dotenv

load_dotenv()

client= ollama.Client()
model = os.environ.get('ai_model')


process = psutil.Process(os.getpid())

ram = process.memory_info().rss

tools=[
  {
    "type":"function",
    "function":{
      "name":"Calculator",
      "description":"Multiply the first two numbers together, then add the third number. Only use this tool when the user asks for arithmetic. Pass the exact numbers from the user's question.",
      "parameters":{
        "type":"object",
        "properties":{
         "a":{"type":"number"},
         "b":{"type":"number"},
         "c":{"type":"number"},
        },
        "required":["a","b","c"],
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

def Calculator(a,b,c):
 num = float(a) * float(b) + float(c)
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

def extract_number_calculator(text):
    match = re.search(
        r"(?:(?:answer|result|is|=)\s*:?\s*)(-?\d+(?:,\d+)*|\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    return None

def extract_number_converter(text): 
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text) 
 
    if numbers: 
        return float(numbers[-1]) 
 
    return None
def convert_to_datetime(date_string):
    date_string = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_string)
    date_string = date_string.strip()

    formats = [
    "%d %B %Y",      # 1 September 2026
    "%B %d, %Y",     # September 1, 2026
    "%m/%d/%Y",     # 9/1/2026
    "%m-%d-%Y",     # 9-1-2026
    "%Y-%m-%d",     # 2026-09-01
    "%d %b %Y",     # 03 Sep 2026
    ]

    for fmt in formats:
        try:
            date= datetime.strptime(date_string, fmt)
            print("date :",date)
            print("date strftime :",date.strftime("%B %d, %Y").replace(" 0"," "))
            return date.strftime("%B %d, %Y").replace(" 0"," ")
        except ValueError:
            continue

    return None


def extract_date(text):
    month_names = (
        r"January|February|March|April|May|June|July|"
        r"August|September|October|November|December"
    )

    patterns = [
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{month_names})\s+\d{{4}}\b",
        rf"\b(?:{month_names})\s+\d{{1,2}},\s+\d{{4}}\b",
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            print("match :",match.group(0))
            return convert_to_datetime(match.group(0))

    return None

def llm_judge(question,response):
 judge_prompt = f"""
You are an evaluator.

Question:
{question}

Agent answer:
{response}

Is the agent's answer correct and relevant?

Reply with only:
PASS
or
FAIL
"""
 result =client.generate(
  model=model,
  prompt=judge_prompt
 )
 return result.response.strip().upper()

system_prompt="""
You are a helpful assistant with access to exactly three tools:
Calculator, Converter, and DateTool.

Your job is to choose the correct tool based on the user's request.

You may:
- call one tool
- call multiple tools
- call no tool


TOOL 1: Calculator

Use Calculator ONLY when the user asks for arithmetic.

Calculator computes:

a * b + c

It requires exactly three numbers:
a, b, and c.

Pass the exact numbers from the user's question.

Example:
User: "What is 7 * 8 + 4?"

→ call Calculator with a=7, b=8, c=4.

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

with open("../../test_cases.json","r") as f:
  tests=json.load(f)

passed =0
total =len(tests)
judge_results=[]
for test in tests:
 tool_used =""
 expected=""
 prompt = test["question"]

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
   args=tool_call.function.arguments
   print("name:",name)
   print("args:",args)
   tool_used =name

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
  if tool_used == "Calculator":
   expected = extract_number_calculator(response2.message.content)
  elif tool_used == "Converter":
   expected = extract_number_converter(response2.message.content)
  else:
   expected = extract_date(response2.message.content)
   print("expected date:",expected)
  eval_count= response2.get("eval_count",0)
  print("eval_count:",eval_count)
  eval_duration=response2.get("eval_duration",0)
  print("eval_duration:",eval_duration)
 else:
  print("Bot: ",message.content)
  eval_count= response2.get("eval_count",0)
  print("eval_count:",eval_count)
  eval_duration=response2.get("eval_duration",0)
  print("eval_duration:",eval_duration)

 # code testing 
 if test["type"] == "code":
  p=0
  if expected == test["expected_answer"]:
   print(f"{test["id"]}->value: PASSED")
   p+=1
  else:
   print(f"{test["id"]}->value: Failed")

  if tool_used == test["expected_tool"]:
   print(f"{test["id"]}->tool: PASSED")
   p+=1
  else:
   print(f"{test["id"]}->tool: Failed")
  
  if p==2:
   passed+=1
  print("print",passed)

# llm judge testing
 elif test["type"] == "llm":
  judge_result=llm_judge(test["question"],message.content)
  print(f"{test["id"]}->judge: {judge_result}")
  judge_results.append(judge_result)


print("\n==================")
print("EVAL RESULTS")
print("==================")

judge_pass=judge_results.count("PASS")
judge_failed=judge_results.count("FAIL")

total_passed = passed+judge_pass
score = total_passed/total*100

if eval_duration > 0:
    tokens_sur_sec = eval_count / (eval_duration / 1_000_000_000)
else:
    tokens_sur_sec = 0

print("\n==================")
print(f"Model:{model}")
print(f"Pass Rate : {score}%")
print(f"Tokens/sec : {tokens_sur_sec}")
print("RAM used:", ram / (1024**2), "MB")
print("==================")