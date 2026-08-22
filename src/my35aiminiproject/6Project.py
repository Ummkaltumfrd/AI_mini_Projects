"""
  THE 6TH PROJECT : Latency	&	Throughput	Baseline
Learn:
 1. the concept of the latency (how long time the model response for you request) and how to mesure it (using time.pref_counter).
 2.the concept of the throughput(how much the model can generate in amount of time) => tokens/sec
 3.the diffrence between cold and worm req.
 4.Ollama keep_alive keep the model loaded (certain period of time).
 5.using library pandas,saving results into a CSV file with it.
"""
import os
import time
from datetime import datetime

import ollama
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

client =ollama.Client()
model = os.environ.get('ai_model')
prompt="How are you doing today!"

start =time.perf_counter()
response=client.generate(
  model=model,
  prompt=prompt,
  keep_alive="10m"
)
end =time.perf_counter()

print("Bot:",response)
promptLength=len(prompt)
responseLength=len(response.response)

timestamp = datetime.fromisoformat(
    response.created_at.replace("Z", "+00:00")
)
duration =end - start
tokens= response.eval_count
tokens_per_second = tokens / duration
# regalite it depend on the experince
condition="warm"

print("Prompt length: ",promptLength)
print("Response length: ",responseLength)
print("Timestamp: ",timestamp)
print("duration: ",duration)
print("tokens: ",tokens)
print("tokens_per_second: ",tokens_per_second)

log={
"Prompt length":promptLength,
"Response length":responseLength,
"Timestamp":timestamp,
"duration":duration,
"tokens":tokens,
"tokens_per_second":tokens_per_second,
"condition":condition
}

df = pd.DataFrame([log])

file_exsists=os.path.exists("latency_p.csv")
df.to_csv("latency_p.csv",mode="a",header=not file_exsists,index=False)