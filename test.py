from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import json
import torch
import time  # Thêm thư viện time để đo thời gian
import re
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)

def build_prompt(cmd: str) -> str:
    return f"""You are an AI agent. Given an English user command, output ONLY one JSON object:
{{
  "name": "<function_name>",
  "arguments": {{ ... }}
}}

Allowed functions:
- turn_on_light(room: string)
- transfer_money(recipient: string, amount: integer)

Examples:
User: "Turn on the living room light"
Output: {{"name":"turn_on_light","arguments":{{"room":"living room"}}}}

User: "{cmd}"
Output:"""

def agent_parse(cmd: str):
    prompt = build_prompt(cmd)

    # Bắt đầu đo thời gian trước khi gọi pipeline
    start_time = time.time()

    gen = pipe(prompt, max_new_tokens=64, do_sample=False, eos_token_id=tokenizer.eos_token_id)
    raw = gen[0]["generated_text"]  
    body = raw
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(body)


# Demo
print(agent_parse("Turn on the bedroom light"))
