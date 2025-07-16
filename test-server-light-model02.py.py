from transformers import pipeline
import torch
import time

pipe = pipeline(
    "text-generation",
    model="google/gemma-3-1b-it",
    device="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    return_full_text=False
)

command = ""

def build_prompt(command: str) -> str:
    return f"""You are an AI agent. Given an English user command, output exactly one **complete** JSON object (include all closing braces):
{{
  "name": "<function_name>",
  "arguments": {{ … }}
}}

Allowed functions:
- turn_on_light(room: string)
- transfer_money(recipient: string, amount: integer, token: string)

User: "{command}"
Output (complete JSON):"""

prompt = build_prompt(command)

for _ in range(3):
    start_time = time.time()
    output = pipe(prompt, max_new_tokens=50)
    end_time = time.time()
    print(output)
    print(f"Generate time: {end_time - start_time:.2f} seconds")
