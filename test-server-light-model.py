import json
import re
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

def build_prompt(cmd: str) -> str:
    return f"""You are an AI agent. Given an English user command, convert user command to output ONLY one JSON object, just create from command do not add more content:
{{
  "name": "<function_name>",
  "arguments": {{ ... }}
}}

Allowed functions:
- turn_on_light(room: string)
- transfer_money(recipient: string, amount: integer)
- fly_high(recipient: string, height: integer)

Examples:
User: "Turn on the living room light"
Output: {{"name":"turn_on_light","arguments":{{"room":"living room"}}}}

User: "{cmd}"
Output:"""

def agent_parse(cmd: str):
    prompt = build_prompt(cmd)
    start_time = time.time()
    gen = pipe(
        prompt,
        max_new_tokens=64,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )
    elapsed = time.time() - start_time
    print(f"[Timing] Generate took {elapsed:.3f} seconds")
    raw = gen[0]["generated_text"].strip()
    print("Raw completion:", raw.replace("\n", "\\n"))
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON.")
            return None
    else:
        print("Error: No JSON object found in the output.")
        return None

if __name__ == "__main__":
    commands = [
        "Turn on the bedroom light",
    ]
    for cmd in commands:
        print(f"\n-- Command: {cmd}")
        result = agent_parse(cmd)
        print("Parsed result:", result)
        if result:
            print("Function Name:", result.get("name"))
            print("Arguments:", result.get("arguments"))
