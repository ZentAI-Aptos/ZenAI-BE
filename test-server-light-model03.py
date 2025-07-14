#!/usr/bin/env python3
# test_turn_on_light.py

from llama_cpp import Llama
import json

MODEL_REPO = "microsoft/Phi-3-mini-4k-instruct-gguf"
MODEL_FILE = "Phi-3-mini-4k-instruct-q4.gguf"
NUM_THREADS = 4


def load_model(repo: str, filename: str, threads: int) -> Llama:
    """Load and return a Llama GGUF model."""
    return Llama.from_pretrained(
        repo_id=repo,
        filename=filename,
        n_threads=threads,
    )


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


def safe_load(raw: str): 
    # strip whitespace
    s = raw.strip()
    # count braces
    opens = s.count("{")
    closes = s.count("}")
    # append missing closes
    s += "}" * (opens - closes)
    return json.loads(s)


def run_command(llm: Llama, command: str) -> str:
    """Send the prompt to the model and return its raw JSON response."""
    prompt = build_prompt(command)
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
        temperature=0.0,
    )
    print(response['choices'][0]['message']['content'].strip())
    # return (safe_load(response['choices'][0]['message']['content'].strip()))


def main():
    llm = load_model(MODEL_REPO, MODEL_FILE, NUM_THREADS)
    command = ""
    result = run_command(llm, command)
    print(result)


if __name__ == "__main__":
    main()
