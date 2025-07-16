# server.py
import os
import re
import json
import torch
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
from abc import ABC, abstractmethod
from typing import Literal, Dict, Any, Union

# --- 1. Configuration Management ---
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
    processing_mode: Literal['local', 'gemini'] = Field(
        default='gemini', 
        description="Processing mode: 'local' for local model, 'gemini' for Gemini API."
    )
    local_model_name: str = "google/gemma-3-1b-it"
    gemini_api_key: str = "YOUR_GEMINI_API_KEY_HERE"
    gemini_api_url: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

settings = Settings()


# --- 2. Prompting and Parsing Logic (IMPROVED) ---

def build_prompt(command: str) -> str:
    """
    Builds an improved prompt for the LLM to clearly distinguish between Command and Chat modes.
    """
    # Using f-string with triple quotes for better readability
    return f"""Your task is to act as a command parser. You have two distinct modes:

1.  **Command Mode**: If the user's request can be mapped to one of the `ALLOWED FUNCTIONS`, you MUST output a valid JSON object representing that function call and nothing else. If some arguments are missing, set their value to `null`. Do not add any extra text or explanation.

2.  **Chat Mode**: If the user's request is a general question, a greeting, or anything that does NOT map to an allowed function, you MUST reply as a friendly chatbot in plain text. DO NOT mention your functions or capabilities. DO NOT output JSON.

---
**ALLOWED FUNCTIONS:**
- `transfer_money(recipient: str, amount: int, token: str)`
- `swap_token(from_token: str, to_token: str, amount: int)`
- `get_token_price(token_name: str)`
- `get_balance(wallet_address: str, token_name: str)`
- `deposit_vault(amount: int, token: str)`
---

**EXAMPLES:**

# Example 1 (Command Mode - Full command)
User: "transfer 100 USDT to user_abc with token 123456"
Output:
```json
{{
  "name": "transfer_money",
  "arguments": {{
    "recipient": "user_abc",
    "amount": 100,
    "token": "123456"
  }}
}}
```

# Example 2 (Chat Mode - Not an allowed function)
User: "turn on the light in the kitchen"
Output:
Sorry, I can't control home devices.

# Example 3 (Chat Mode - General conversation)
User: "how are you?"
Output:
I'm fine, thank you! How can I help you today?

# Example 4 (Command Mode - Missing arguments)
User: "transfer 50 usdc to bob"
Output:
```json
{{
  "name": "transfer_money",
  "arguments": {{
    "recipient": "bob",
    "amount": 50,
    "token": null
  }}
}}
```

# Example 5 (Command Mode - Price check)
User: "what is btc price?"
Output:
```json
{{
  "name": "get_token_price",
  "arguments": {{
    "token_name": "btc"
  }}
}}
```

# Example 6 (Command Mode - Short price check)
User: "eth price"
Output:
```json
{{
  "name": "get_token_price",
  "arguments": {{
    "token_name": "eth"
  }}
}}
```

# Example 7 (Command Mode - Terse price check)
User: "sol?"
Output:
```json
{{
  "name": "get_token_price",
  "arguments": {{
    "token_name": "sol"
  }}
}}
```

# Example 8 (Command Mode - Balance Check)
User: "check usdt balance for wallet 0x123abc"
Output:
```json
{{
  "name": "get_balance",
  "arguments": {{
    "wallet_address": "0x123abc",
    "token_name": "usdt"
  }}
}}
```

# Example 9 (Command Mode - Deposit to vault)
User: "deposit 1 apt to vault"
Output:
```json
{{
  "name": "deposit_vault",
  "arguments": {{
    "amount": 1,
    "token": "apt"
  }}
}}
```

# Example 10 (Command Mode - Another deposit phrasing)
User: "put 0.2 apt into vault"
Output:
```json
{{
  "name": "deposit_vault",
  "arguments": {{
    "amount": 0.2,
    "token": "apt"
  }}
}}
---

**Current Request:**

User: "{command}"
Output:
"""

def parse_llm_output(text: str) -> Union[Dict[str, Any], str]:
    """
    Attempts to extract and parse a JSON object from the LLM's output text.
    This function is designed to find JSON even when it's wrapped in markdown.
    If no valid JSON is found, it returns the cleaned text.
    """
    # Search for a JSON block wrapped in markdown ```json ... ```
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not match:
        # If not found, search for any JSON object
        match = re.search(r'(\{.*?\})', text, re.DOTALL)

    if match:
        json_str = match.group(1)
        try:
            # Try to parse the found string into JSON
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If parsing fails, the original text will be returned at the end of the function
            pass

    # If there's no JSON or parsing fails, return the cleaned original text.
    return text.strip()


# --- 3. LLM Service Abstraction ---
class LLMService(ABC):
    @abstractmethod
    def process_command(self, command: str) -> Union[Dict[str, Any], str]:
        pass

class GeminiService(LLMService):
    def __init__(self, api_key: str, api_url: str):
        if not api_key or "YOUR_GEMINI_API_KEY" in api_key:
            raise ValueError("GEMINI_API_KEY is not configured. Please set it in your .env file.")
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {"Content-Type": "application/json", "X-goog-api-key": self.api_key}

    def process_command(self, command: str) -> Union[Dict[str, Any], str]:
        prompt = build_prompt(command)
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        print("--- Sending request to Gemini API ---")
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            response_json = response.json()
            print(f"Gemini Raw Response: {response_json}")

            text_output = response_json["candidates"][0]["content"]["parts"][0]["text"]
            return parse_llm_output(text_output)
        except requests.exceptions.RequestException as e:
            print(f"Error calling Gemini API: {e}")
            raise HTTPException(status_code=503, detail=f"Error communicating with Gemini API: {e}")
        except (KeyError, IndexError):
            print(f"Error parsing Gemini response: {response_json}")
            raise HTTPException(status_code=500, detail="Invalid response structure from Gemini API.")

class LocalModelService(LLMService):
    def __init__(self, model_name: str):
        print("Loading local model. This may take a while...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            self.llm = pipeline("text-generation", model=model_name, device=self.device, torch_dtype=self.dtype, return_full_text=False)
            print(f"Model '{model_name}' loaded successfully on '{self.device}'.")
        except Exception as e:
            print(f"Failed to load local model '{model_name}'. Error: {e}")
            raise RuntimeError(f"Could not load local model.") from e

    def process_command(self, command: str) -> Union[Dict[str, Any], str]:
        prompt = build_prompt(command)
        max_new_tokens = len(command.split()) + 150 
        
        print("--- Running local model inference ---")
        try:
            response = self.llm(prompt, max_new_tokens=max_new_tokens, do_sample=False)
            text_output = response[0]['generated_text']
            print(f"Local Model Raw Response: {text_output}")
            return parse_llm_output(text_output)
        except Exception as e:
            print(f"Error during local model inference: {e}")
            raise HTTPException(status_code=500, detail=f"Error during local model inference: {e}")

# --- 4. FastAPI Application Setup ---
app = FastAPI(title="LLM Command Processor", description="API to process commands using a local model or Gemini API.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup_event():
    print(f"Application starting in '{settings.processing_mode}' mode.")
    if settings.processing_mode == 'local':
        app.state.llm_service = LocalModelService(model_name=settings.local_model_name)
    elif settings.processing_mode == 'gemini':
        app.state.llm_service = GeminiService(api_key=settings.gemini_api_key, api_url=settings.gemini_api_url)
    else:
        raise ValueError(f"Invalid PROCESSING_MODE: {settings.processing_mode}")

class CommandRequest(BaseModel):
    command: str

@app.post("/command", summary="Process a user command")
async def command_endpoint(req: CommandRequest):
    if not hasattr(app.state, 'llm_service') or app.state.llm_service is None:
        raise HTTPException(status_code=503, detail="LLM Service is not available.")
    service: LLMService = app.state.llm_service
    return service.process_command(req.command)

@app.get("/", summary="Health Check")
def read_root():
    return {"status": "ok", "processing_mode": settings.processing_mode}

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    print("Starting server with uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
