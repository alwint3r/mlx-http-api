from mlx_lm import generate, load
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

model_alias = {
    "zephyr-7b": "zephyr-7b-beta-mlx-4bit",
    "mistral-7b": "mistral-7b-v0.1-mlx-4bit",
    "qwen1.5-0.5b": "Qwen/Qwen1.5-0.5B",
    "mistral-7b-instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "phi2": "microsoft/phi-2"
}

def get_model_path(model_name):
    actual_model_name = model_alias[model_name]
    is_mlx_model = "mlx-4bit" in actual_model_name
    if is_mlx_model:
        return f"/Users/alwin/llms/mlx/{actual_model_name}"
    return actual_model_name


class GenerateOptions(BaseModel):
    temperature: float = 0.1
    max_length: int = 100

class GenerateRequest(BaseModel):
    model: str = ""
    prompt: str
    options: GenerateOptions = GenerateOptions()

class ChatOptions(BaseModel):
    temperature: float = 0.1
    max_length: int = 100

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    options: ChatOptions = ChatOptions()

app = FastAPI()

@app.get("/insiders/models")
async def get_all_models():
    return { "models": list(model_alias.keys()) }

@app.post("/api/generate")
async def generate_text(request: GenerateRequest):
    is_model_supported = request.model in model_alias
    if not is_model_supported:
        raise HTTPException(status_code=400, detail="Model not supported")
    model_name = get_model_path(request.model)
    model, tokenizer = load(model_name)

    response = generate(model, tokenizer,
                        prompt=request.prompt,
                        max_tokens=request.options.max_length,
                        temp=request.options.temperature)

    return { "model": request.model, "response": response }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    is_model_supported = request.model in model_alias
    if not is_model_supported:
        raise HTTPException(status_code=400, detail="Model not supported")
    model_name = get_model_path(request.model)
    model, tokenizer = load(model_name)

    messages = request.messages
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer,
                        prompt=prompt,
                        max_tokens=request.options.max_length,
                        temp=request.options.temperature)
    if response.startswith("<|assistant|>\n"):
        response = response[len("<|assistant|>\n"):]
    return { "model": request.model, "message": { "role": "assistant", "content": response } }

