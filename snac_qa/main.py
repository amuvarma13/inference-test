import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
from gzf import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)


# Add this block after creating the FastAPI app instance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and tokenizer
tokeniser_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokeniser_name)

model_id = "amuvarma/convo-tts-tune-7contentonly"

# 2. Device and dtype setup
import torch
device = "cpu"
dtype = torch.float32
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
    print(f"Using {device} device")
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
    print(f"Using {device} device")

config = GazelleConfig(
    audio_model_id="facebook/wav2vec2-base-960h",
    text_model_id=model_id,
    audio_token_index=134411,
    vocab_size=len(tokenizer),  # Updated vocab_size
)
model = GazelleForConditionalGeneration(config).to(dtype=dtype)

output_dir = "amuvarma/snac-e2e-projonly-3"
loaded_model_custom = GazelleForConditionalGeneration.from_pretrained(output_dir, config=special_config, new_vocab_size=True)

class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 500  # Default value of 500, can be overridden in the request
    prepend_tokens: Optional[List[int]] = None  # Optional list of tokens to prepend

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.post("/inference")
async def inference(prompt_data: PromptRequest):
    prompt = prompt_data.prompt
    max_length = prompt_data.max_length

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)

    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
    input_ids = modified_input_ids
    attention_mask = torch.ones_like(input_ids)

    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")
    stop_token = 128258

    start_time = time.time()
    

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=2500,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.2,
        top_k=50,
        # top_p=0.95,
        repetition_penalty=1.05,
        eos_token_id=stop_token,
    )




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)