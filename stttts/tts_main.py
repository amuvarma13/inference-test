import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
from tts_parse_output import parse_output
from tts_convert_to_wav import process_audio_and_get_vq_id
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()


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

model_name = "amuvarma/luna-2.6k-w-tags"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
model = model.to("cuda")

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
    # prepend_tokens = prompt_data.prepend_tokens

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)

    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

    # if prepend_tokens:
    #     prepend_tensor = torch.tensor([prepend_tokens], dtype=torch.int64)
    #     modified_input_ids = torch.cat([prepend_tensor, modified_input_ids], dim=1)

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


    # generated_text, numpy_audio = parse_output(generated_ids)

    print("generated ids", generated_ids.shape)
    numpy_audio = parse_output(generated_ids)

    end_time = time.time()
    total_time = end_time - start_time
    print("time taken", end_time - start_time)
    print(numpy_audio.shape)

    return {
        "input_prompt": prompt,
        "generated_text": "Generated in: " + str(total_time) + " s",
        "inference_time": end_time - start_time,
        "generated_shape": generated_ids.shape[1],
        "max_length": max_length, 
        "numpy_audio": numpy_audio.tolist(),
        "generated_ids": generated_ids.tolist(), 
        
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)