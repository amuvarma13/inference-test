import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
from tts_parse_output import parse_output
from tts_convert_to_wav import process_audio_and_get_vq_id
from custom_generate import custom_generate
from format_prompt import format_prompt
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

model_name = "amuvarma/luna-3days-tagged-noreps"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

model = model.to("cuda")

class PromptRequest(BaseModel):
    prompt: str
    max_length: float = 1500.0
    prepend_tokens: Optional[List[int]] = None
    temperature: float
    duration: float
    emotion: str

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.post("/inference")
async def inference(prompt_data: PromptRequest):
    prompt = prompt_data.prompt
    emotion = prompt_data.emotion
    max_length = int(prompt_data.max_length)
    temperature_adjuster = prompt_data.temperature
    # prepend_tokens = prompt_data.prepend_tokens

    print("request received")

    prompt = prompt.strip()

    # Replace '!' and '?' with '.'
    prompt = prompt.replace('!', '.').replace('?', '.')

    prompt = prompt[0].upper() + prompt[1:]
    if not prompt.endswith('.'):
        prompt += '.'

    print("prompt", prompt)




    prompt, temperature, top_p, repetition_penalty, custom_rep_pen, top_k = format_prompt(prompt, emotion, temperature_adjuster)

    print("modified prompt", prompt)



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
    
    print("topk", top_k)
    # Example usage:
    generated_ids = custom_generate(
        model=model,
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty, 
        custom_rep_pen=custom_rep_pen
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