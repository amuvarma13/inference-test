import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Load model and tokenizer
tokeniser_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokeniser_name)

model_name = "amuvarma/convo-fpsft-13k"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
model = model.to("cuda")

class Prompt(BaseModel):
    prompt: str

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.post("/inference")
async def inference(prompt_data: Prompt):
    prompt = prompt_data.prompt

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    start_token = torch.tensor([[ 128259]], dtype=torch.int64)
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
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.01,
        eos_token_id=stop_token,
    )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    end_time = time.time()

    return {
        "input_prompt": prompt,
        "generated_text": generated_text,
        "inference_time": end_time - start_time,
        "generated_shape": generated_ids.shape[1]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)