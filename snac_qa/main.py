import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import transformers
import torchaudio
from IPython.display import Audio
from extract_tokens_after_value import extract_tokens_after_value
app = FastAPI()
from gzf import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)
from pydub import AudioSegment
import torch

from snac import SNAC
import torch

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to("cuda")


sound = AudioSegment.from_mp3("3.mp3")
sound.export("recorded_audio.wav", format="wav")
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

number_add_tokens = 6 * 1024 + 10  # 6144 + 10 = 6154
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]  # 6155 tokens
tokenizer.add_tokens(new_tokens)
tokenizer.add_special_tokens({'additional_special_tokens': ['<|audio|>']})

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
special_config =  model.config
output_dir = "amuvarma/snac-e2e-projonly-3"
loaded_model_custom = GazelleForConditionalGeneration.from_pretrained(output_dir, config=special_config, new_vocab_size=True)
loaded_model_custom = loaded_model_custom.to(device=device, dtype=dtype)

class PromptRequest(BaseModel):
    audio: Optional[List[float]] = None
    max_length: int = 500  # Default value of 500, can be overridden in the request
    prepend_tokens: Optional[List[int]] = None  # Optional list of tokens to prepend, 
    samples_list: List[float]

class TextPromptRequest(BaseModel):
    prompt: str
    max_length: int = 500  # Default value of 500, can be overridden in the request
    prepend_tokens: Optional[List[int]] = None  # Optional list of tokens to prepend, 


@app.get("/ping")
async def ping():
    return {"message": "pong"}

def new_inference_collator():
    user_phrase = "<|audio|>" #<|audio|>"
    user_input_ids = tokenizer(user_phrase, return_tensors="pt").input_ids
    end_of_text = torch.tensor([[128009]], dtype=torch.int64)
    start_of_system = torch.tensor([[128256+8]], dtype=torch.int64)
    end_of_system = torch.tensor([[128256+9]], dtype=torch.int64)

    system_message = "You are an AI assistant who will answer the user's questions with short responses."
    system_input_ids = tokenizer(system_message, return_tensors="pt").input_ids
    system_tokens = torch.cat(
        [start_of_system, system_input_ids, end_of_text, end_of_system],  dim=1)

    # print("user_input_ids", user_input_ids.shape)

    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)
    final_tokens = torch.tensor([[128009]], dtype=torch.int64)


    user_tokens = torch.cat(
        [system_tokens, start_token, user_input_ids, end_tokens], dim=1)

    return user_tokens

@app.post("/inference")
async def inference(prompt_data: PromptRequest):
    max_length = prompt_data.max_length
    samples_list = prompt_data.samples_list
    user_tokens = new_inference_collator()
    


    audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    audio_values = audio_processor(
        audio=samples_list, return_tensors="pt", sampling_rate=16000
    ).input_values



    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)


    myinputs= {
        "audio_values": audio_values.to(loaded_model_custom.device).to(loaded_model_custom.dtype),
        "input_ids": user_tokens.to(loaded_model_custom.device),
        # "input_ids": tokenizer("Okay, so what would be a healthier breakfast option then? Can you tell me?", return_tensors="pt").input_ids.to("cuda")
    }
    stop_token = 128258
  
    start_time = time.time()
    

    outs = loaded_model_custom.generate(
        **myinputs,
        max_new_tokens=100,
        temperature=0.3,
        repetition_penalty=1.2,
        top_p=0.8,
        eos_token_id=128258,
        )
    
    print(outs)
    print(tokenizer.decode(outs[0], skip_special_tokens=True))



@app.post("/inference-text")
async def inference_text(prompt_data: TextPromptRequest):
    prompt = prompt_data.prompt
    max_length = prompt_data.max_length

    user_tokens = new_inference_collator()


    myinputs= {
        # "audio_values": audio_values.to(loaded_model_custom.device).to(loaded_model_custom.dtype),
        # "input_ids": "What is a healthy breakfast option?",
        "input_ids": tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    }

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
    

    outs = loaded_model_custom.generate(
      **myinputs,
      max_new_tokens=500,
      temperature=0.7,
      repetition_penalty=1.1,
      top_p=0.9,
      eos_token_id=128258,
    )

    text_tokens = extract_tokens_after_value(outs[0], 128261, 128257)
    text_tokens = text_tokens[1:-1]
    text_response = tokenizer.decode(text_tokens)
    
    token_to_find = 128257
    token_to_remove = 128263

    # Check if the token exists in the tensor
    token_indices = (outs == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = outs[:, last_occurrence_idx+1:]
    else:
        cropped_tensor = outs

    mask = cropped_tensor != token_to_remove
    cropped_tensor = cropped_tensor[mask].view(cropped_tensor.size(0), -1)

    processed_tensor = cropped_tensor - 128266
    original_shape = processed_tensor.shape
    new_dim_1 = (original_shape[1] // 7) * 7
    processed_tensor = processed_tensor[:, :new_dim_1]
    code_list = processed_tensor[0].tolist()
    def redistribute_codes(code_list):
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range((len(code_list)+1)//7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1]-4096)
            layer_3.append(code_list[7*i+2]-(2*4096))
            layer_3.append(code_list[7*i+3]-(3*4096))
            layer_2.append(code_list[7*i+4]-(4*4096))
            layer_3.append(code_list[7*i+5]-(5*4096))
            layer_3.append(code_list[7*i+6]-(6*4096))

        codes = [torch.tensor(layer_1).unsqueeze(0).to("cuda"),
                torch.tensor(layer_2).unsqueeze(0).to("cuda"),
                torch.tensor(layer_3).unsqueeze(0).to("cuda")]
        audio_hat = snac_model.decode(codes)
        return audio_hat


    samples = redistribute_codes(code_list)
    print("my samples are", samples)

    return {
        "input_prompt": prompt,
        "inference_time": time.time() - start_time,
        # "generated_shape": generated_ids.shape[1],
        "text_response": text_response,
        "max_length": max_length, 
        "numpy_audio": samples.detach().cpu().numpy().tolist(),
        # "generated_ids": generated_ids.tolist(), 

    }







if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)