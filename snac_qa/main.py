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
app = FastAPI()
from gzf import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)
from pydub import AudioSegment


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

class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 500  # Default value of 500, can be overridden in the request
    prepend_tokens: Optional[List[int]] = None  # Optional list of tokens to prepend

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
    prompt = prompt_data.prompt
    max_length = prompt_data.max_length
    user_tokens = new_inference_collator()



    test_audio, sr = torchaudio.load("recorded_audio.wav")
    print(test_audio.shape, sr)

    if sr != 16000:
        print("resampling audio")
        test_audio = torchaudio.transforms.Resample(sr, 16000)(test_audio)
    test_audio = test_audio[0]
    print("new", test_audio.shape)

    audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    audio_values = audio_processor(
        audio=test_audio, return_tensors="pt", sampling_rate=16000
    ).input_values

    myinputs= {
        "audio_values": audio_values.to(loaded_model_custom.device).to(loaded_model_custom.dtype),
        "input_ids": user_tokens.to(loaded_model_custom.device),
        # "input_ids": tokenizer("Okay, so what would be a healthier breakfast option then? Can you tell me?", return_tensors="pt").input_ids.to("cuda")
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
        max_new_tokens=1000,
        temperature=0.3,
        repetition_penalty=1.2,
        top_p=0.8,
        eos_token_id=128258,
        )
    
    print(outs)




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)