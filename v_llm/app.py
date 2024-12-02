from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from typing import List
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from convert_to_wav import process_input_ids
from extract_parts import extract_content_tokens
# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize LLM and sampling parameters
llm = LLM(model="amuvarma/3days-tagged-noreps-caps")
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1000,
    repetition_penalty=1.1
)

class InferenceResponse(BaseModel):
    wavs: List

@app.get("/generate", response_model=InferenceResponse)
async def generate_text(prompt: str):
    try:
        outputs = llm.generate([prompt], sampling_params)
        token_ids = outputs[0].outputs[0].token_ids
        # content_tokens_response = extract_content_tokens(token_ids)
        wavs = process_input_ids(token_ids)
        return InferenceResponse(wavs=wavs.tolist())
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)