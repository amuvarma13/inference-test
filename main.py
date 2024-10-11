import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import time

tokeniser_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokeniser_name)

prompt = '''I'm not sure how we can prevent churn and grow revenue?'''
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

start_token = torch.tensor([[ 128259]], dtype=torch.int64)
end_tokens = torch.tensor([[128009, 128260, 128261]], dtype=torch.int64)

modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

input_ids = modified_input_ids
attention_mask = torch.ones_like(input_ids)

input_ids = input_ids.to("cuda")
attention_mask = attention_mask.to("cuda")
stop_token = 128258

model_name = "amuvarma/convo-fpsft-13k"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
model = model.to("cuda")

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

print(f"Time taken: {time.time() - start_time}")

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

print(f"Time taken: {time.time() - start_time}")

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

print(f"Time taken: {time.time() - start_time}")

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

print(f"Time taken: {time.time() - start_time}")



print(generated_ids.shape)