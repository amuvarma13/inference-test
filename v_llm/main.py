from vllm import LLM, SamplingParams
prompts = [
    "Hello, my name is Joe.",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100, repetition_penalty=1.1)

llm = LLM(model="amuvarma/3days-tagged-noreps-caps")



outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].token_ids)
