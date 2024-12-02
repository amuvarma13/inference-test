from math import inf
import torch
import torch.nn.functional as F
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "amuvarma/750k-shuffled-25-10-convo-tune-contentonly"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
model = model.to("cuda")

def custom_content_generate(input_ids, max_length=2000, temperature=0.1, top_k=50, top_p=0.95):
    input_ids = input_ids.cuda()
    generated = input_ids.clone()
    device = model.device

    past_key_values = None
    eos_token_id = 128258
    target_token_id = 128257

    target_positions = []
    last_printed = {}
    current_length = input_ids.shape[1]

    for i in range(max_length):
        i = i + 1
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids=generated, use_cache=True)
            else:
                outputs = model(input_ids=generated[:, -1:], use_cache=True, past_key_values=past_key_values)

            logits = outputs.logits[:, -1, :] / temperature

            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                values, indices = torch.topk(logits, top_k_val)
                mask = torch.full_like(logits, float('-inf'))
                mask.scatter_(1, indices, values)
                logits = mask

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_value = next_token.item()

            current_length += 1

            if next_token_value == target_token_id:
                target_positions.append(current_length - 1)
                last_printed[current_length - 1] = 0

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token_value == eos_token_id:
                break

    return generated