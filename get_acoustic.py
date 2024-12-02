from math import inf
import torch
import torch.nn.functional as F
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM


custom_model_name = "amuvarma/convo-tts-tune-76layer"
custom_model = AutoModelForCausalLM.from_pretrained(custom_model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
custom_model = custom_model.cuda()

def custom_generate(input_ids, myc, max_length=2000, temperature=0.7, top_k=50, top_p=0.9):
    
    print("recorded max length is", max_length)

    input_ids = input_ids.cuda()
    myc = myc.cuda()
    myc = myc + 128266
    generated = input_ids.clone()
    device = custom_model.device

    past_key_values = None
    eos_token_id = 128258
    target_token_id = 128257

    # Track tokens in the specified range
    token_range_start = 128266
    token_range_end = 128266 + 1024
    special_tokens_counter = 0
    special_tokens_list = []

    target_positions = []
    last_printed = {}
    current_length = input_ids.shape[1]
    content_tokens = torch.tensor([[]], dtype=torch.long, device=device)

    for i in range(max_length):
        i = i +1
        with torch.no_grad():
            if past_key_values is None:
                outputs = custom_model(input_ids=generated, use_cache=True)
            else:
                outputs = custom_model(input_ids=generated[:, -1:], use_cache=True, past_key_values=past_key_values)

            logits = outputs.logits[:, -1, :] / temperature

            # Get the most likely token
            most_likely_token = torch.argmax(logits).item()

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


            if (token_range_start <= most_likely_token <= token_range_end):
              new_tensor = torch.full_like(logits, float('-inf'))
              new_tensor[0][myc[0][special_tokens_counter]] = 0
              logits = new_tensor



            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_value = next_token.item()

            # Track special tokens
            if token_range_start <= next_token_value <= token_range_end:
                special_tokens_counter += 1
                special_tokens_list.append(next_token_value)
                # print(f"Special token generated: {next_token_value}")
                # print(f"Total special tokens generated so far: {special_tokens_counter}")

            current_length += 1

            if next_token_value == target_token_id:
                target_positions.append(current_length - 1)
                last_printed[current_length - 1] = 0

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token_value == eos_token_id:
                break


    return generated
