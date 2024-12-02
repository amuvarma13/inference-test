from math import inf
import torch
import torch.nn.functional as F

def custom_generate(model, input_ids, max_length=2000, temperature=0.01, top_p=1.0, repetition_penalty=1.2, custom_rep_pen=1.05, top_k=0):
    input_ids = input_ids.cuda()
    generated = input_ids.clone()
    device = model.device
    past_key_values = None
    eos_token_id = 128258
    current_length = input_ids.shape[1]
    window_size = 10

    range_tokens = torch.tensor([], dtype=torch.int64)
    range_tokens = range_tokens.to("cuda")

    for _ in range(max_length - current_length):
        with torch.no_grad():
            # Use cached key-values for efficiency after first forward pass
            if past_key_values is None:
                outputs = model(input_ids=generated, use_cache=True)
            else:
                outputs = model(
                    input_ids=generated[:, -1:],
                    use_cache=True,
                    past_key_values=past_key_values
                )

            # Get logits and apply temperature
            logits = outputs.logits[:, -1, :].clone()
            past_key_values = outputs.past_key_values

            # Apply standard repetition penalty
            if repetition_penalty != 1.0:
                score = torch.gather(logits, 1, generated)
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                logits.scatter_(1, generated, score)

            # Apply custom repetition penalty for sliding window
            if generated.size(1) > 1:  # Only if we have generated tokens
                # Get last window_size tokens
                last_tokens = generated[0, max(0, generated.size(1) - window_size):]

                # Count frequencies of tokens in the window
                unique_tokens, counts = torch.unique(last_tokens, return_counts=True)

                # Apply custom penalty based on frequency
                for token, count in zip(unique_tokens, counts):
                    penalty = custom_rep_pen ** count.item()
                    if logits[0, token] < 0:
                        logits[0, token] *= penalty
                    else:
                        logits[0, token] /= penalty

            # Prevent 3 consecutive identical range tokens
            if range_tokens.shape[0] >= 2:
                last_two_range = range_tokens[-2:]
                if last_two_range[0] == last_two_range[1]:
                    # If the last two range tokens are identical, prevent generating the same token again
                    logits[0, last_two_range[0]] = float('-inf')

            logits = logits / temperature

            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                values, indices = torch.topk(logits, top_k_val)
                mask = torch.full_like(logits, float('-inf'))
                mask.scatter_(1, indices, values)
                logits = mask


            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1),
                    dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1,
                    sorted_indices,
                    sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token[0][0]>128266 and next_token[0][0]<(128266+1024):
                range_tokens = torch.cat([range_tokens, next_token[0]])

            # Append next token to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            current_length += 1

            # Stop if EOS token is generated
            if next_token.item() == eos_token_id:
                break

    return generated
