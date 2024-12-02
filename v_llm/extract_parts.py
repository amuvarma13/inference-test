def extract_text_tokens(tensor, start_token=128259, end_token=128009):

    # Extract the 1D sequence from shape (1,n)
    tokens = tensor[0].tolist() if hasattr(tensor[0], 'tolist') else tensor[0]
    
    # Find the last occurrence of start_token
    start_idx = len(tokens) - 1 - tokens[::-1].index(start_token)
    
    # Find the last occurrence of end_token before start_token
    # We only search up to start_idx
    end_idx = len(tokens) - 1 - tokens[::-1].index(end_token)
    
    return tokens[start_idx:end_idx]

def extract_content_tokens (generated_ids):
    token_to_find = 128257
    token_to_remove = 128263

    # Check if the token exists in the tensor
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()  # Get the last occurrence index
        # Crop the tensor to the values after the last occurrence
        cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    else:
        # If token not found, no cropping is done
        cropped_tensor = generated_ids

    mask = cropped_tensor != token_to_remove
    cropped_tensor = cropped_tensor[mask].view(cropped_tensor.size(0), -1)

    processed_tensor = cropped_tensor - 128266
    return processed_tensor

