def extract_tokens_after_value(tensor, target_start=128257, target_end=128258):
    tensor_list = tensor.tolist()

    start_index = tensor_list.index(target_start)
    try:
        end_index = tensor_list.index(target_end, start_index)
        return tensor_list[start_index + 1:end_index]
    except ValueError:
        return tensor_list[start_index + 1:]