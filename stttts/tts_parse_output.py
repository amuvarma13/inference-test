from transformers import AutoModelForCausalLM, AutoTokenizer
tokeniser_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokeniser_name)
from tts_convert_to_wav import process_input_ids

def parse_output (generated_ids):
    # sos_indices = (generated_ids[0] == 128000).nonzero(as_tuple=True)[0]
    # second_sos_index = sos_indices[-1].item()

    # # Find the next occurrence of 128009 (EOS) after the second SOS
    # eos_index = (generated_ids[0][second_sos_index:] == 128009).nonzero(as_tuple=True)[0][0].item() + second_sos_index

    # # Extract the tokens between SOS and EOS
    # extracted_tokens = generated_ids[0][second_sos_index +1: eos_index]

    # decoded_text = tokenizer.decode(extracted_tokens)

    numpy_audio = process_input_ids(generated_ids)

    # print(f"numpy_audio shape: {numpy_audio.shape}")


    return  numpy_audio
    
    # return decoded_text, 