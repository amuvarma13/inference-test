import requests
import json
import numpy as np
import wave

def text_to_speech_api_call(prompt, max_length, prepend_tokens=None):
    url = "https://e2a9jg7rwha7y7-8080.proxy.runpod.net/inference"
    
    payload = {
        "prompt": prompt,
        "max_length": max_length
    }
    
    if prepend_tokens:
        payload["prepend_tokens"] = prepend_tokens
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

def write_audio_to_wav(audio_data, filename, sample_rate=16000):
    audio_array = np.array(audio_data, dtype=np.float32)
    audio_array_int = (audio_array * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array_int.tobytes())

def process_response(response_data, index):
    if response_data:
        print(f"\nResponse {index}:")
        print(f"Input Prompt: {response_data['input_prompt']}")
        print(f"Generated Text: {response_data['generated_text']}")
        print(f"Inference Time: {response_data['inference_time']} seconds")
        print(f"Generated Shape: {response_data['generated_shape']}")
        print(f"Max Length: {response_data['max_length']}")
        
        write_audio_to_wav(response_data['numpy_audio'], f"output_audio_{index}.wav")
        print(f"Audio file 'output_audio_{index}.wav' has been created.")
        
        return response_data.get('generated_shape')
    else:
        print(f"Failed to get response {index} from the API.")
        return None

# Example usage
prompts = [
    "You will have a conversation and help support me: my iphone isn't working, could you please tell me how to fix it?",
    "Thank you for the advice. My phone screen is completely black. What should I do?",
    "I've tried that, but it's still not working. Are there any other solutions?"
]
max_length = 3000

prepend_tokens = None

for i, prompt in enumerate(prompts, 1):
    response_data = text_to_speech_api_call(prompt, max_length, prepend_tokens)
    generated_shape = process_response(response_data, i)
    
    if i < len(prompts) and generated_shape:
        prepend_tokens = list(range(1, generated_shape + 1)) + [128258, 128262]

print("\nAll API calls completed.")