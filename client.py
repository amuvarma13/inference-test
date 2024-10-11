import requests
import json
import numpy as np
import wave

def text_to_speech_api_call(prompt, max_length, prepend_tokens=None):
    url = "https://e2a9jg7rwha7y7-8080.proxy.runpod.net/inference"
    
    payload = {
        "prompt": prompt,
        "max_length": max_length, 
        "prepend_tokens": prepend_tokens
    }
    
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
    # Convert the list back to a numpy array
    audio_array = np.array(audio_data, dtype=np.float32)
    
    # Normalize the audio to 16-bit range
    audio_array_int = (audio_array * 32767).astype(np.int16)
    
    # Write the WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array_int.tobytes())

# Example usage
prompt = "My name is John, how are you?"
max_length = 3000

response_data = text_to_speech_api_call(prompt, max_length)
write_audio_to_wav(response_data['numpy_audio'], "output_audio.wav")

pre_ids = response_data['generated_ids'][0]
pre_ids = pre_ids + [128262]

max_length = 6000
prompt = "What is my name?"

response_data1 = text_to_speech_api_call(prompt, max_length, pre_ids)
write_audio_to_wav(response_data1['numpy_audio'], "output_audio1.wav")

pre_ids = response_data1['generated_ids'][0]
pre_ids = pre_ids + [128262]

max_length = 9000
prompt = "I told you my name at the beginning of the conversation. Do you remember what it is?"

response_data2 = text_to_speech_api_call(prompt, max_length, pre_ids)
write_audio_to_wav(response_data2['numpy_audio'], "output_audio2.wav")

if response_data:
    print(f"Input Prompt: {response_data['input_prompt']}")
    print(f"Generated Text: {response_data['generated_text']}")
    print(f"Inference Time: {response_data['inference_time']} seconds")
    print(f"Generated Shape: {response_data['generated_shape']}")
    print(f"generated_ids Shape: {response_data['generated_ids']}")
    print(f"Max Length: {response_data['max_length']}")
    
    # Write the audio data to a WAV file
    write_audio_to_wav(response_data['numpy_audio'], "output_audio.wav")
    print("Audio file 'output_audio.wav' has been created.")
else:
    print("Failed to get response from the API.")