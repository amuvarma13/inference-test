import requests
import librosa
import soundfile as sf

# Read the audio file
data, samplerate = sf.read("recorded_audio.wav")

# data is a NumPy array. If you need a regular Python list of samples:
samples_list = data.tolist()
print(samples_list)

def main():
    # Replace with your actual inference endpoint
    url = "https://l05w722d1e25uv-8080.proxy.runpod.net/inference"
    
    # Sample payload that matches the PromptRequest model
    payload = {
        "prompt": "Hello world, what do you think about AI?",
        "max_length": 150, 
        "samples_list": samples_list
    }
    
    # Send POST request
    response = requests.post(url, json=payload)
    
    # Print status code and response data
    print("Status Code:", response.status_code)
    print("Response:", response.text)

# if __name__ == "__main__":
    # main()
