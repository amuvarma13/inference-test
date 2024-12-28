import requests

def main():
    # Replace with your actual inference endpoint
    url = "https://l05w722d1e25uv-8080.proxy.runpod.net/inference"
    
    # Sample payload that matches the PromptRequest model
    payload = {
        "prompt": "Hello world, what do you think about AI?",
        "max_length": 150
    }
    
    # Send POST request
    response = requests.post(url, json=payload)
    
    # Print status code and response data
    print("Status Code:", response.status_code)
    print("Response:", response.text)

if __name__ == "__main__":
    main()
