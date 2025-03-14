import requests

# Define the URL of the Flask server
url = "http://127.0.0.1:5000/chat/"

# Define the query you want to send
query = "What is Astrocytoma"

# Create the JSON payload
payload = {
    "query": query
}

# Set the headers to specify JSON content
headers = {
    "Content-Type": "application/json"
}

# Send the POST request
try:
    response = requests.post(url, json=payload, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        print("Response from server:")
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
except Exception as e:
    print(f"An error occurred: {e}")