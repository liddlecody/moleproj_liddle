import requests

base_url = "https://api.isic-archive.com/api/v2/images/"

response = requests.get(base_url + "?limit=10")

if response.status_code == 200:
    images_metadata = response.json()
    print(images_metadata)  # Print the full response to inspect the data structure
else:
    print(f"Failed to retrieve data: {response.status_code}")
    print(response.text)