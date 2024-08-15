import requests
import pandas as pd
import os
from time import sleep

base_url = "https://api.isic-archive.com/api/v2/images/"


limit = 100  
download_count = 5000  
output_dir = "isic_images"
os.makedirs(output_dir, exist_ok=True)

data = []
next_url = base_url + f"?limit={limit}"

while next_url and len(data) < download_count:
    try:
        response = requests.get(next_url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve data: {e}")
        break
    
    images_metadata = response.json()
    
    for image in images_metadata.get('results', []):
        if len(data) >= download_count:
            break

        try:
            diagnosis = image['metadata']['clinical'].get('benign_malignant', 'unknown')
            if diagnosis in ['benign', 'malignant']:
                data.append({
                    'image_id': image['isic_id'],
                    'diagnosis': diagnosis,
                    'url': image['files']['full']['url']
                })
        except KeyError:
            continue


    next_url = images_metadata.get('next', None)


df = pd.DataFrame(data)


for index, row in df.iterrows():
    filename = f"{output_dir}/{row['diagnosis']}_{row['image_id']}.jpg"
    

    if os.path.exists(filename):
        print(f"Skipping {filename}, already exists.")
        continue

    try:
        response = requests.get(row['url'], timeout=10)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image {row['image_id']}: {e}")
        sleep(1) 