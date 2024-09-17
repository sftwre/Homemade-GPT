import json
import os
import urllib.request as request
 
def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    file_path = "./data/instruction-data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    
    data = download_and_load_file(file_path, url)
    print("Number of entries:", len(data))