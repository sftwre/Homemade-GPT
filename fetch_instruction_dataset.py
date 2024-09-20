import os
import urllib.request as request
from urllib.error import URLError


if __name__ == "__main__":

    file_path = "./data/instruction-data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

    if os.path.exists(file_path):
         print(f"Dataset already present at: {file_path}")
    else:
        try:
            with request.urlopen(url) as response:
                text_data = response.read().decode("utf-8")

                dirs = os.path.dirname(file_path)
                os.makedirs(dirs, exist_ok=True)
                
                with open(file_path, "w", encoding="utf-8") as file:
                        file.write(text_data)

            print("Instruction dataset successfully downloaded!")

        except (URLError, IOError) as e:
            print(e)