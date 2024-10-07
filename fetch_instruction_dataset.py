import os
import urllib.request as request
from urllib.error import URLError
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Downloads an instruction dataset to a provided directory."
    )
    parser.add_argument("--url", type=str, help="dataset url")
    parser.add_argument("--file_path", type=str, help="dataset file path")
    args = parser.parse_args()

    url = args.url
    file_path = args.file_path

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
