import requests
import json
import os
import io
import zipfile

if not os.path.exists("dataset.jsonl"):
    url = "https://data.jobtechdev.se/annonser/historiska/2023.jsonl.zip"

    try:
        zip = requests.get(url)
        zip.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(zip.content)) as zip_file:
            file_name = zip_file.namelist()[0]
            jsonl_file = zip_file.open(file_name)
            print("File loaded successfully!")

            # Write to the JSONL file
            with open("dataset.jsonl", "ab") as output_file:  # Use "ab" mode to append
                for line in jsonl_file:
                    output_file.write(line)
            print("JSONL data appended successfully!")

    except requests.exceptions.RequestException as e:
        print("Failed to fetch data:", e)
    except Exception as e:
        print("Error:", e)
