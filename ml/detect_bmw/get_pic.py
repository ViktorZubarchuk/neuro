import os
import requests

from duckduckgo_search import DDGS


QUERY = "BMW E46 sedan"

SAVE_FOLDER = "dataset/e46"

MAX_IMAGES = 100

os.makedirs(SAVE_FOLDER, exist_ok=True)


def download_image(url, path):

    try:

        response = requests.get(url, timeout=10)

        if response.status_code == 200:

            with open(path, "wb") as f:
                f.write(response.content)

            return True

    except:
        return False


with DDGS() as ddgs:

    results = ddgs.images(
        QUERY,
        max_results=MAX_IMAGES
    )

    for i, result in enumerate(results):

        image_url = result["image"]

        file_path = os.path.join(
            SAVE_FOLDER,
            f"img_{i}.jpg"
        )

        success = download_image(
            image_url,
            file_path
        )

        if success:
            print(f"Saved {file_path}")

print("DONE")