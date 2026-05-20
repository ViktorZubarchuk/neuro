from ddgs import DDGS
import requests
import os
import time
import hashlib
from PIL import Image
from io import BytesIO


# ЗАПРОСЫ
queries = [
    "BMW series G20 sedan",
    "BMW series G20 driving",
    "BMW series G20 side view",
    "BMW series G20 front view",
    "BMW series G20 rear view",
]


# ПАПКА
folder = "dataset_raw/g20"
os.makedirs(folder, exist_ok=True)


# ФИЛЬТРЫ
BAD_HINTS = [
    "toy",
    "diecast",
    "hotwheels",
    "render",
    "cgi",
    "drawing",
    "artstation",
    "wallpaper",
    "forza",
    "beamng",
    "lego",
    "modelcar",
    "matchbox",
    "illustration",
    "ai-generated",
    "sticker",
    "vector",
    "poster",
    "miniature",
    "3d-model",
    "roblox",
    "minecraft"
]

seen_hashes = set()

target = 1000
i = 0


# ПРОВЕРКА URL
def is_bad_url(url):
    url = url.lower()
    return any(bad in url for bad in BAD_HINTS)


# ПРОВЕРКА КАРТИНКИ
def is_good_image(content):
    try:
        img = Image.open(BytesIO(content)).convert("RGB")

        w, h = img.size

        # слишком маленькие
        if w < 150 or h < 150:
            return False

        # слишком вытянутые
        if w / h > 4 or h / w > 4:
            return False

        return True

    except:
        return False


# СКАЧИВАНИЕ
with DDGS() as ddgs:

    for query in queries:

        print(f"\nQUERY: {query}")

        try:
            results = ddgs.images(query, max_results=200)

            for r in results:

                try:
                    url = r["image"]

                    # мусорные ссылки
                    if is_bad_url(url):
                        continue

                    time.sleep(0.05)

                    response = requests.get(url, timeout=10)

                    if response.status_code != 200:
                        continue

                    content = response.content

                    # дубликаты
                    img_hash = hashlib.md5(content).hexdigest()

                    if img_hash in seen_hashes:
                        continue

                    seen_hashes.add(img_hash)

                    # проверка изображения
                    if not is_good_image(content):
                        continue

                    # сохранение
                    path = f"{folder}/{i}.jpg"

                    with open(path, "wb") as f:
                        f.write(content)

                    print(f"Saved {i}")

                    i += 1

                    # лимит
                    if i >= target:
                        break

                except Exception as e:
                    print("skip:", e)

        except Exception as e:
            print("query failed:", e)

        if i >= target:
            break


print(f"\nDONE. Saved: {i}")