from ddgs import DDGS
import requests
import os
import time
import hashlib
from PIL import Image
from io import BytesIO

query = "BMW 3 series sedan e30"
folder = "dataset_raw/e30"

os.makedirs(folder, exist_ok=True)

seen_hashes = set()

#простые "car-фильтры" через ключевые слова в URL/метаданных
CAR_HINTS = ["car", "bmw", "vehicle", "auto", "sedan", "road"]

def is_probably_car(url):
    url = url.lower()
    return any(hint in url for hint in CAR_HINTS)

def is_good_image(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes))
        img = img.convert("RGB")

        w, h = img.size

        # слишком маленькие изображения
        if w < 200 or h < 200:
            return False

        # слишком “узкие” (часто логотипы/иконки)
        if w / h > 4 or h / w > 4:
            return False

        return True

    except:
        return False


with DDGS() as ddgs:
    results = ddgs.images(query, max_results=300)

    i = 0
    for r in results:
        try:
            url = r["image"]

            # грубый фильтр по ссылке
            if not is_probably_car(url):
                continue

            time.sleep(0.25)

            img = requests.get(url, timeout=10)

            if img.status_code != 200:
                continue

            content = img.content

            # убираем дубликаты
            img_hash = hashlib.md5(content).hexdigest()
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)

            # 🧹 фильтр качества изображения
            if not is_good_image(content):
                continue

            # сохраняем
            path = f"{folder}/{i}.jpg"
            with open(path, "wb") as f:
                f.write(content)

            print("Saved", i)
            i += 1

        except Exception as e:
            print("skip", i, e)