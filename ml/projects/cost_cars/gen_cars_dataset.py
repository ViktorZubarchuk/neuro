import pandas as pd
import numpy as np

np.random.seed(42)

# Бренды и модели
brands_models = {
    "BMW": ["3 Series", "5 Series", "7 Series", "3 Series_M", "5 Series_M", "7 Series_M", 
            "X3", "X5", "X7", "X3_M", "X5_M", "X7_M"],
}

# Тип кузова
body_types = {
    "3 Series": "Sedan",
    "5 Series": "Sedan",
    "7 Series": "Sedan",
    "3 Series_M": "Sedan",
    "5 Series_M": "Sedan",
    "7 Series_M": "Sedan",
    "X3": "SUV",
    "X5": "SUV",
    "X7": "SUV",
    "X3_M": "SUV",
    "X5_M": "SUV",
    "X7_M": "SUV"
}

data = []

for _ in range(3000):    
    k = np.random.randint(63,68)

    brand = np.random.choice(list(brands_models.keys()))
    model = np.random.choice(brands_models[brand])
    year = np.random.randint(2010, 2023)

    # Пробег (чем новее машина, тем меньше)
    mileage = int(np.random.normal(18000 * (year-2010), 500 * (year-2010)))
    mileage = max(5000, mileage)
    
    # Двигатель и объём
    if brand == "BMW":
        if model[-2:] == "_M":
            engine_volume = round(np.random.normal(3.3, 0.2),1)
            engine_hp = round(engine_volume*k, 1)
            base_price = (1500000 + (year-2010)*300000)*1.3
        else:
            engine_volume = round(np.random.normal(2.5, 0.5),1)
            engine_hp = round(engine_volume*k, 1)
            base_price = 1500000 + (year-2010)*300000
        
        if body_types[model] == "SUV":
            base_price *= 1.2
    
    # Число владельцев (1–4, чем старше, тем больше)
    n_owners = min(1 + (2025-year)//3 + np.random.randint(0,2), 4)
    
    # Коробка передач
    transmission = np.random.choice(["Manual", "Automatic"], p=[0.3, 0.7])
    
    # Цена в рублях с влиянием пробега, мощности, объёма
    price = base_price - mileage*10 + engine_hp*5000 + engine_volume*20000
    price = int(max(200000, price + np.random.randint(-50000, 50000)))
    
    data.append([
        brand,
        model,
        year,
        mileage,
        engine_volume,
        engine_hp,
        n_owners,
        transmission,
        body_types[model],
        price
    ])

# Создаём DataFrame
df = pd.DataFrame(data, columns=[
    "brand", "model", "year", "mileage_km", "engine_volume",
    "engine_hp", "n_owners", "transmission", "body_type", "price_rub"
])

for col in ["mileage_km", "engine_volume", "engine_hp", "n_owners"]:
    mask = np.random.rand(len(df)) < 0.02  # 2% пропусков по каждому признаку
    df.loc[mask, col] = np.nan

# Сохраняем
df.to_csv("cost_cars/cars_dataset.csv", sep=";", index=False)

# Показываем первые 10 строк
print(df.head(10))