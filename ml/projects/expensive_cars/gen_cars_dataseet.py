import pandas as pd
import numpy as np

np.random.seed(42)

brands_models = {
    "Toyota": ["Corolla", "Yaris", "Camry", "RAV4", "Prius"],
    "BMW": ["3 Series", "5 Series", "X3", "X5", "1 Series"]
}

body_types = {
    "Corolla": "Sedan",
    "Yaris": "Hatchback",
    "Camry": "Sedan",
    "RAV4": "SUV",
    "Prius": "Hatchback",
    "3 Series": "Sedan",
    "5 Series": "Sedan",
    "X3": "SUV",
    "X5": "SUV",
    "1 Series": "Hatchback"
}

data = []

for _ in range(300):
    brand = np.random.choice(list(brands_models.keys()))
    model = np.random.choice(brands_models[brand])
    
    year = np.random.randint(2010, 2023)
    
    mileage = int(np.random.normal(80000 - (year - 2010)*3000, 10000))
    mileage = max(5000, mileage)
    
    if brand == "Toyota":
        engine = np.random.randint(90, 210)
        base_price = 8000 + (year - 2010)*1000
    else:
        engine = np.random.randint(120, 350)
        base_price = 15000 + (year - 2010)*1500

    transmission = np.random.choice(["Manual", "Automatic"], p=[0.4, 0.6])
    
    price_eur = base_price - mileage * 0.03 + engine * 20
    price_eur = int(max(5000, price_eur + np.random.randint(-2000, 2000)))
    
    expensive = 1 if price_eur > 18000 else 0
    
    data.append([
        brand,
        model,
        year,
        mileage,
        engine,
        transmission,
        body_types[model],
        expensive
    ])

df = pd.DataFrame(data, columns=[
    "brand", "model", "year", "mileage_km",
    "engine_hp", "transmission", "body_type", "expensive"
])

df.to_csv("expensive_cars/cars_dataset.csv", sep=";", index=False)

print(df.head())