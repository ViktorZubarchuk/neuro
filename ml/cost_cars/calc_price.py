import pandas as pd
import numpy as np

np.random.seed(42)

def calc_price(car_row):
    """
    car_row: pd.Series или словарь с ключами:
        brand, model, year, mileage_km, engine_volume, engine_hp, body_type
    Возвращает: int цена в рублях
    """
    brand = car_row['brand']
    model = car_row['model']
    year = car_row['year']
    mileage = car_row['mileage_km']
    engine_volume = car_row['engine_volume']
    engine_hp = car_row['engine_hp']
    body_type = car_row['body_type']
    
    # Базовая цена
    if brand == "BMW":
        if model[-2:] == "_M":
            base_price = (1500000 + (year-2010)*300000)*1.3
        else:
            base_price = 1500000 + (year-2010)*300000
        if body_type == "SUV":
            base_price *= 1.2
    else:
        base_price = 1000000  # для других брендов
    
    # Формула с пробегом, мощностью, объёмом
    price = base_price - mileage*10 + engine_hp*5000 + engine_volume*20000
    
    # Минимум 200000 и шум ±50 000
    price = int(max(200000, price + np.random.randint(-50000, 50000)))
    return price
