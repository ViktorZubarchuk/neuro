import pandas as pd
import numpy as np

def generate_animals_dataset(n=1000, random_state=42):
    np.random.seed(random_state)
    
    # --- Категориальные признаки ---
    species = np.random.choice(
        ["Cat", "Dog", "Rabbit", "Parrot", "Hamster"], 
        size=n, 
        p=[0.3, 0.3, 0.2, 0.15, 0.05]  # классы несбалансированы
    )
    
    color = np.random.choice(
        ["Black", "White", "Brown", "Spotted"], 
        size=n
    )
    
    # --- Числовые признаки ---
    age = np.random.randint(1, 15, size=n)  # годы
    weight = np.round(np.random.normal(5, 2, size=n), 1)  # кг
    height = np.round(np.random.normal(30, 10, size=n), 1)  # см
    
    # --- Пропуски ---
    mask_weight = np.random.rand(n) < 0.1
    weight[mask_weight] = np.nan
    
    mask_height = np.random.rand(n) < 0.1
    height[mask_height] = np.nan
    
    # --- Регрессионная цель ---
    # Пример: "Energy level" зависит от веса, роста и вида
    energy_level = (
        100 - weight*3 + height*0.5 + np.random.normal(0, 5, n)
    )
    
    # --- Классификационная цель ---
    # Пример: "Is it a pet-friendly species?" (0/1)
    is_pet_friendly = np.array([1 if s in ["Cat", "Dog", "Rabbit", "Hamster"] else 0 for s in species])
    
    # Добавим немного шума для классификации
    flip_mask = np.random.rand(n) < 0.05
    is_pet_friendly[flip_mask] = 1 - is_pet_friendly[flip_mask]
    
    # --- Собираем DataFrame ---
    df = pd.DataFrame({
        "species": species,
        "color": color,
        "age": age,
        "weight": weight,
        "height": height,
        "energy_level": energy_level,
        "is_pet_friendly": is_pet_friendly
    })
    
    return df

# --- Генерация датасета ---
df_animals = generate_animals_dataset()
df_animals.to_csv("animals_dataset.csv", sep=";", index=False)
