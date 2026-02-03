import pandas as pd
import numpy as np

# Number of rows you want
num_rows = 1000

# Random seed for reproducibility
np.random.seed(42)

# Generate features
area = np.random.randint(1500, 6000, size=num_rows)          # square feet
bedrooms = np.random.randint(1, 6, size=num_rows)            # 1-5 bedrooms
bathrooms = np.random.randint(1, 4, size=num_rows)           # 1-3 bathrooms
stories = np.random.randint(1, 4, size=num_rows)             # 1-3 stories
parking = np.random.randint(0, 3, size=num_rows)             # 0-2 parking spots

# Generate price with some randomness
price = (area * 100) + (bedrooms * 10000) + (bathrooms * 5000) + (stories * 2000) + (parking * 3000)
price = price + np.random.normal(0, 20000, size=num_rows)    # adding noise

# Create DataFrame
df = pd.DataFrame({
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "parking": parking,
    "price": price.astype(int)
})

# Save to CSV
df.to_csv("data/housing_large.csv", index=False)

print("Synthetic housing dataset generated: data/housing_large.csv")

