import pandas as pd

# Load your Excel file
df = pd.read_excel("car_database.xlsx")

# Show column names
print("Columns in Excel:", df.columns.tolist())

# Show first 3 rows
print(df.head(3))
