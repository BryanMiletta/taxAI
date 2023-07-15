import pandas as pd

csv_file = 'db/updated_csv.csv'
json_file = 'db/json_file.json'

# Read CSV file
df = pd.read_csv(csv_file)

# Convert DataFrame to JSON
df.to_json(json_file, orient='records', lines=True)
