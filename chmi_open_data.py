import pandas as pd

# Load the JSON data from a file
file_path = "aux_data/dly-0-203-0-11742.json"  # Replace with your file path
with open(file_path, "r") as file:
    data = pd.read_json(file)

# Convert the 'values' list into a pandas DataFrame
df = pd.DataFrame(data['data']['values'], columns=["STATION", "ELEMENT", "VTYPE", "DT", "VAL", "FLAG", "QUALITY"])

# Desired date, type, and element(s)
selected_date = "2022-02-03T20:00:00Z"
vtype = "20:00"
elements = ["TMA", "TMI", "P"]  # List of ELEMENT(s) to filter by

# Filter the DataFrame based on the criteria
filtered_rows = df[(df["VTYPE"] == vtype) & (df["DT"] == selected_date) & (df["ELEMENT"].isin(elements))]

# Check if data is found
if not filtered_rows.empty:
    for _, row in filtered_rows.iterrows():
        print(f"Value for ELEMENT {row['ELEMENT']}, date {selected_date}, and type {vtype}: {row['VAL']}")
else:
    print(f"No data found for the specified criteria.")
