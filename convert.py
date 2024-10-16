import json
import csv

# Read JSON data from the file
with open('interim_results_500.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Open a CSV file for writing
with open('interim_results_500.csv', 'w', newline='', encoding='utf-8') as csv_file:
    # Create a CSV writer object
    writer = csv.writer(csv_file)
    
    # Write header row
    header = data[0].keys()
    writer.writerow(header)
    
    # Write data rows
    for item in data:
        writer.writerow(item.values())

    print("CSV file has been created successfully.")
    
