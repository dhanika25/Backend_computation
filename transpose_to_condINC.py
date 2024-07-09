import sqlite3
import pandas as pd

# Paths to the input and output SQLite database files
input_db_path = 'output_restructured.sqlite'
output_db_path = 'conditions_incremental.sqlite'

# Connect to the input SQLite database
conn_input = sqlite3.connect(input_db_path)

# Load the data from the restructured table
data_query = "SELECT * FROM NSE_with_indicators_restructured;"
data = pd.read_sql_query(data_query, conn_input)

# Define a function to check the conditions (example conditions)
def check_conditions(row):
    conditions = []
    if row['Field Name'] == 'C_5_MORE_THAN_10' and row['Field Value'] == 1:
        conditions.append('C_5_MORE_THAN_10')
    if row['Field Name'] == 'C_10_MORE_THAN_5' and row['Field Value'] == 1:
        conditions.append('C_10_MORE_THAN_5')
    return conditions

# Initialize a list to hold the rows for the new table
condition_rows = []

# Iterate through the data and check conditions
for _, row in data.iterrows():
    met_conditions = check_conditions(row)
    for condition in met_conditions:
        condition_rows.append({
            'Condition ID': condition,
            'Stock ID': row['Stock'],
            'Date when condition met': row['Date']
        })

# Create a new DataFrame for the conditions
conditions_df = pd.DataFrame(condition_rows)

# Connect to the new SQLite database
conn_output = sqlite3.connect(output_db_path)

# Store the conditions DataFrame in the new SQLite database
conditions_df.to_sql('Conditions_Incremental', conn_output, if_exists='replace', index=False)

# Optional: Verify the stored data
result_query = "SELECT * FROM Conditions_Incremental LIMIT 50;"
result = pd.read_sql_query(result_query, conn_output)
print(result.head(50))

# Close the connections
conn_input.close()
conn_output.close()
