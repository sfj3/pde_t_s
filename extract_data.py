import os
import pandas as pd

# Set the path to your data directory
data_dir = './data/'

# Read and concatenate all CSV files into a single DataFrame
dfs = []
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_dir, file))
        dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Print the merged DataFrame
print(data.head())
print(data.shape)
data.to_pickle('combined_data.pkl')


# sort data 
# import pandas as pd

# # Assuming your DataFrame is named 'df'
# df_sorted = df.sort_values(by=['Year', 'Month', 'Day', 'Hour', 'Minute'])

# # Reset the index of the sorted DataFrame
# df_sorted = df_sorted.reset_index(drop=True)

# print(df_sorted)