import pandas as pd
import glob
import os

data_path = "data"

# Get all csv files in data folder
files = glob.glob(os.path.join(data_path, "*.csv"))

df_list = []

for file in files:
    print("Loading:", file)
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df_list.append(df)

full_df = pd.concat(df_list, ignore_index=True)

print("Final merged shape:", full_df.shape)

full_df.to_csv("data/full_week.csv", index=False)
print("Saved as data/full_week.csv")