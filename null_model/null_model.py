#imports
import numpy as np
import pandas as pd

#settings
measurement_file = "10142025.csv"
train_test_split = 0.8

#read csv
df = pd.read_csv(f"../data/measurements/{measurement_file}")
df.drop(columns = ["ID", "Student Name", "Notes", "Date Completed", "Unnamed: 24", "Unnamed: 25", "Unnamed: 26", "Unnamed: 27"], inplace = True)
df = df.dropna()

#train test split
all_idx = np.random.permutation(len(df))
train_sz = int(len(df) * train_test_split)
train_idx = all_idx[:train_sz]
test_idx = all_idx[train_sz:]

#model
model = np.zeros(10)

#training
for i in train_idx:
    model += df.iloc[i][:10].to_numpy()
    model += df.iloc[i][10:].to_numpy()
model /= 2 * len(train_idx)

print(model)

#testing
err = np.zeros(10)
for i in test_idx:
    lside = df.iloc[i][:10].to_numpy()
    rside = df.iloc[i][10:].to_numpy()
    err += np.abs((lside - model) / lside)
    err += np.abs((rside - model) / rside)
err /= 2 * len(test_idx)
for i in err:
    print(f'{i.item():.4f}', end = ' ')
print()
