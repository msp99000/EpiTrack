import pandas as pd

# Replace 'file_path.pkl' with the path to your .pkl file
file_path = 'cnn_pred.pkl'

# Load the DataFrame from a .pkl file
df = pd.read_pickle(file_path)

# Now you can work with the DataFrame
print(df)
