import pandas as pd

df = pd.read_csv('test/books_data.csv')

print(df.head())

df.dropna(inplace=True)

# df.fillna(0, inplace=True)

# df.drop_duplicates(inplace=True)

# df = df.astype(str)
# print(df.head())

df = df.head(1000)

df.to_csv('test.csv', index=False)


