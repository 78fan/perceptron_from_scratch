import pandas as pd

def preprocess(file: str):
    df = pd.read_csv(file)
    df = df.drop(columns="Cabin")  # too much NaN lol
    df = df.drop(columns="Ticket")  # Not very useful
    df = df.drop(columns="Name")
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Sex"] = pd.factorize(df["Sex"])[0]
    df = pd.get_dummies(df, columns=["Embarked"], dtype=int)
    if file == "test.csv":
        return df.to_numpy()
    categories = df["Survived"].to_numpy()
    features = df.loc[:, df.columns != 'Survived'].to_numpy()
    return categories, features

if __name__ == '__main__':
    preprocess('train.csv')