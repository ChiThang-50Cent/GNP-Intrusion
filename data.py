import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    df = pd.read_csv(path)

    list_drop = ["id", "attack_cat"]
    df.drop(list_drop, axis=1, inplace=True)

    # Clamp extreme Values
    df_numeric = df.select_dtypes(include=[np.number])
    for feature in df_numeric.columns:
        if (
            df_numeric[feature].max() > 10 * df_numeric[feature].median()
            and df_numeric[feature].max() > 10
        ):
            df[feature] = np.where(
                df[feature] < df[feature].quantile(0.95),
                df[feature],
                df[feature].quantile(0.95),
            )

    df_numeric = df.select_dtypes(include=[np.number])
    for feature in df_numeric.columns:
        if df_numeric[feature].nunique() > 50:
            if df_numeric[feature].min() == 0:
                df[feature] = np.log(df[feature] + 1)
            else:
                df[feature] = np.log(df[feature])

    df_cat = df.select_dtypes(exclude=[np.number])
    for feature in df_cat.columns:
        if df_cat[feature].nunique() > 6:
            df[feature] = np.where(
                df[feature].isin(df[feature].value_counts().head().index),
                df[feature],
                "-",
            )

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    for col in list(X.select_dtypes(exclude=np.number).columns):
        lb = LabelEncoder()
        lb.fit(X[col])
        X[col] = lb.transform(X[col])

    sc = StandardScaler()
    X = sc.fit_transform(X)

    return (X, y.values)

if __name__ == "__main__":
    for x in ['./UNSW_NB15_training-set.csv', './UNSW_NB15_testing-set.csv']:
        X, y = load_data(x)
        print(type(X), type(y))


