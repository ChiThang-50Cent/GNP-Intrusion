import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data():
    df = pd.read_csv("./UNSW_NB15_training-set.csv")

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

    ct = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(), [1, 2, 3])], remainder="passthrough"
    )
    X = np.array(ct.fit_transform(X))

    sc = StandardScaler()

    X[:, 18:] = sc.fit_transform(X[:, 18:])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    costs = np.ones((X.shape[1],))

    return X_train, y_train, X_test, y_test, costs
