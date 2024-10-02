"""
required
___

Serialized objects (joblib) requires external objects (i.e. functions)
to be in the same `namespace` to be used!

This defines the `drop_missing_convert_dt` func, used in the pipeline!

"""


import pandas as pd

def drop_missing_convert_dt(df: pd.DataFrame):
    """Function to transform data types & drop rows with missing values"""

    # Reorder the input!
    df = df[df.columns]  # It fits on training data

    # In case Null comes in future input!
    df.dropna(inplace=True)

    # Drop tenure 0 rows and TotalCharges = " " rows
    df.drop(df[df.tenure == 0].index, inplace=True)
    df.drop(df[df["TotalCharges"] == " "].index, inplace=True)

    # Convert MonthlyCharges to number
    df["TotalCharges"] = df["TotalCharges"].astype("float")

    return df