import os
import pandas as pd

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer
)


# ======================================================
# Create Dataset Folder
# ======================================================

os.makedirs("datasets", exist_ok=True)


# ======================================================
# Convert Dataset to DataFrame
# ======================================================

def dataset_to_dataframe(dataset):

    df = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )

    df["Target"] = dataset.target

    return df


# ======================================================
# Dataset Information
# ======================================================

def dataset_information(name, dataset):

    df = dataset_to_dataframe(dataset)

    print("=" * 70)
    print(f"{name.upper()} DATASET")
    print("=" * 70)

    print("\nShape")
    print(df.shape)

    print("\nFeature Names")
    for feature in dataset.feature_names:
        print("-", feature)

    print("\nTarget Names")

    try:
        for target in dataset.target_names:
            print("-", target)

    except:
        print("Target names not available.")

    print("\nFirst Five Rows")
    print(df.head())

    print("\nLast Five Rows")
    print(df.tail())

    print("\nDataset Information")
    print(df.info())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nStatistical Summary")
    print(df.describe())

    print("\nClass Distribution")

    print(df["Target"].value_counts())

    return df


# ======================================================
# Export CSV
# ======================================================

def export_dataset(df, filename):

    path = os.path.join("datasets", filename)

    df.to_csv(path, index=False)

    print(f"\nDataset exported successfully -> {path}")


# ======================================================
# Explore Iris
# ======================================================

def iris_dataset():

    dataset = load_iris()

    df = dataset_information("Iris", dataset)

    export_dataset(df, "iris.csv")


# ======================================================
# Explore Wine
# ======================================================

def wine_dataset():

    dataset = load_wine()

    df = dataset_information("Wine", dataset)

    export_dataset(df, "wine.csv")


# ======================================================
# Explore Breast Cancer
# ======================================================

def breast_cancer_dataset():

    dataset = load_breast_cancer()

    df = dataset_information("Breast Cancer", dataset)

    export_dataset(df, "breast_cancer.csv")


# ======================================================
# Menu
# ======================================================

def menu():

    while True:

        print("\n" + "=" * 60)
        print("        ML DATASET EXPLORER")
        print("=" * 60)

        print("1. Explore Iris Dataset")
        print("2. Explore Wine Dataset")
        print("3. Explore Breast Cancer Dataset")
        print("4. Explore All Datasets")
        print("5. Exit")

        choice = input("\nEnter Choice: ")

        if choice == "1":
            iris_dataset()

        elif choice == "2":
            wine_dataset()

        elif choice == "3":
            breast_cancer_dataset()

        elif choice == "4":

            iris_dataset()
            wine_dataset()
            breast_cancer_dataset()

        elif choice == "5":

            print("\nThank you.")
            break

        else:

            print("Invalid Choice.")


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":
    menu()