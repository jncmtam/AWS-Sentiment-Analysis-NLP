import pandas as pd
import matplotlib.pyplot as plt
import os

def exploratory_data_analysis(file_path="data/processed/train.csv"):
    df = pd.read_csv(file_path)
    print(df.head())
    print(df["label"].value_counts())

    
    df["label"].value_counts().plot(kind="bar")
    plt.title("Distribution of Sentiment Labels")
    plt.xlabel("Label")
    plt.ylabel("Count")
    
    
    output_dir = "reports/figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(f"{output_dir}/label_distribution.png")
    plt.close()

if __name__ == "__main__":
    exploratory_data_analysis()