import pandas as pd
from langsmith import Client

from dotenv import load_dotenv
load_dotenv(".env-langsmith")
client = Client()

def export_csv_to_dataset(csv_fpath):

    df = pd.read_csv(csv_fpath)
    dataset_name = csv_fpath.split("/")[-1].replace(".csv", "")
    dataset = client.create_dataset(dataset_name=dataset_name, description="Relevance filtering dataset")
    for _, row in df.iterrows():
        client.create_example(
            inputs={"title": row["title"], "summary":row["summary"]},
            outputs={"relevant": row["relevant"]},
            dataset_id=dataset.id,
        )
    print(f"Dataset {dataset_name} created at {dataset.id}")

if __name__ == "__main__":

    import os
    datasets = os.listdir(".")
    for dataset in datasets:
        if dataset.endswith(".csv"):
            export_csv_to_dataset(dataset)