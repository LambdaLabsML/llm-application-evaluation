import os
import pandas as pd
from datetime import datetime, timedelta
from langsmith import Client


def pull_runs(last_n_days=7):
    client = Client()
    runs = client.list_runs(
        project_name="ml-times-prod",
        start_time=datetime.now() - timedelta(days=7),
        run_type="chain",
        filter='eq(name, "restrictive_relevance_v0")'
    )
    records = []
    for r in runs:
        r = dict(r)
        records.append(r['inputs'])
    records = records[:200]
    return records

def run_to_csv(runs, csv_fpath):
    import pandas as pd
    df = pd.DataFrame(runs).drop_duplicates()
    df.to_csv(csv_fpath, index=False)


if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv()

    # time now in format month-day-year
    timestamp = datetime.now().strftime("%m-%d-%Y")
    runs = pull_runs(2)
    run_to_csv(runs, f"/tmp/runs_{timestamp}.csv")
    print("Done")