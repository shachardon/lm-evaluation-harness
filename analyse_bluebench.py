import glob
import os

import pandas as pd


results_path = "outputs/bluebench"
model_done_dirs = [
    dir
    for dir in os.listdir(results_path)
    if os.path.isdir(os.path.join(results_path, dir))
]
model_done_dir = model_done_dirs

results = []
for model_done_dir in model_done_dirs:
    # use glob to find a file that has 'results' in its name
    model_done = glob.glob(os.path.join(results_path, model_done_dir, "*results*"))[-1]
    df = pd.read_json(model_done)

    multi_metric_dict = {
        "legalbench": "accuracy,none",
        "20_newsgroups_short": "accuracy,none",
        "product_help_cfpb": "accuracy,none",
        "rag_response_generation_clapnq": "rag.response_generation.correctness.bert_score.deberta_large_mnli,none",
    }

    for k, v in df["results"].items():
        if "bluebench_" not in k or len(v) < 3:
            print(f"key {k}, removed")
            continue
        print(f"key {k}, kept")

        dataset_name = v["alias"]
        numerical_items = {
            item: value for item, value in v.items() if isinstance(value, float)
        }
        if len(set(numerical_items.values())) == 0:
            raise KeyError("No metric scores here")
        # check if there is more than one distrinct metric score
        elif len(set(numerical_items.values())) > 1:
            # find the right key from multi_metric_dict
            for k, v in multi_metric_dict.items():
                if k in dataset_name:
                    chosen_metric = v

            numerical_items = {chosen_metric: numerical_items[chosen_metric]}

        results.append(
            {
                "model": model_done.split("bluebench/")[-1].split("/results")[0],
                "score": list(numerical_items.values())[0],
                "metric": list(numerical_items.keys())[0],
                "scenario": dataset_name.split("bluebench_")[-1],
            }
        )

df = pd.DataFrame(results)
df = df.pivot(index='model', columns='scenario', values='score')
df.to_csv('outputs/bluebench/processsed_results.csv')
