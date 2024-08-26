from langchain_openai import ChatOpenAI
from langchain import hub, prompts
from langsmith import traceable
from langsmith.evaluation import evaluate

@traceable
def get_response(client, prompt, prompt_inputs):
    runnable = prompt | client
    response = runnable.invoke(prompt_inputs)
    return response.content


@traceable
def parse_response(response):
    import json
    response = json.loads(response)
    relevance_score = int(response['relevant'])
    return int(relevance_score > 2), response['comment']


@traceable
def filter_relevance(client, prompt, prompt_inputs):
    response = get_response(client, prompt_inputs=prompt_inputs,prompt=prompt)
    relevance_pred, comment = parse_response(response)
    return {'relevant' : relevance_pred, 'comment' : comment}


def run_experiment(model_client, prompt_name, dataset_name):
    from utils.evaluators import exact_match, true_positive_rate, true_negative_rate

    prompt = hub.pull(prompt_name)
    res = evaluate(
        lambda prompt_inputs: filter_relevance(model_client, prompt, prompt_inputs),
        data=dataset_name,
        evaluators=[exact_match],
        summary_evaluators=[true_positive_rate, true_negative_rate],
        experiment_prefix=f"{model_client.model_name}_{prompt_name}",
        metadata={
            "llm": model_client.model_name,
            "prompt": prompt_name
            }
    )


def run_all_experiments(models, prompts, datasets):
    """
    models: list of model (see below for format) to evaluate
    prompts: list of prompts (by name) to evaluate
    datasets: list of datasets (by name) to evaluate

    models = [
        {model:<model_name>, url=<endpoint_url>, api_key=<api_key>},
        ...
    ]
    """
    for model in models:

        model_name = model["model_name"]
        model_client = ChatOpenAI(
            model=model_name,
            base_url= model["url"],
            api_key=model["api_key"],
            temperature=0.0)

        for dataset_name in datasets:
            for prompt_name in prompts:
                print("Experiment: ", model_name, prompt_name, dataset_name)
                run_experiment(model_client, prompt_name, dataset_name)


if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv(".env-langsmith")

    import json
    models = json.load(open(".env-llm.json"))
    prompt_names = ["filter_concise", "filter_verbose"]
    datasets = ["filter-dataset"]

    run_all_experiments(models, prompt_names, datasets)
