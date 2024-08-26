from langsmith.evaluation import EvaluationResult
from langsmith.evaluation import run_evaluator

@run_evaluator
def exact_match(run, example):
    prediction = int(run.outputs.get("relevant"))
    target = int(example.outputs.get("relevant"))
    match = prediction == target
    return EvaluationResult(key="accuracy", score=match)


def true_positive_rate(runs, examples):
    true_positives = 0
    actual_positives = 0
    for run, example in zip(runs, examples):
        reference = int(example.outputs.get("relevant"))
        prediction = int(run.outputs.get("relevant"))
        
        if reference == 1:
            actual_positives += 1
            if prediction == reference:
                true_positives += 1
    
    if actual_positives == 0:
        return

    score = true_positives / actual_positives
    return EvaluationResult(key="true_positive_rate", score=score)

def true_negative_rate(runs, examples):
    true_negatives = 0
    actual_negatives = 0
    for run, example in zip(runs, examples):
        reference = int(example.outputs.get("relevant"))
        prediction = int(run.outputs.get("relevant"))
        
        if reference == 0:
            actual_negatives += 1
            if prediction == 0:
                true_negatives += 1

    if actual_negatives == 0:
        return

    score = true_negatives / actual_negatives
    return EvaluationResult(key="true_negative_rate", score=score)