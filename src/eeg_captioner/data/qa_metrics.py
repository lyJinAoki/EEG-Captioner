import numpy as np
from .rules import TYPE_OPTIONS

def classification_metrics(truth, predictions, labels):
    recalls = {}
    supports = {}
    for class_id, label in enumerate(labels):
        members = [index for index, value in enumerate(truth) if value == class_id]
        supports[label] = len(members)
        recalls[label] = (
            sum(predictions[index] == class_id for index in members) / len(members)
            if members else None
        )
    observed = [value for value in recalls.values() if value is not None]
    return {
        "n": len(truth),
        "accuracy": sum(a == b for a, b in zip(truth, predictions)) / max(len(truth), 1),
        "balanced_accuracy": sum(observed) / max(len(observed), 1),
        "class_recall": recalls,
        "class_support": supports,
    }


def summarize_predictions(truth, predictions, cases, questions):
    by_head = {}
    task_head_scores = {name: [] for name in TYPE_OPTIONS}
    for head_index, question in enumerate(questions):
        selected = [index for index, case in enumerate(cases) if case["head_index"] == head_index]
        metrics = classification_metrics(
            [truth[index] for index in selected],
            [predictions[index] for index in selected],
            TYPE_OPTIONS[question["type"]],
        )
        metrics.update(
            id=question["id"],
            type=question["type"],
            condition_key=question["condition_key"],
        )
        by_head[str(head_index)] = metrics
        task_head_scores[question["type"]].append(metrics["balanced_accuracy"])

    by_type = {}
    for type_name, labels in TYPE_OPTIONS.items():
        selected = [index for index, case in enumerate(cases) if case["type"] == type_name]
        pooled = classification_metrics(
            [truth[index] for index in selected],
            [predictions[index] for index in selected],
            labels,
        )
        pooled["macro_head_balanced_accuracy"] = (
            sum(task_head_scores[type_name]) / len(task_head_scores[type_name])
        )
        pooled["heads"] = len(task_head_scores[type_name])
        by_type[type_name] = pooled

    head_scores = [row["balanced_accuracy"] for row in by_head.values()]
    return {
        "n": len(truth),
        "accuracy": sum(a == b for a, b in zip(truth, predictions)) / len(truth),
        "macro_head_balanced_accuracy": sum(head_scores) / len(head_scores),
        "macro_head_definition": (
            f"mean of balanced accuracy over {len(questions)} fixed condition heads"
        ),
        "by_type": by_type,
        "by_head": by_head,
    }
