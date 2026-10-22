import re
import math

def _normalized_caption(text: str) -> list[str]:
    text = re.sub(r"</?caption>", " ", text, flags=re.IGNORECASE)
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)?", text.casefold())


def _lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_token in right:
        current = [0]
        for index, left_token in enumerate(left, 1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def caption_overlap_metrics(reference: str, prediction: str) -> dict:
    reference_tokens = _normalized_caption(reference)
    prediction_tokens = _normalized_caption(prediction)
    reference_counts = {}
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    overlap = 0
    for token in prediction_tokens:
        count = reference_counts.get(token, 0)
        if count:
            overlap += 1
            reference_counts[token] = count - 1
    precision = overlap / max(len(prediction_tokens), 1)
    recall = overlap / max(len(reference_tokens), 1)
    word_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    lcs = _lcs_length(reference_tokens, prediction_tokens)
    rouge_precision = lcs / max(len(prediction_tokens), 1)
    rouge_recall = lcs / max(len(reference_tokens), 1)
    rouge_l_f1 = (
        2 * rouge_precision * rouge_recall / (rouge_precision + rouge_recall)
        if rouge_precision + rouge_recall else 0.0
    )
    return {
        "word_f1": word_f1,
        "rouge_l_f1": rouge_l_f1,
        "reference_words": len(reference_tokens),
        "prediction_words": len(prediction_tokens),
        "exact_match": reference_tokens == prediction_tokens,
    }


def corpus_bleu(rows: list[dict], max_order: int) -> float:
    """Single-reference corpus BLEU with standard clipped n-gram counts."""

    if max_order <= 0:
        raise ValueError("max_order must be positive")
    clipped = [0] * max_order
    totals = [0] * max_order
    reference_length = 0
    prediction_length = 0
    for row in rows:
        reference = _normalized_caption(row["reference"])
        prediction = _normalized_caption(row["response"])
        reference_length += len(reference)
        prediction_length += len(prediction)
        for order in range(1, max_order + 1):
            ref_counts = {}
            pred_counts = {}
            for index in range(max(len(reference) - order + 1, 0)):
                gram = tuple(reference[index:index + order])
                ref_counts[gram] = ref_counts.get(gram, 0) + 1
            for index in range(max(len(prediction) - order + 1, 0)):
                gram = tuple(prediction[index:index + order])
                pred_counts[gram] = pred_counts.get(gram, 0) + 1
            clipped[order - 1] += sum(
                min(count, ref_counts.get(gram, 0))
                for gram, count in pred_counts.items()
            )
            totals[order - 1] += max(len(prediction) - order + 1, 0)
    if prediction_length == 0 or any(total == 0 for total in totals):
        return 0.0
    precisions = [match / total for match, total in zip(clipped, totals)]
    if any(precision == 0.0 for precision in precisions):
        return 0.0
    brevity_penalty = (
        1.0 if prediction_length > reference_length
        else math.exp(1.0 - reference_length / prediction_length)
    )
    return brevity_penalty * math.exp(
        sum(math.log(value) for value in precisions) / max_order
    )
