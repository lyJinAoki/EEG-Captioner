from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from eeg_captioner.io import read_jsonl, write_json, write_jsonl
from eeg_captioner.data.dataset import PairDataset, eval_pairs, load_manifest
from eeg_captioner.data.qa_metrics import summarize_predictions
from eeg_captioner.models.language_ops import generate_batch
from .caption_metrics import caption_overlap_metrics, corpus_bleu


def parse_answer(text, options):
    text = " ".join(text.strip().lower().split()).rstrip(".。!！")
    for i, option in enumerate(options):
        if text in {option.lower(), chr(97 + i), f"{chr(97 + i)}. {option.lower()}"}:
            return i
    return -1


def qa_report(truth, predictions, heads, questions):
    if not truth:
        raise ValueError("No valid QA cases in the evaluation split")
    cases = [{"head_index": h, "type": questions[h]["type"]} for h in heads]
    result = summarize_predictions(truth, predictions, cases, questions)
    result["missing_classes"] = {h: [k for k, v in r["class_support"].items() if v == 0]
                                 for h, r in result["by_head"].items() if any(v == 0 for v in r["class_support"].values())}
    result["all_heads_and_classes_present"] = not bool(result["missing_classes"])
    return result


@torch.inference_mode()
def evaluate_probe(model, rows, questions, split, device, per_class=25, batch_size=16):
    model.eval()
    pairs = eval_pairs(rows, questions, split, per_class)
    loader = DataLoader(PairDataset(rows, questions, pairs), batch_size=batch_size)
    truth, pred, heads = [], [], []
    for b in loader:
        logits = model(b["x"].to(device), b["head"].to(device))
        truth.extend(b["label"].tolist()); pred.extend(logits.argmax(-1).tolist()); heads.extend(b["head"].tolist())
    return qa_report(truth, pred, heads, questions)


@torch.inference_mode()
def evaluate_language(components, rows, questions, split, device, per_class=25, caption_cases=500, batch_size=4,
                      qa_max_new_tokens=12, caption_max_new_tokens=320):
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    if min(per_class, caption_cases) < 0 or min(batch_size, qa_max_new_tokens, caption_max_new_tokens) <= 0:
        raise ValueError("Invalid evaluation limits")
    encoder, projector, qwen, tokenizer = components
    encoder.eval(); projector.eval(); qwen.eval()
    loader = DataLoader(PairDataset(rows, questions, eval_pairs(rows, questions, split, per_class)), batch_size=batch_size)
    truth, pred, heads, qa_rows = [], [], [], []
    for b in loader:
        responses = generate_batch(b, encoder, projector, qwen, tokenizer, device, qa_max_new_tokens)
        for i, response in enumerate(responses):
            head = int(b["head"][i]); label = int(b["label"][i])
            prediction = parse_answer(response, questions[head]["options"])
            truth.append(label); pred.append(prediction); heads.append(head)
            qa_rows.append({"id": b["id"][i], "head_index": head, "target": label, "prediction": prediction, "response": response})
    report = qa_report(truth, pred, heads, questions)
    candidates = [i for i, r in enumerate(rows) if r["split"] == split and r.get("caption")]
    rng = np.random.default_rng(42 + (0 if split == "val" else 1) + 90001)
    indices = rng.choice(candidates, len(candidates) if caption_cases == 0 else min(caption_cases, len(candidates)), replace=False)
    pairs = [(int(i), -1) for i in indices]
    cap_rows = []
    for b in DataLoader(PairDataset(rows, questions, pairs), batch_size=batch_size):
        responses = generate_batch(b, encoder, projector, qwen, tokenizer, device, caption_max_new_tokens)
        for i, response in enumerate(responses):
            index = int(b["sample_index"][i]); reference = rows[index]["caption"]
            cap_rows.append({"id": b["id"][i], "reference": reference, "response": response,
                             **caption_overlap_metrics(reference, response)})
    report["caption"] = {"examples": len(cap_rows), "mean_rouge_l_f1": float(np.mean([r["rouge_l_f1"] for r in cap_rows])) if cap_rows else None,
                         "corpus_bleu_4": corpus_bleu(cap_rows, 4)}
    report["protocol"] = "independent_filename_split"
    report["split"] = split
    report["generation"] = {"qa_max_new_tokens": qa_max_new_tokens, "caption_max_new_tokens": caption_max_new_tokens}
    return report, qa_rows, cap_rows


def run(config):
    from eeg_captioner.models.language import load_language
    rows = load_manifest(config["manifest"])
    questions = read_jsonl(Path(__file__).parents[1] / "data/questions_25.jsonl")
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    components = load_language(config["qwen_path"], config["encoder_checkpoint"], config["checkpoint"],
                               device, False, config.get("task", "joint"), config.get("dtype", "bfloat16"))
    report, qa, captions = evaluate_language(components, rows, questions, config.get("split", "test"), device,
                                            config.get("per_class", 25), config.get("caption_cases", 500), config.get("batch_size", 4),
                                            config.get("qa_max_new_tokens", 12), config.get("caption_max_new_tokens", 320))
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    write_jsonl(output / "qa_predictions.jsonl", qa)
    write_jsonl(output / "caption_predictions.jsonl", captions)
    return report
