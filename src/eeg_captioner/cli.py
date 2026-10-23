"""Lazy imports keep help and data-only commands independent of LLM dependencies."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="eeg-captioner")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preprocess", help="TUEG EDF -> subject H5 in microvolts")
    p.add_argument("--edf-root", required=True); p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=1); p.add_argument("--max-subjects", type=int)
    p = sub.add_parser("features", help="Extract early/middle/last numerical features", add_help=False)
    p = sub.add_parser("summaries", help="Render numerical summaries", add_help=False)
    p = sub.add_parser("source-artifacts", help="Build optional MNE source-space artifacts")
    p.add_argument("--config", required=True)
    p = sub.add_parser("annotate", help="Numerical summaries -> API-generated captions")
    for name in ("summaries", "output", "base-url", "model"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--api-key-env", default="EEG_CAPTION_API_KEY"); p.add_argument("--limit", type=int)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--compact-input", action="store_true")
    p = sub.add_parser("verify-captions", help="Verify original caption archive hashes and counts")
    p.add_argument("--captions", required=True)
    p = sub.add_parser("prepare", help="Build portable manifest and 25 QA labels")
    for name in ("processed", "summaries", "captions", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--numeric-only", action="store_true"); p.add_argument("--seed", type=int, default=42)
    for command in ("train", "evaluate"):
        p = sub.add_parser(command); p.add_argument("--config", required=True)
    p = sub.add_parser("export", help="Export trusted research encoder and final adapter")
    p.add_argument("--encoder", required=True); p.add_argument("--adapter", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("export-encoder", help="Extract encoder from a trusted classifier checkpoint")
    p.add_argument("--source", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("infer", help="Caption or answer a question about one preprocessed segment")
    p.add_argument("--bundle", required=True); p.add_argument("--qwen-path", required=True)
    p.add_argument("--eeg-h5", required=True); p.add_argument("--trial", type=int, default=0)
    p.add_argument("--start-sample", type=int, default=0); p.add_argument("--question")
    p.add_argument("--device", default="cuda"); p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-new-tokens", type=int, default=320)
    args, rest = parser.parse_known_args(argv)
    if args.command in {"features", "summaries"}:
        original = sys.argv
        try:
            sys.argv = ["eeg-captioner " + args.command, *rest]
            if args.command == "features":
                from .features.pipeline_runner import main as entry
            else:
                from .features.summary import main as entry
                if "--resume" not in rest:
                    sys.argv.append("--resume")
            entry()
        finally:
            sys.argv = original
        return
    if rest:
        parser.error("unrecognized arguments: " + " ".join(rest))
    if args.command == "preprocess":
        from .preprocessing.run import run
        run(args.edf_root, args.output, args.workers, args.max_subjects)
    elif args.command == "source-artifacts":
        from .features.source_artifacts import build_source_artifacts
        from .io import load_config
        build_source_artifacts(load_config(args.config)["features"]["source_network"])
    elif args.command == "annotate":
        from .data.annotate import annotate
        annotate(args.summaries, args.output, args.base_url, args.model, args.api_key_env, args.limit,
                 args.temperature, args.max_tokens, args.compact_input)
    elif args.command == "verify-captions":
        from .data.captions import verify_captions
        print(json.dumps(verify_captions(args.captions)))
    elif args.command == "prepare":
        from .data.prepare import prepare
        print(prepare(args.processed, args.summaries, args.captions, args.output, not args.numeric_only, args.seed))
    elif args.command in {"train", "evaluate"}:
        from .io import load_config
        config = load_config(args.config)
        if args.command == "train":
            from .training.engine import train
            train(config)
        else:
            from .evaluation.run import run
            print(json.dumps(run(config), indent=2))
    elif args.command == "export":
        from .export import export_bundle
        print(export_bundle(args.encoder, args.adapter, args.output))
    elif args.command == "export-encoder":
        from .models.checkpoint import export_encoder
        export_encoder(args.source, args.output)
    elif args.command == "infer":
        from .data.dataset import load_eeg
        from .models.language import Captioner, CAPTION_QUESTION
        bundle = Path(args.bundle)
        model = Captioner(args.qwen_path, bundle / "encoder.pt", bundle, args.device, args.dtype)
        eeg = load_eeg({"eeg_path": args.eeg_h5, "trial_index": args.trial, "start_sample": args.start_sample})
        print(model.generate(eeg, args.question or CAPTION_QUESTION, args.max_new_tokens)[0])

