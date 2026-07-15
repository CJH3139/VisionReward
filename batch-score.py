# -*- encoding: utf-8 -*-
"""Batch-score a nested experiment tree with VisionReward-Image.

Expected layout:
    <root>/<experiment>/<model>/<scheduler>/<prompt_slug>.png

You also provide a prompts.json mapping filename stem -> prompt text, e.g.
    { "a_red_fox": "a red fox in a snowy forest", ... }

Outputs:
    - results.csv     one row per image (resumable — safe to re-run)
    - summary.csv     mean score grouped by (experiment, model, scheduler)

Example (from repo root):
    python batch-score.py --bf16 \
        --root /content/drive/MyDrive/runs \
        --prompts prompts.json \
        --results results.csv \
        --summary summary.csv
"""
import os
import sys
import csv
import json
import argparse
from collections import defaultdict

import torch
from sat.model.mixins import CachedAutoregressiveMixin

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.utils import llama2_text_processor_inference, get_image_processor, llama3_tokenizer
from utils.models import VisualLlamaEVA

from importlib import import_module
inference_image = import_module("inference-image")
cal_score = inference_image.cal_score

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def walk_tree(root):
    """Yield (image_path, experiment, model, scheduler, prompt_stem) for every image."""
    for experiment in sorted(os.listdir(root)):
        exp_dir = os.path.join(root, experiment)
        if not os.path.isdir(exp_dir):
            continue
        for model in sorted(os.listdir(exp_dir)):
            model_dir = os.path.join(exp_dir, model)
            if not os.path.isdir(model_dir):
                continue
            for scheduler in sorted(os.listdir(model_dir)):
                sched_dir = os.path.join(model_dir, scheduler)
                if not os.path.isdir(sched_dir):
                    continue
                for name in sorted(os.listdir(sched_dir)):
                    stem, ext = os.path.splitext(name)
                    if ext.lower() not in IMAGE_EXTS:
                        continue
                    yield (os.path.join(sched_dir, name), experiment, model, scheduler, stem)


def load_done(results_path):
    """Return set of image_paths already scored (for resume)."""
    if not os.path.exists(results_path):
        return set()
    done = set()
    with open(results_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("score"):
                done.add(row["image_path"])
    return done


def write_summary(results_path, summary_path):
    groups = defaultdict(list)
    with open(results_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("score"):
                continue
            key = (row["experiment"], row["model"], row["scheduler"])
            groups[key].append(float(row["score"]))
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "model", "scheduler", "n", "mean_score"])
        for (exp, mdl, sch), scores in sorted(groups.items()):
            w.writerow([exp, mdl, sch, len(scores), sum(scores) / len(scores)])
    print(f"Wrote summary -> {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root of the experiment tree")
    parser.add_argument("--prompts", required=True, help="JSON: {filename_stem: prompt_text}")
    parser.add_argument("--results", default="results.csv")
    parser.add_argument("--summary", default="summary.csv")
    parser.add_argument("--summary_only", action="store_true", help="Skip scoring; just rebuild summary.csv from results.csv")

    # Passthrough args used by cal_score / model init (mirrors inference-image.py)
    parser.add_argument("--max_length", type=int, default=3328)
    parser.add_argument("--top_p", type=float, default=0.4)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--version", type=str, default="vqa", choices=["chat", "vqa", "chat_old", "base"])
    parser.add_argument("--from_pretrained", type=str, default="THUDM/VisionReward-Image")
    parser.add_argument("--tokenizer_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--stream_chat", action="store_true")
    parser.add_argument("--ques_file", type=str, default="VisionReward_Image/VisionReward_image_qa_select.txt")
    parser.add_argument("--weight_file", type=str, default="VisionReward_Image/weight.json")
    args = parser.parse_args()

    if args.summary_only:
        write_summary(args.results, args.summary)
        return

    with open(args.prompts, "r", encoding="utf-8") as f:
        prompt_map = json.load(f)

    entries = list(walk_tree(args.root))
    if not entries:
        print(f"No images found under {args.root}")
        return

    missing = {stem for _, _, _, _, stem in entries if stem not in prompt_map}
    if missing:
        print(f"WARNING: {len(missing)} filename stems have no prompt in {args.prompts}:")
        for s in sorted(missing)[:10]:
            print(f"  {s}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    done = load_done(args.results)
    todo = [e for e in entries if e[0] not in done and e[4] in prompt_map]
    print(f"Total images: {len(entries)}  already scored: {len(done)}  to score now: {len(todo)}")
    if not todo:
        write_summary(args.results, args.summary)
        return

    print("Loading model...")
    model, model_args = VisualLlamaEVA.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
            deepspeed=None,
            local_rank=0,
            rank=0,
            world_size=1,
            model_parallel_size=1,
            mode="inference",
            skip_init=True,
            use_gpu_initialization=True,
            device="cuda",
            **vars(args),
        ),
    )
    model = model.eval()
    model.add_mixin("auto-regressive", CachedAutoregressiveMixin())
    tokenizer = llama3_tokenizer(args.tokenizer_path, signal_type=args.version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    fieldnames = ["image_path", "experiment", "model", "scheduler", "prompt_stem", "score"]
    write_header = not os.path.exists(args.results)
    with torch.no_grad(), open(args.results, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for i, (img_path, exp, mdl, sch, stem) in enumerate(todo, 1):
            prompt = prompt_map[stem]
            try:
                score = cal_score(args, img_path, prompt, model, text_processor_infer, image_processor)
                print(f"[{i}/{len(todo)}] {score:.4f}  {img_path}")
                writer.writerow({"image_path": img_path, "experiment": exp, "model": mdl,
                                 "scheduler": sch, "prompt_stem": stem, "score": score})
            except Exception as e:
                print(f"[{i}/{len(todo)}] FAILED {img_path}: {e}")
                writer.writerow({"image_path": img_path, "experiment": exp, "model": mdl,
                                 "scheduler": sch, "prompt_stem": stem, "score": ""})
            f.flush()

    write_summary(args.results, args.summary)


if __name__ == "__main__":
    main()