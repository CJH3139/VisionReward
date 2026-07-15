# -*- encoding: utf-8 -*-
"""Batch-score every image in a folder against a shared prompt.

Example:
    python score-images.py --bf16 --image_dir path/to/imgs --prompt "a red fox"
"""
import os
import sys
import csv
import argparse
import torch
from sat.model.mixins import CachedAutoregressiveMixin

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.utils import llama2_text_processor_inference, get_image_processor, llama3_tokenizer
from utils.models import VisualLlamaEVA

# Reuse the scoring routine already defined in inference-image.py so weights,
# mask logic, and alignment scoring stay in one place.
from importlib import import_module
inference_image = import_module("inference-image")
cal_score = inference_image.cal_score

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(image_dir):
    files = []
    for name in sorted(os.listdir(image_dir)):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            files.append(os.path.join(image_dir, name))
    return files


def main():
    parser = argparse.ArgumentParser()
    # Required inputs
    parser.add_argument("--image_dir", required=True, help="Folder containing images to score")
    parser.add_argument("--prompt", required=True, help="Text prompt shared by all images")
    parser.add_argument("--output_csv", default="scores.csv", help="Where to write image_path,score")

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

    images = collect_images(args.image_dir)
    if not images:
        print(f"No images found in {args.image_dir}")
        return
    print(f"Found {len(images)} images. Loading model...")

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

    results = []
    with torch.no_grad(), open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "score"])
        for img in images:
            try:
                score = cal_score(args, img, args.prompt, model, text_processor_infer, image_processor)
                print(f"{img}\t{score}")
                writer.writerow([img, score])
                results.append((img, score))
                f.flush()
            except Exception as e:
                print(f"Failed on {img}: {e}")
                writer.writerow([img, ""])

    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 5:")
        for img, s in results[:5]:
            print(f"  {s:.4f}  {img}")
    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()