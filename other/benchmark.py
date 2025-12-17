from datasets import load_dataset
import torch
from pathlib import Path
from cobra import load

hf_token = Path("/home/agf64/project/thinking_cobra/.hf_token").read_text().strip()
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU")
else:
    device = torch.device('cpu')
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub)
model_id = "cobra+3b"
vlm = load(model_id, hf_token=hf_token)
vlm.to(device, dtype=dtype)

dataset = load_dataset("lmms-lab/COCO-Caption", split="val")
user_prompt = "Please carefully observe the image and come up with a caption for the image."

prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()

ref_captions = {}
generated_captions = {}

for idx, example in enumerate(dataset):
    print(idx)

    image_id = idx
    image = example["image"]

    if image.mode != "RGB":
        image = image.convert("RGB")

    caption = example["answer"]
    ref_captions[image_id] = caption
    generated_text = vlm.generate(
        image,
        prompt_text,
        use_cache=True,
        do_sample=True,
        temperature=0.4,
        max_new_tokens=512,
    )
    generated_captions[idx] = [generated_text]

# from matplotlib import pyplot as plt

import io
from contextlib import redirect_stdout

from bert_score import score
from pycocoevalcap.bleu.bleu import Bleu


def compute_bleu_scores(ref_captions, generated_captions):
    bleu_scorer = Bleu(4)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        bleu_scores, _ = bleu_scorer.compute_score(ref_captions, generated_captions)
    verbose_output = buffer.getvalue().strip()
    if verbose_output:
        print(verbose_output)
    return (
        {
            "BLEU-1": bleu_scores[0],
            "BLEU-2": bleu_scores[1],
            "BLEU-3": bleu_scores[2],
            "BLEU-4": bleu_scores[3],
        },
        verbose_output,
    )


def compute_bertscore_metrics(ref_captions, generated_captions, bert_model="bert-base-uncased", lang="en"):
    sorted_ids = sorted(generated_captions.keys())
    preds = [generated_captions[idx][0] for idx in sorted_ids]
    refs = [ref_captions[idx] for idx in sorted_ids]

    P, R, F1 = score(preds, refs, model_type=bert_model, lang=lang)
    return {
        "Precision": P.mean().item(),
        "Recall": R.mean().item(),
        "F1": F1.mean().item(),
    }


def benchmark_caption_metrics(
    ref_captions,
    generated_captions,
    output_path: str = "caption_metrics_output.txt",
    bert_model: str = "bert-base-uncased",
    lang: str = "en",
):
    """Compute BLEU and BERTScore metrics and write a shared report."""
    bleu_metrics, bleu_verbose = compute_bleu_scores(ref_captions, generated_captions)
    bert_metrics = compute_bertscore_metrics(ref_captions, generated_captions, bert_model, lang)

    sorted_ids = sorted(generated_captions.keys())
    with open(output_path, "w") as f:
        f.write(f"Number of generated captions: {len(generated_captions)}\n\n")
        f.write("Metric Summary (averaged across dataset):\n")
        f.write(
            "BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | BERTScore-P | BERTScore-R | BERTScore-F1\n"
        )
        f.write(
            f"{bleu_metrics['BLEU-1']:.4f} | {bleu_metrics['BLEU-2']:.4f} | {bleu_metrics['BLEU-3']:.4f} | {bleu_metrics['BLEU-4']:.4f} | "
            f"{bert_metrics['Precision']:.4f} | {bert_metrics['Recall']:.4f} | {bert_metrics['F1']:.4f}\n\n"
        )

        if bleu_verbose:
            f.write("BLEU Raw Statistics:\n")
            f.write(bleu_verbose + "\n\n")

        for img_id in sorted_ids:
            f.write(f"Image ID: {img_id}\n")
            f.write("Generated caption:\n")
            f.write(f"  {generated_captions[img_id][0]}\n")
            f.write("Reference captions:\n")
            for ref in ref_captions[img_id]:
                f.write(f"  - {ref}\n")
            f.write("\n")

    print(f"BLEU and BERTScore metrics saved to {output_path}")

    return {"BLEU": bleu_metrics, "BERTScore": bert_metrics}

metrics = benchmark_caption_metrics(
    ref_captions,
    generated_captions,
    output_path="caption_metrics_output.txt",
)
