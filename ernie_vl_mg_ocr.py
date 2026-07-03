"""Megatron Bridge backend: ERNIE-4.5-VL image OCR inference.

Autoregressive generation using the Megatron-Core model via AutoBridge.
Each step does a full-sequence forward pass (no KV-cache) with vision
re-encoding.  This is slow but correct -- meant for validating the
Megatron conversion against the HF backend.

Usage:
    LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH \
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    .venv/bin/torchrun --nproc_per_node=1 --nnodes=1 \
        ernie_vl_mg_ocr.py \
        --hf-model-path ERNIE-4.5-VL-28B-A3B-Thinking \
        --image-path Repo-Mbridge.png
"""

import argparse
import os
import sys
import time
import types

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Create a fake 'decord' module so the ERNIE processor can be imported
# without the decord package installed.
class _FakeVideoReader:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("decord is not installed; video processing is unavailable")

_decord_fake = types.ModuleType("decord")
_decord_fake.VideoReader = _FakeVideoReader
_decord_fake.cpu = lambda x=0: x
_decord_bridge_fake = types.ModuleType("decord.bridge")
_decord_bridge_fake.set_bridge = lambda *a, **kw: None
sys.modules["decord"] = _decord_fake
sys.modules["decord.bridge"] = _decord_bridge_fake

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoProcessor

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func


def print_rank_0(msg: str):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg, flush=True)


# ============================================================================
# Helpers
# ============================================================================

class SingleBatchIterator:
    """One-shot iterator for get_forward_backward_func."""
    def __init__(self, batch):
        self.batch = batch
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def run_megatron_forward(
    megatron_models,
    fwd_bwd_func,
    input_ids,           # [1, seq_len]
    mm_token_type_ids,   # [1, seq_len]
    pixel_values=None,   # [num_patches, patch_dim] or None
    image_grid_thw=None, # [num_images, 3] or None
    seq_len=None,
):
    """Run one forward pass and return logits [batch, seq_len, vocab]."""

    if seq_len is None:
        seq_len = input_ids.shape[1]

    logits_holder = {}

    def forward_step(data_iterator, model, **kwargs):
        batch = next(data_iterator)
        forward_args = {
            "input_ids": batch["tokens"],
            "mm_token_type_ids": batch["mm_token_type_ids"],
            "moe_mm_token_type_ids": batch["mm_token_type_ids"],
            "attention_mask": None,
        }
        if batch.get("pixel_values") is not None:
            forward_args["pixel_values"] = batch["pixel_values"]
        if batch.get("image_grid_thw") is not None:
            forward_args["image_grid_thw"] = batch["image_grid_thw"]

        output = model(**forward_args)

        def loss_func(output_tensor, **kwargs):
            logits_holder["logits"] = output_tensor.clone().detach()
            dummy_loss = output_tensor.sum() * 0
            return dummy_loss, {"captured": True}

        return output, loss_func

    batch = {
        "tokens": input_ids,
        "mm_token_type_ids": mm_token_type_ids,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

    with torch.no_grad():
        fwd_bwd_func(
            forward_step_func=forward_step,
            data_iterator=SingleBatchIterator(batch),
            model=megatron_models,
            num_microbatches=1,
            forward_only=True,
            seq_length=seq_len,
            micro_batch_size=1,
        )

    return logits_holder["logits"]  # [seq_len, batch, vocab]


# ============================================================================
# Main
# ============================================================================

def run_ocr(
    hf_model_path: str,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    tp: int,
    pp: int,
    ep: int,
):
    print_rank_0("=" * 60)
    print_rank_0("ERNIE-4.5-VL OCR — Megatron Bridge Backend")
    print_rank_0("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Load processor
    # ------------------------------------------------------------------ #
    print_rank_0(f"\n[Step 1] Loading processor from {hf_model_path}...")
    processor = AutoProcessor.from_pretrained(hf_model_path, trust_remote_code=True)
    eos_token_id = processor.tokenizer.eos_token_id
    print_rank_0(f"  EOS token id: {eos_token_id}")

    # ------------------------------------------------------------------ #
    # 2. Load Megatron model via AutoBridge
    # ------------------------------------------------------------------ #
    print_rank_0(f"\n[Step 2] Loading Megatron model via AutoBridge...")
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.params_dtype = torch.bfloat16
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=42)

    megatron_models = model_provider.provide_distributed_model(wrap_with_ddp=False)
    for m in megatron_models:
        if hasattr(m, "config"):
            m.config.deallocate_pipeline_outputs = False
        m.eval()
    print_rank_0(f"  Megatron model built: {len(megatron_models)} component(s)")
    print_rank_0(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    fwd_bwd_func = get_forward_backward_func()

    # ------------------------------------------------------------------ #
    # 3. Prepare image input
    # ------------------------------------------------------------------ #
    print_rank_0(f"\n[Step 3] Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print_rank_0(f"  Image size: {image.size}")

    IMG_START = "<|IMAGE_START|>"
    IMG_END = "<|IMAGE_END|>"
    IMG_PLACEHOLDER = "<|image@placeholder|>"
    BOS = "<|begin_of_sentence|>"
    EOS_STR = "<|end_of_sentence|>"

    text = (
        f"{BOS}"
        f"User: {IMG_START}{IMG_PLACEHOLDER}{IMG_END}\n"
        f"{prompt}{EOS_STR}\n"
        f"Assistant: "
    )
    print_rank_0(f"  Input text: {text[:200]}...")

    # Process text + image through the ERNIE processor
    processor.eval()
    inputs = processor(text=text, images=[image])

    # Move to GPU
    device = "cuda"
    input_ids = inputs["input_ids"].to(device)       # [1, seq_len]
    pixel_values = inputs.get("images")               # [num_patches, patch_dim]
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    grid_thw = inputs.get("grid_thw")                 # [num_images, 3]
    if grid_thw is not None:
        grid_thw = grid_thw.to(device)

    # Build mm_token_type_ids from the HF processor's token_type_ids.
    # HF processor outputs token_type_ids with shape [1, seq_len+1].
    # For Megatron, mm_token_type_ids should be [1, seq_len] where
    # 1 = image placeholder positions, 0 = text (including boundary tokens).
    #
    # The HF token_type_ids marks IMAGE_START/END as type 1 (image), but
    # the Megatron model expects boundary tokens to be type 0 (text).
    # We use the actual image_token_id to identify true image placeholders.
    import json
    with open(os.path.join(hf_model_path, "config.json")) as f:
        hf_cfg = json.load(f)
    image_token_id = hf_cfg.get("image_token_id", 100295)

    mm_token_type_ids = torch.zeros(1, input_ids.shape[1], dtype=torch.int32, device=device)
    image_placeholder_mask = (input_ids == image_token_id)
    mm_token_type_ids[image_placeholder_mask] = 1

    seq_len = input_ids.shape[1]
    num_image_tokens = image_placeholder_mask.sum().item()
    print_rank_0(f"  Input IDs shape: {input_ids.shape}")
    print_rank_0(f"  Pixel values shape: {pixel_values.shape if pixel_values is not None else 'None'}")
    print_rank_0(f"  Grid THW: {grid_thw.tolist() if grid_thw is not None else 'None'}")
    print_rank_0(f"  Image placeholder tokens: {num_image_tokens}")
    print_rank_0(f"  mm_token_type_ids unique values: {mm_token_type_ids.unique().tolist()}")

    # ------------------------------------------------------------------ #
    # 4. Autoregressive generation
    # ------------------------------------------------------------------ #
    print_rank_0(f"\n[Step 4] Generating text (max {max_new_tokens} tokens)...")
    generated_ids = []
    current_input_ids = input_ids.clone()
    current_mm_token_type_ids = mm_token_type_ids.clone()

    t0 = time.time()
    for step in range(max_new_tokens):
        cur_seq_len = current_input_ids.shape[1]

        # Run Megatron forward pass
        logits = run_megatron_forward(
            megatron_models=megatron_models,
            fwd_bwd_func=fwd_bwd_func,
            input_ids=current_input_ids,
            mm_token_type_ids=current_mm_token_type_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            seq_len=cur_seq_len,
        )
        # logits shape: [batch, seq_len, vocab]
        # (GPTModel transposes [s,b,h] -> [b,s,h] when labels=None)

        # Extract last position logits and greedy-decode
        next_token_logits = logits[0, -1, :]  # [vocab]
        next_token_id = next_token_logits.argmax().item()

        # Check for EOS
        if next_token_id == eos_token_id:
            print_rank_0(f"  Step {step + 1}: EOS reached")
            break

        generated_ids.append(next_token_id)

        # Append new token
        new_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        current_input_ids = torch.cat([current_input_ids, new_token], dim=1)

        # Extend mm_token_type_ids with 0 (text token)
        new_type = torch.zeros(1, 1, dtype=torch.int32, device=device)
        current_mm_token_type_ids = torch.cat([current_mm_token_type_ids, new_type], dim=1)

        # Progress reporting
        if (step + 1) % 10 == 0 or step == 0:
            token_text = processor.tokenizer.decode([next_token_id])
            elapsed = time.time() - t0
            tok_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            print_rank_0(
                f"  Step {step + 1}: token={next_token_id} "
                f"'{token_text}' "
                f"({tok_per_sec:.2f} tok/s)"
            )

    elapsed_total = time.time() - t0
    num_tokens = len(generated_ids)
    print_rank_0(f"\n  Generation complete: {num_tokens} tokens in {elapsed_total:.1f}s "
                 f"({num_tokens / elapsed_total:.2f} tok/s)" if elapsed_total > 0 else
                 f"\n  Generation complete: {num_tokens} tokens")

    # ------------------------------------------------------------------ #
    # 5. Decode and output
    # ------------------------------------------------------------------ #
    output_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    print_rank_0("\n" + "=" * 60)
    print_rank_0("ERNIE-4.5-VL OCR Output (Megatron Bridge Backend):")
    print_rank_0("=" * 60)
    print_rank_0(output_text)
    print_rank_0("=" * 60)

    # Save to file for comparison
    output_path = "/tmp/ernie_vl_mg_ocr_output.txt"
    with open(output_path, "w") as f:
        f.write(output_text)
    print_rank_0(f"\nOutput saved to {output_path}")

    # Also save the generated token IDs for detailed comparison
    torch.save({
        "generated_ids": generated_ids,
        "output_text": output_text,
        "num_tokens": num_tokens,
    }, "/tmp/ernie_vl_mg_ocr_result.pt")
    print_rank_0(f"Token IDs saved to /tmp/ernie_vl_mg_ocr_result.pt")

    return output_text


@torchrun_main
def _run(
    hf_model_path: str,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    tp: int,
    pp: int,
    ep: int,
):
    run_ocr(
        hf_model_path=hf_model_path,
        image_path=image_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        tp=tp,
        pp=pp,
        ep=ep,
    )


def main():
    parser = argparse.ArgumentParser(
        description="ERNIE-4.5-VL OCR inference (Megatron Bridge backend)"
    )
    parser.add_argument("--hf-model-path", required=True, help="Path to HF model directory")
    parser.add_argument("--image-path", required=True, help="Path to image file")
    parser.add_argument(
        "--prompt",
        default="请OCR识别这张图片中的所有文字。只输出文字内容，不要额外解释。",
        help="OCR prompt text",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")
    args = parser.parse_args()

    _run(
        hf_model_path=args.hf_model_path,
        image_path=args.image_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
    )


if __name__ == "__main__":
    main()
