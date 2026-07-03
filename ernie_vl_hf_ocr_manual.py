"""HF backend: ERNIE-4.5-VL image OCR inference.

Usage:
    CUDA_VISIBLE_DEVICES=0 python /tmp/ernie_vl_hf_ocr.py \
        --model-path ERNIE-4.5-VL-28B-A3B-Thinking \
        --image-path Repo-Mbridge.png
"""

import argparse
import os
import sys
import types

# Disable torch.compile to avoid triton issues and meta-device errors
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Create a fake 'decord' module so the ERNIE processor can be imported
# without the decord package installed.  Only video processing uses decord;
# image-only inference does not call any decord functions.
# The processor defines `class VideoReaderWrapper(decord.VideoReader)`, so
# we need a real class (not None) that can be subclassed.
class _FakeVideoReader:
    """Placeholder for decord.VideoReader to allow subclassing."""
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
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="ERNIE-4.5-VL HF OCR inference")
    parser.add_argument("--model-path", required=True, help="Path to HF model directory")
    parser.add_argument("--image-path", required=True, help="Path to image file")
    parser.add_argument("--prompt", default="请OCR识别这张图片中的所有文字。只输出文字内容，不要额外解释。",
                        help="OCR prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens")
    args = parser.parse_args()

    model_path = args.model_path
    image_path = args.image_path

    # ---- Load processor and model ----
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to("cuda")
    model.eval()
    # Disable static cache and compile-based prefill to avoid meta-device errors
    model._supports_static_cache = False
    model._supports_cache_class = False
    print(f"Model loaded on GPU. Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # ---- Prepare image input ----
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")

    # The ERNIE processor uses a custom __call__ interface (no chat_template).
    # Text must contain image placeholders:
    #   <|IMAGE_START|><|image@placeholder|><|IMAGE_END|>
    # The processor splits on these, processes each image via _add_image(),
    # and constructs input_ids, token_type_ids, position_ids, images, grid_thw.
    IMG_START = "<|IMAGE_START|>"
    IMG_END = "<|IMAGE_END|>"
    IMG_PLACEHOLDER = "<|image@placeholder|>"
    BOS = "<|begin_of_sentence|>"
    EOS = "<|end_of_sentence|>"

    # Build text with role prefixes (matching processor.role_prefixes)
    text = (
        f"{BOS}"
        f"User: {IMG_START}{IMG_PLACEHOLDER}{IMG_END}\n"
        f"{args.prompt}{EOS}\n"
        f"Assistant: "
    )
    print(f"Input text: {text[:200]}...")

    # Process text + image through the ERNIE processor
    processor.eval()  # disable label generation
    inputs = processor(text=text, images=[image])

    # Move to GPU
    device = "cuda"
    model_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            model_inputs[k] = v.to(device)
    print(f"Input IDs shape: {model_inputs['input_ids'].shape}")
    print(f"Available keys: {list(model_inputs.keys())}")

    # The processor outputs raw pixel values (0-255, uint8) with do_rescale=False,
    # do_normalize=False.  The HF model's vision_forward() asserts bfloat16 dtype.
    # Apply CLIP normalization on-device: rescale to [0,1], normalize with CLIP mean/std.
    if "images" in model_inputs and model_inputs["images"].dtype == torch.uint8:
        images = model_inputs["images"].to(torch.float32) / 255.0
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
        # images shape: [total_patches, C*patch_size^2] where C=3, patch_size=14
        # Expand mean/std to match flattened patch layout: [588]
        patch_size = 14
        pixels_per_patch = patch_size * patch_size
        pixel_mean = clip_mean.repeat_interleave(pixels_per_patch)
        pixel_std = clip_std.repeat_interleave(pixels_per_patch)
        images = (images - pixel_mean) / pixel_std
        model_inputs["images"] = images.to(torch.bfloat16)
        print(f"Normalized images: shape={model_inputs['images'].shape}, dtype={model_inputs['images'].dtype}")

    # ---- Generate ----
    print("Generating text...")
    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=False,  # Avoid DynamicCache compat issue with custom model code
        )

    # Decode only the new tokens
    input_len = model_inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    output_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("ERNIE-4.5-VL OCR Output (HF Backend):")
    print("=" * 60)
    print(output_text)
    print("=" * 60)

    # Also save to file for comparison
    with open("/tmp/ernie_vl_hf_ocr_output.txt", "w") as f:
        f.write(output_text)
    print(f"\nOutput saved to /tmp/ernie_vl_hf_ocr_output.txt")


if __name__ == "__main__":
    main()
