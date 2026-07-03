"""HF backend: ERNIE-4.5-VL image OCR inference using model.generate().

Usage:
    CUDA_VISIBLE_DEVICES=0 python ernie_vl_hf_ocr.py \
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


# ============================================================================
# Monkey-patches needed for model.generate() compatibility with ERNIE-4.5-VL
# on transformers >= 5.x (new DynamicCache architecture).
# ============================================================================

def _apply_generate_patches(model):
    """Apply monkey-patches to make model.generate() work with ERNIE-4.5-VL.

    Issues addressed:
    1. DynamicCache has no __getitem__ — the ERNIE model code accesses
       past_key_values[layer_idx][0].shape[1] which requires subscription.
    2. _update_model_kwargs_for_generation signature mismatch — transformers
       5.x passes num_new_tokens=1 kwarg but ERNIE's override doesn't accept it.
    3. ERNIE model's Ernie4_5_Model.forward() accesses past_key_values[0] to
       detect cache presence and past_key_values[idx] per layer — both need
       __getitem__ on the cache object.
    """
    from transformers.cache_utils import DynamicCache

    # --- Patch 1: DynamicCache.__getitem__ ---
    # The ERNIE model does: past_key_values[0][0].shape[1] (to get cache_length)
    # and past_key_values[idx] (to get per-layer cache).
    # DynamicCache stores layers in self.layers list. Each DynamicLayer has
    # .keys and .values tensors. We return [keys, values] to match the
    # tuple-style cache the model expects.
    if not hasattr(DynamicCache, '__getitem__'):
        def _dynamic_cache_getitem(self, idx):
            if idx < len(self.layers) and self.layers[idx].is_initialized:
                return [self.layers[idx].keys, self.layers[idx].values]
            return None
        DynamicCache.__getitem__ = _dynamic_cache_getitem
        print("  Patched DynamicCache.__getitem__")

    # Also need __len__ for the model's `past_key_values is not None` checks
    # and iteration patterns.
    if not hasattr(DynamicCache, '__len__'):
        def _dynamic_cache_len(self):
            return len(self.layers)
        DynamicCache.__len__ = _dynamic_cache_len
        print("  Patched DynamicCache.__len__")

    # --- Patch 2: Fix _update_model_kwargs_for_generation signature ---
    # Transformers 5.x calls: model._update_model_kwargs_for_generation(
    #     outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1)
    # But ERNIE's override signature is: (self, outputs, model_kwargs, is_encoder_decoder=False)
    # which raises TypeError on the unexpected num_new_tokens kwarg.
    model_cls = type(model)
    orig_update = model_cls._update_model_kwargs_for_generation

    def _patched_update_model_kwargs(self, outputs, model_kwargs, is_encoder_decoder=False, **kwargs):
        return orig_update(self, outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder)

    model_cls._update_model_kwargs_for_generation = _patched_update_model_kwargs
    print("  Patched _update_model_kwargs_for_generation to accept **kwargs")


def _fix_model_tensors(model):
    """Fix non-parameter tensor attributes that model.to('cuda') doesn't move.

    from_pretrained() creates modules on meta device; nn.Parameter/buffers get
    materialized, but plain tensor attributes (inv_freq, experts_type_mask,
    expert_usage, one, zero, eps, etc.) remain on meta/CPU.
    Walk all modules and move stray tensors to CUDA.
    """
    fixed_count = 0
    for name, module in model.named_modules():
        for attr_name in list(vars(module).keys()):
            obj = getattr(module, attr_name)
            if isinstance(obj, torch.Tensor) and not isinstance(obj, torch.nn.Parameter):
                if obj.device.type == "meta":
                    new_tensor = torch.zeros(
                        obj.shape, dtype=obj.dtype, device="cuda"
                    )
                    setattr(module, attr_name, new_tensor)
                    fixed_count += 1
                elif obj.device.type == "cpu":
                    setattr(module, attr_name, obj.to("cuda"))
                    fixed_count += 1
            elif isinstance(obj, list):
                new_list = []
                changed = False
                for item in obj:
                    if isinstance(item, torch.Tensor):
                        if item.device.type == "meta":
                            new_list.append(torch.zeros(item.shape, dtype=item.dtype, device="cuda"))
                            changed = True
                        elif item.device.type == "cpu":
                            new_list.append(item.to("cuda"))
                            changed = True
                        else:
                            new_list.append(item)
                    else:
                        new_list.append(item)
                if changed:
                    setattr(module, attr_name, new_list)
                    fixed_count += 1
    print(f"  Moved {fixed_count} non-parameter tensors to CUDA")

    # VisionRotaryEmbedding: inv_freq was recreated as zeros above — recompute
    # properly, and monkey-patch forward() so `seq` is created on the right device.
    for name, module in model.named_modules():
        if module.__class__.__name__ == "VisionRotaryEmbedding":
            dim = module.inv_freq.shape[0] * 2
            theta = 10000.0
            module.inv_freq = 1.0 / theta ** (
                torch.arange(start=0, end=dim, step=2, dtype=torch.float32, device="cuda") / dim
            )
            print(f"  Recreated {name}.inv_freq on CUDA (dim={dim})")

            def _patched_forward(self, seqlen):
                seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
                freqs = torch.outer(input=seq, vec2=self.inv_freq)
                return freqs

            import types as _types
            module.forward = _types.MethodType(_patched_forward, module)
            print(f"  Patched {name}.forward() for device-aware seq creation")

    # Top2Gate / TopKGate: experts_type_ids and experts_type_mask were recreated
    # as all-zeros above, but they encode expert-group membership and must be
    # recomputed from the model config.
    gate_fix_count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ in ("Top2Gate", "TopKGate"):
            num_experts_list = getattr(module, "num_experts", None)
            if isinstance(num_experts_list, (list, tuple)) and hasattr(module, "experts_type_mask"):
                total = sum(num_experts_list)
                experts_ids = torch.zeros([total], dtype=torch.int64, device="cuda")
                offset = 0
                for i, expert_num in enumerate(num_experts_list):
                    experts_ids[offset:offset + expert_num] = i
                    offset += expert_num
                module.experts_type_ids = experts_ids
                module.experts_type_mask = [(experts_ids == i) for i in range(len(num_experts_list))]
                gate_fix_count += 1
    if gate_fix_count > 0:
        print(f"  Recomputed experts_type_ids/mask for {gate_fix_count} gate modules")


def main():
    parser = argparse.ArgumentParser(description="ERNIE-4.5-VL HF OCR inference (generate API)")
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

    # Fix non-parameter tensors stuck on meta/CPU devices
    _fix_model_tensors(model)

    # Apply monkey-patches for generate() compatibility
    _apply_generate_patches(model)

    print(f"Model loaded on GPU. Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # ---- Prepare image input ----
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")

    IMG_START = "<|IMAGE_START|>"
    IMG_END = "<|IMAGE_END|>"
    IMG_PLACEHOLDER = "<|image@placeholder|>"
    BOS = "<|begin_of_sentence|>"
    EOS = "<|end_of_sentence|>"

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
    # Apply CLIP normalization on-device.
    if "images" in model_inputs and model_inputs["images"].dtype == torch.uint8:
        images_tensor = model_inputs["images"].to(torch.float32) / 255.0
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
        patch_size = 14
        pixels_per_patch = patch_size * patch_size
        pixel_mean = clip_mean.repeat_interleave(pixels_per_patch)
        pixel_std = clip_std.repeat_interleave(pixels_per_patch)
        images_tensor = (images_tensor - pixel_mean) / pixel_std
        model_inputs["images"] = images_tensor.to(torch.bfloat16)
        print(f"Normalized images: shape={model_inputs['images'].shape}, dtype={model_inputs['images'].dtype}")

    # NOTE: Do NOT pad token_type_ids here.
    # prepare_inputs_for_generation() in the model code pads it with +1 zero.
    # If we pre-pad AND the model also pads, we get shape mismatches.
    # The token_type_ids from the processor has shape [1, seq_len].
    # The model forward() asserts token_type_ids.shape[1] == input_ids.shape[1] + 1.
    # prepare_inputs_for_generation() handles this by doing:
    #   token_type_ids = cat([token_type_ids, zeros([len, 1])], dim=-1)
    # So we pass the UN-padded version and let the model pad it.

    # However, for the very first call (prefill), the model.generate() flow
    # calls prepare_inputs_for_generation() which does the padding.
    # We need to pass token_type_ids with shape [1, seq_len] (not seq_len+1).
    print(f"token_type_ids shape (before generate): {model_inputs.get('token_type_ids', 'N/A')}")

    # ---- Generate using model.generate() ----
    print("Generating text with model.generate()...")
    eos_token_id = processor.tokenizer.eos_token_id

    generate_kwargs = {
        "input_ids": model_inputs["input_ids"],
        "images": model_inputs.get("images"),
        "token_type_ids": model_inputs.get("token_type_ids"),
        "position_ids": model_inputs.get("position_ids"),
        "grid_thw": model_inputs.get("grid_thw"),
        "image_type_ids": model_inputs.get("image_type_ids"),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,  # greedy
        "eos_token_id": eos_token_id,
        "use_cache": True,
    }
    # Remove None values
    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)

    # output_ids includes the input_ids prefix; extract only generated tokens
    input_len = model_inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_len:].tolist()

    print(f"  Generation complete: {len(generated_ids)} tokens")
    output_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("ERNIE-4.5-VL OCR Output (HF Backend - generate API):")
    print("=" * 60)
    print(output_text)
    print("=" * 60)

    # Also save to file for comparison
    with open("/tmp/ernie_vl_hf_ocr_output.txt", "w") as f:
        f.write(output_text)
    print(f"\nOutput saved to /tmp/ernie_vl_hf_ocr_output.txt")


if __name__ == "__main__":
    main()
