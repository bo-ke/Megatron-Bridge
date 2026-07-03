"""Compare ERNIE-4.5-VL logits: HF backend vs Megatron backend.

Validates that the Megatron-Bridge conversion produces numerically equivalent
logits to the original HuggingFace model on the same text input.

Must be launched with torchrun:
    torchrun --nproc_per_node=1 --nnodes=1 /tmp/ernie_vl_logit_compare.py \
        --hf-model-path ERNIE-4.5-VL-28B-A3B-Thinking

The script:
1. Loads the HF model, runs a text-only forward pass, saves logits
2. Frees GPU memory
3. Loads the Megatron model via AutoBridge, runs the same forward pass
4. Compares logits: cosine similarity, max abs diff, top-k agreement
"""

import argparse
import gc
import os
import sys
import traceback

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.distributed as dist

from megatron.bridge.models.decorators import torchrun_main


def print_rank_0(msg: str):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg, flush=True)


def run_comparison(
    hf_model_path: str,
    prompt: str = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is",
):
    """Run HF and Megatron forward passes and compare logits."""

    print_rank_0("=" * 70)
    print_rank_0("ERNIE-4.5-VL Logit Comparison: HF vs Megatron")
    print_rank_0("=" * 70)

    # ================================================================
    # Phase 1: HF Model Forward
    # ================================================================
    print_rank_0("\n[Phase 1] HF Model Forward Pass")
    print_rank_0("-" * 50)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    input_ids_list = tokenizer.encode(prompt)
    input_ids_tensor = torch.tensor([input_ids_list], dtype=torch.long, device="cuda")
    seq_len = input_ids_tensor.shape[1]

    # For text-only: token_type_ids = [batch, seq_len + 1] (extra position for shifted labels)
    token_type_ids = torch.zeros(1, seq_len + 1, dtype=torch.long, device="cuda")
    # For text-only: position_ids = [batch, seq_len, 3] with all 3 dims same value
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).unsqueeze(-1).expand(1, seq_len, 3).contiguous()

    print_rank_0(f"  Prompt: {prompt!r}")
    print_rank_0(f"  Token IDs ({seq_len} tokens): {input_ids_list}")

    print_rank_0("  Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    hf_model.eval()
    print_rank_0(f"  HF model loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    print_rank_0("  Running HF forward pass...")
    with torch.no_grad():
        hf_output = hf_model(
            input_ids=input_ids_tensor,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
    hf_logits = hf_output.logits.cpu().float()  # [1, seq_len, vocab_size]
    print_rank_0(f"  HF logits shape: {hf_logits.shape}")
    print_rank_0(f"  HF logits stats: mean={hf_logits.mean():.4f}, std={hf_logits.std():.4f}")

    # Save for later comparison
    torch.save({"logits": hf_logits, "input_ids": input_ids_list}, "/tmp/hf_logits.pt")

    # Free HF model
    del hf_model, hf_output
    gc.collect()
    torch.cuda.empty_cache()
    print_rank_0(f"  HF model freed. GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # ================================================================
    # Phase 2: Megatron Model Forward
    # ================================================================
    print_rank_0("\n[Phase 2] Megatron Model Forward Pass")
    print_rank_0("-" * 50)

    from megatron.bridge import AutoBridge
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    print_rank_0("  Loading HF model and converting to Megatron via AutoBridge...")
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size = 1
    model_provider.pipeline_model_parallel_size = 1
    model_provider.expert_model_parallel_size = 1
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

    # Prepare Megatron input (same tokens)
    mg_input_ids = torch.tensor([input_ids_list], dtype=torch.long, device="cuda")
    mm_token_type_ids = torch.zeros(1, seq_len, dtype=torch.int32, device="cuda")

    # Use a holder dict to capture logits from inside the loss_func
    mg_logits_holder = {}

    class SingleBatchIterator:
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

    def forward_step(data_iterator, model, **kwargs):
        batch = next(data_iterator)
        output = model(
            input_ids=batch["tokens"],
            mm_token_type_ids=batch["mm_token_type_ids"],
            attention_mask=None,
        )

        def loss_func(output_tensor, **kwargs):
            # Capture full logits for comparison
            mg_logits_holder["logits"] = output_tensor.clone().detach()
            dummy_loss = output_tensor.sum() * 0
            return dummy_loss, {"captured": True}

        return output, loss_func

    batch = {"tokens": mg_input_ids, "mm_token_type_ids": mm_token_type_ids}
    fwd_bwd_func = get_forward_backward_func()

    print_rank_0("  Running Megatron forward pass...")
    with torch.no_grad():
        output = fwd_bwd_func(
            forward_step_func=forward_step,
            data_iterator=SingleBatchIterator(batch),
            model=megatron_models,
            num_microbatches=1,
            forward_only=True,
            seq_length=seq_len,
            micro_batch_size=1,
        )

    # Extract Megatron logits
    mg_logits = mg_logits_holder["logits"].cpu().float()
    # Megatron returns [seq_len, batch, vocab] -> transpose to [batch, seq_len, vocab]
    if mg_logits.dim() == 3 and mg_logits.shape[0] == seq_len:
        mg_logits = mg_logits.transpose(0, 1)
    print_rank_0(f"  MG logits shape: {mg_logits.shape}")
    print_rank_0(f"  MG logits stats: mean={mg_logits.mean():.4f}, std={mg_logits.std():.4f}")

    torch.save({"logits": mg_logits, "input_ids": input_ids_list}, "/tmp/mg_logits.pt")

    # ================================================================
    # Phase 3: Compare Logits
    # ================================================================
    print_rank_0("\n[Phase 3] Logit Comparison")
    print_rank_0("-" * 50)

    assert hf_logits.shape == mg_logits.shape, (
        f"Shape mismatch: HF={hf_logits.shape} MG={mg_logits.shape}"
    )

    # 1. Absolute difference
    abs_diff = (hf_logits - mg_logits).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    p99_abs_diff = torch.quantile(abs_diff.flatten(), 0.99).item()

    # 2. Relative difference (avoid div by zero)
    denominator = torch.clamp(hf_logits.abs().max(dim=-1, keepdim=True).values, min=1e-6)
    rel_diff = abs_diff / denominator
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    # 3. Cosine similarity per position
    cos_sim_per_pos = torch.nn.functional.cosine_similarity(
        hf_logits.reshape(seq_len, -1),
        mg_logits.reshape(seq_len, -1),
        dim=-1,
    )
    min_cos_sim = cos_sim_per_pos.min().item()
    mean_cos_sim = cos_sim_per_pos.mean().item()

    # 4. Top-k token agreement
    hf_top5 = hf_logits.topk(5, dim=-1).indices  # [1, seq, 5]
    mg_top5 = mg_logits.topk(5, dim=-1).indices

    top1_match = (hf_top5[..., 0] == mg_top5[..., 0]).float().mean().item()
    top5_overlap = sum(
        len(set(hf_top5[0, i].tolist()) & set(mg_top5[0, i].tolist())) / 5
        for i in range(seq_len)
    ) / seq_len

    # Print results
    print_rank_0(f"\n  Absolute difference:")
    print_rank_0(f"    Max:  {max_abs_diff:.6f}")
    print_rank_0(f"    Mean: {mean_abs_diff:.6f}")
    print_rank_0(f"    P99:  {p99_abs_diff:.6f}")
    print_rank_0(f"\n  Relative difference:")
    print_rank_0(f"    Max:  {max_rel_diff:.6f}")
    print_rank_0(f"    Mean: {mean_rel_diff:.6f}")
    print_rank_0(f"\n  Cosine similarity:")
    print_rank_0(f"    Min:  {min_cos_sim:.6f}")
    print_rank_0(f"    Mean: {mean_cos_sim:.6f}")
    print_rank_0(f"\n  Top-k token agreement:")
    print_rank_0(f"    Top-1 match: {top1_match:.4f} ({int(top1_match * seq_len)}/{seq_len})")
    print_rank_0(f"    Top-5 overlap: {top5_overlap:.4f}")

    # 5. Per-position next-token predictions
    hf_next = hf_logits[0].argmax(dim=-1)
    mg_next = mg_logits[0].argmax(dim=-1)

    print_rank_0(f"\n  Per-position next-token predictions (first 15):")
    for i in range(min(seq_len, 15)):
        hf_tok = tokenizer.decode([hf_next[i].item()])
        mg_tok = tokenizer.decode([mg_next[i].item()])
        match = "OK" if hf_next[i] == mg_next[i] else "DIFF"
        hf_logit_val = hf_logits[0, i, hf_next[i]].item()
        mg_logit_val = mg_logits[0, i, mg_next[i]].item()
        pos_cos = cos_sim_per_pos[i].item()
        print_rank_0(
            f"    pos {i:2d}: HF='{hf_tok}' ({hf_logit_val:+.2f}) | "
            f"MG='{mg_tok}' ({mg_logit_val:+.2f}) | "
            f"cos={pos_cos:.6f} [{match}]"
        )

    # ================================================================
    # Verdict
    # ================================================================
    print_rank_0("\n" + "=" * 70)

    # Thresholds for bf16 model comparison with different attention backends
    PASS = (
        mean_cos_sim > 0.99
        and top1_match >= 0.8
        and max_abs_diff < 5.0  # bf16 can have larger individual differences
    )

    if PASS:
        print_rank_0("RESULT: PASS - Megatron logits closely match HF logits")
    else:
        print_rank_0("RESULT: FAIL - Significant divergence between HF and Megatron logits")
        print_rank_0(f"  Criteria: mean_cos_sim > 0.99 (got {mean_cos_sim:.6f})")
        print_rank_0(f"            top1_match >= 0.8 (got {top1_match:.4f})")
        print_rank_0(f"            max_abs_diff < 5.0 (got {max_abs_diff:.6f})")

    print_rank_0("=" * 70)

    return PASS


@torchrun_main
def _run(hf_model_path: str, prompt: str):
    """Entry point for torchrun-launched logit comparison."""
    passed = run_comparison(hf_model_path=hf_model_path, prompt=prompt)
    if not passed:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="ERNIE-4.5-VL logit comparison: HF vs Megatron")
    parser.add_argument("--hf-model-path", required=True, help="Path to HF model directory")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is",
        help="Text prompt for forward pass comparison",
    )
    args = parser.parse_args()

    _run(hf_model_path=args.hf_model_path, prompt=args.prompt)


if __name__ == "__main__":
    main()
