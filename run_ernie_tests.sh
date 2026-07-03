#!/bin/bash
set -xeuo pipefail

export CUDA_VISIBLE_DEVICES="0,1"
export LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /root/paddlejob/workspace/env_run/workspace/Megatron-Bridge

.venv/bin/python -m pytest \
  -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
  tests/functional_tests/test_groups/models/ernie_vl/test_ernie45_vl_conversion.py \
  2>&1 | tee /tmp/ernie_vl_test_output.log
