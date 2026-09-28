#!/usr/bin/env bash
set -eo pipefail

echo "=========================================================="
echo "⚡ Validating NVIDIA TensorRT-LLM Multi-GPU Engine"
echo "=========================================================="

python3 -m py_compile trtllm_engine_builder.py
echo "✓ trtllm_engine_builder.py syntax validated."

python3 -m py_compile manim_flow.py
echo "✓ manim_flow.py syntax validated."

python3 -c "
from trtllm_engine_builder import TensorRTEngineConfig
cfg = TensorRTEngineConfig('meta-llama/Llama-3.1-70B', tensor_parallel_size=8)
spec = cfg.compile_engine_spec()
assert spec['tensor_parallel'] == 8, 'TP size mismatch'
print('✓ Unit verification passed: TensorRT-LLM engine spec clean.')
"

echo "=========================================================="
echo "✅ All SRE validation checks passed successfully!"
echo "=========================================================="
