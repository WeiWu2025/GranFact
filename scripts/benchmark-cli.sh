#!/usr/bin/env bash
set -euo pipefail

# Release-friendly CLI wrapper for the 3-stage Grain-HAL benchmark.
# All machine-specific paths are intentionally empty by default and must be
# provided via CLI options.

# ---------------------------
# Defaults
# ---------------------------
dataset_root=""
output_root="./model_outputs"

model_to_evaluate="your_model_name"
stage1_backend="local"  # local | api
model_to_evaluate_ckpt=""

# Bash glob patterns. Models matching these patterns use Stage1 vLLM by default.
vllm_model_patterns=(
  "qwen3*"
)

stage1_gpus="0"
stage1_vllm_tensor_parallel_size=1
stage1_vllm_gpu_memory_utilization=0.8
stage1_vllm_dtype="auto"

api_model=""
api_base_url=""
api_key_env_name="OPENAI_API_KEY"
reasoning_effort="none"
stage1_max_new_tokens=2048

judge_model="your_judge_model_name"
judge_model_ckpt=""
stage2_gpus="0"
stage2_llm_backend="vllm"  # transformers | vllm
stage2_vllm_tensor_parallel_size=1
stage2_vllm_gpu_memory_utilization=0.85
stage2_vllm_dtype="auto"
stage2_max_new_tokens=2048

priority_image_paths_file="./priority_files.txt"
use_priority_file=1

prompt_style="aggressive"  # aggressive | conservative | neutral
stage1_run_id=""
stage2_run_id=""
stage3_run_id=""

start_stage="stage1"
end_stage="stage3"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<EOF_USAGE
Usage: $0 [OPTIONS]

Required for most runs:
  --dataset-root PATH                 Benchmark dataset root.
  --model-to-evaluate NAME            Logical name for the evaluated model.

Required for local Stage1:
  --model-to-evaluate-ckpt PATH       HF repo or local checkpoint path.

Required for Stage2:
  --judge-model NAME                  Logical judge/extractor LLM name.
  --judge-ckpt PATH                   HF repo or local checkpoint path for Stage2 judge.

General options:
  --output-root PATH                  Output root. Default: ${output_root}
  --prompt-style STYLE                aggressive/conservative/neutral. Default: ${prompt_style}
  --stage1-run-id RUN_ID
  --stage2-run-id RUN_ID
  --stage3-run-id RUN_ID
  --start-stage STAGE                 stage1/stage2/stage3 or 1/2/3. Default: ${start_stage}
  --end-stage STAGE                   stage1/stage2/stage3 or 1/2/3. Default: ${end_stage}

Stage1 backend:
  --stage1-backend BACKEND            local/api. Default: ${stage1_backend}
  --backend BACKEND                   Alias of --stage1-backend

Stage1 local runtime:
  --stage1-gpus IDS                   Comma-separated GPU ids. Default: ${stage1_gpus}
  --stage1-vllm-tp N                  Stage1 vLLM tensor parallel size. Default: ${stage1_vllm_tensor_parallel_size}
  --stage1-vllm-gpu-memory-utilization X
  --stage1-vllm-dtype DTYPE           Default: ${stage1_vllm_dtype}

Stage1 API runtime:
  --api-model NAME                    API model name. Default: same as --model-to-evaluate
  --api-base-url URL                  OpenAI-compatible base URL, e.g. https://api.openai.com/v1
  --base-url URL                      Alias of --api-base-url
  --api-key-env-name NAME             Env var storing API key. Default: ${api_key_env_name}
  --reasoning-effort VALUE            Optional reasoning effort. Default: ${reasoning_effort}
  --stage1-max-new-tokens N           Max new tokens for Stage1. Default: ${stage1_max_new_tokens}
  --max-new-tokens N                  Alias of --stage1-max-new-tokens

Stage2 runtime:
  --stage2-gpus IDS                   Comma-separated GPU ids. Default: ${stage2_gpus}
  --stage2-llm-backend BACKEND        transformers/vllm. Default: ${stage2_llm_backend}
  --stage2-vllm-tp N                  Stage2 vLLM tensor parallel size. Default: ${stage2_vllm_tensor_parallel_size}
  --stage2-vllm-gpu-memory-utilization X
  --stage2-vllm-dtype DTYPE           Default: ${stage2_vllm_dtype}
  --stage2-max-new-tokens N           Default: ${stage2_max_new_tokens}
  --priority-image-paths-file PATH    Optional priority file. Default: ${priority_image_paths_file}
  --no-priority-file                  Do not pass a priority file to Stage2.

Examples:
  # Local model, run all stages. If model name matches qwen3*, Stage1 uses vLLM.
  $0 \\
    --dataset-root /path/to/your_testset \\
    --model-to-evaluate your_model_name \\
    --model-to-evaluate-ckpt /path/to/your_model_checkpoint \\
    --judge-model your_judge_model_name \\
    --judge-ckpt /path/to/your_judge_checkpoint \\
    --stage1-gpus 0,1,2,3 \\
    --stage1-vllm-tp 4 \\
    --stage2-gpus 0,1,2,3 \\
    --stage2-vllm-tp 4

  # API model, run all stages.
  export OPENAI_API_KEY=your_api_key
  $0 \\
    --dataset-root /path/to/your_testset \\
    --stage1-backend api \\
    --model-to-evaluate your_api_model_display_name \\
    --api-model your_api_model_name \\
    --api-base-url https://your.openai-compatible.endpoint/v1 \\
    --api-key-env-name OPENAI_API_KEY \\
    --judge-model your_judge_model_name \\
    --judge-ckpt /path/to/your_judge_checkpoint

  # Only run Stage2/Stage3 from existing Stage1 outputs.
  $0 \\
    --dataset-root /path/to/your_testset \\
    --model-to-evaluate your_model_name \\
    --judge-model your_judge_model_name \\
    --judge-ckpt /path/to/your_judge_checkpoint \\
    --stage1-run-id your_existing_stage1_run_id \\
    --start-stage 2 \\
    --end-stage 3

  -h, --help                          Show this help message.
EOF_USAGE
}

require_value() {
  local arg_name="$1"
  local arg_value="${2:-}"
  if [[ -z "${arg_value}" || "${arg_value}" == --* ]]; then
    die "Missing value for ${arg_name}"
  fi
}

stage_to_idx() {
  local stage="$1"
  case "${stage}" in
    1|stage1) echo 1 ;;
    2|stage2) echo 2 ;;
    3|stage3) echo 3 ;;
    *) die "Invalid stage: ${stage}. Valid values: stage1, stage2, stage3, 1, 2, 3" ;;
  esac
}

model_matches_vllm_patterns() {
  local model_name="$1"
  local model_name_lc="${model_name,,}"
  local pattern pattern_lc
  for pattern in "${vllm_model_patterns[@]}"; do
    pattern_lc="${pattern,,}"
    case "${model_name_lc}" in
      ${pattern_lc}) return 0 ;;
    esac
  done
  return 1
}

# ---------------------------
# Parse CLI
# ---------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      require_value "$1" "${2:-}"
      dataset_root="$2"
      shift 2
      ;;
    --output-root)
      require_value "$1" "${2:-}"
      output_root="$2"
      shift 2
      ;;
    --model-to-evaluate)
      require_value "$1" "${2:-}"
      model_to_evaluate="$2"
      shift 2
      ;;
    --model-to-evaluate-ckpt)
      require_value "$1" "${2:-}"
      model_to_evaluate_ckpt="$2"
      shift 2
      ;;
    --stage1-backend|--backend)
      require_value "$1" "${2:-}"
      stage1_backend="$2"
      shift 2
      ;;
    --stage1-gpus)
      require_value "$1" "${2:-}"
      stage1_gpus="$2"
      shift 2
      ;;
    --stage1-vllm-tp)
      require_value "$1" "${2:-}"
      stage1_vllm_tensor_parallel_size="$2"
      shift 2
      ;;
    --stage1-vllm-gpu-memory-utilization)
      require_value "$1" "${2:-}"
      stage1_vllm_gpu_memory_utilization="$2"
      shift 2
      ;;
    --stage1-vllm-dtype)
      require_value "$1" "${2:-}"
      stage1_vllm_dtype="$2"
      shift 2
      ;;
    --api-model)
      require_value "$1" "${2:-}"
      api_model="$2"
      shift 2
      ;;
    --api-base-url|--base-url)
      require_value "$1" "${2:-}"
      api_base_url="$2"
      shift 2
      ;;
    --api-key-env-name)
      require_value "$1" "${2:-}"
      api_key_env_name="$2"
      shift 2
      ;;
    --reasoning-effort)
      require_value "$1" "${2:-}"
      reasoning_effort="$2"
      shift 2
      ;;
    --stage1-max-new-tokens|--max-new-tokens)
      require_value "$1" "${2:-}"
      stage1_max_new_tokens="$2"
      shift 2
      ;;
    --judge-model)
      require_value "$1" "${2:-}"
      judge_model="$2"
      shift 2
      ;;
    --judge-ckpt)
      require_value "$1" "${2:-}"
      judge_model_ckpt="$2"
      shift 2
      ;;
    --stage2-gpus)
      require_value "$1" "${2:-}"
      stage2_gpus="$2"
      shift 2
      ;;
    --stage2-llm-backend)
      require_value "$1" "${2:-}"
      stage2_llm_backend="$2"
      shift 2
      ;;
    --stage2-vllm-tp)
      require_value "$1" "${2:-}"
      stage2_vllm_tensor_parallel_size="$2"
      shift 2
      ;;
    --stage2-vllm-gpu-memory-utilization)
      require_value "$1" "${2:-}"
      stage2_vllm_gpu_memory_utilization="$2"
      shift 2
      ;;
    --stage2-vllm-dtype)
      require_value "$1" "${2:-}"
      stage2_vllm_dtype="$2"
      shift 2
      ;;
    --stage2-max-new-tokens)
      require_value "$1" "${2:-}"
      stage2_max_new_tokens="$2"
      shift 2
      ;;
    --priority-image-paths-file)
      require_value "$1" "${2:-}"
      priority_image_paths_file="$2"
      use_priority_file=1
      shift 2
      ;;
    --no-priority-file)
      use_priority_file=0
      shift 1
      ;;
    --prompt-style)
      require_value "$1" "${2:-}"
      prompt_style="$2"
      shift 2
      ;;
    --stage1-run-id)
      require_value "$1" "${2:-}"
      stage1_run_id="$2"
      shift 2
      ;;
    --stage2-run-id)
      require_value "$1" "${2:-}"
      stage2_run_id="$2"
      shift 2
      ;;
    --stage3-run-id)
      require_value "$1" "${2:-}"
      stage3_run_id="$2"
      shift 2
      ;;
    --start-stage)
      require_value "$1" "${2:-}"
      start_stage="$2"
      shift 2
      ;;
    --end-stage)
      require_value "$1" "${2:-}"
      end_stage="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "Unknown argument: $1"
      ;;
  esac
done

case "${stage1_backend}" in
  local|api) ;;
  *) die "Invalid --stage1-backend: ${stage1_backend}. Valid values: local, api" ;;
esac

case "${stage2_llm_backend}" in
  transformers|vllm) ;;
  *) die "Invalid --stage2-llm-backend: ${stage2_llm_backend}. Valid values: transformers, vllm" ;;
esac

case "${prompt_style}" in
  aggressive)
    prompt="Describe the image in as much detail as possible. Respond in English only."
    ;;
  conservative)
    prompt="Describe the image factually and refrain from guessing unobserved details. Respond in English only."
    ;;
  neutral)
    prompt="Describe the image. Respond in English only."
    ;;
  *) die "Invalid --prompt-style: ${prompt_style}. Valid values: aggressive, conservative, neutral" ;;
esac

start_stage_idx="$(stage_to_idx "${start_stage}")"
end_stage_idx="$(stage_to_idx "${end_stage}")"

if (( start_stage_idx > end_stage_idx )); then
  die "Invalid stage range: --start-stage ${start_stage} is after --end-stage ${end_stage}"
fi

# ---------------------------
# Validate required settings
# ---------------------------
[[ -n "${dataset_root}" ]] || die "--dataset-root is required. Use /path/to/your_testset."
[[ -d "${dataset_root}" ]] || die "--dataset-root does not exist or is not a directory: ${dataset_root}"
[[ -n "${model_to_evaluate}" && "${model_to_evaluate}" != "your_model_name" ]] || die "--model-to-evaluate is required."

if (( start_stage_idx <= 1 && end_stage_idx >= 1 )); then
  if [[ "${stage1_backend}" == "local" ]]; then
    [[ -n "${model_to_evaluate_ckpt}" ]] || die "--model-to-evaluate-ckpt is required when --stage1-backend local."
  else
    [[ -n "${api_base_url}" ]] || die "--api-base-url is required when --stage1-backend api."
    [[ -n "${api_key_env_name}" ]] || die "--api-key-env-name is required when --stage1-backend api."
    if [[ -z "${api_model}" ]]; then
      api_model="${model_to_evaluate}"
    fi
  fi
fi

if (( start_stage_idx <= 2 && end_stage_idx >= 2 )); then
  [[ -n "${judge_model}" && "${judge_model}" != "your_judge_model_name" ]] || die "--judge-model is required for Stage2."
  [[ -n "${judge_model_ckpt}" ]] || die "--judge-ckpt is required for Stage2."
fi

if [[ -z "${stage1_run_id}" ]]; then
  stage1_run_id="stage1-${prompt_style}-max${stage1_max_new_tokens}"
fi
if [[ -z "${stage2_run_id}" ]]; then
  stage2_run_id="stage2-${judge_model}-extract-match"
fi
if [[ -z "${stage3_run_id}" ]]; then
  stage3_run_id="stage3-${judge_model}-eval"
fi

stage1_local_engine="transformers"
if [[ "${stage1_backend}" == "local" ]] && model_matches_vllm_patterns "${model_to_evaluate}"; then
  stage1_local_engine="vllm"
fi

stage1_dir="${output_root}/${model_to_evaluate}/stage1_answers-${stage1_run_id}"
stage2_dir="${stage1_dir}/stage2-ext_match/${stage2_run_id}"

stage2_priority_args=()
if (( use_priority_file == 1 )); then
  if [[ -f "${priority_image_paths_file}" ]]; then
    stage2_priority_args=(--priority-image-paths-file "${priority_image_paths_file}")
  else
    echo "[warn] priority file not found; Stage2 will run without priority file: ${priority_image_paths_file}" >&2
  fi
fi

# ---------------------------
# Print resolved config
# ---------------------------
echo "Resolved config:"
echo "  dataset_root:                  ${dataset_root}"
echo "  output_root:                   ${output_root}"
echo "  model_to_evaluate:             ${model_to_evaluate}"
echo "  stage1_backend:                ${stage1_backend}"
echo "  stage1_local_engine:           ${stage1_local_engine}"
echo "  model_to_evaluate_ckpt:        ${model_to_evaluate_ckpt}"
echo "  stage1_gpus:                   ${stage1_gpus}"
echo "  stage1_vllm_tp:                ${stage1_vllm_tensor_parallel_size}"
echo "  api_model:                     ${api_model}"
echo "  api_base_url:                  ${api_base_url}"
echo "  api_key_env_name:              ${api_key_env_name}"
echo "  reasoning_effort:              ${reasoning_effort}"
echo "  stage1_max_new_tokens:         ${stage1_max_new_tokens}"
echo "  judge_model:                   ${judge_model}"
echo "  judge_model_ckpt:              ${judge_model_ckpt}"
echo "  stage2_gpus:                   ${stage2_gpus}"
echo "  stage2_llm_backend:            ${stage2_llm_backend}"
echo "  stage2_vllm_tp:                ${stage2_vllm_tensor_parallel_size}"
echo "  stage2_max_new_tokens:         ${stage2_max_new_tokens}"
echo "  prompt_style:                  ${prompt_style}"
echo "  stage1_run_id:                 ${stage1_run_id}"
echo "  stage2_run_id:                 ${stage2_run_id}"
echo "  stage3_run_id:                 ${stage3_run_id}"
echo "  start_stage:                   ${start_stage} -> ${start_stage_idx}"
echo "  end_stage:                     ${end_stage} -> ${end_stage_idx}"
echo "  stage1_dir:                    ${stage1_dir}"
echo "  stage2_dir:                    ${stage2_dir}"
echo

# ---------------------------
# Stage1
# ---------------------------
if (( start_stage_idx <= 1 && end_stage_idx >= 1 )); then
  echo "[Stage1] Running answer generation..."
  echo "[Stage1] backend=${stage1_backend}, local_engine=${stage1_local_engine}"

  if [[ "${stage1_backend}" == "api" ]]; then
    python generate_answers.py \
      --dataset-root "${dataset_root}" \
      --model-name "${model_to_evaluate}" \
      --backend api \
      --api-model "${api_model}" \
      --base-url "${api_base_url}" \
      --api-key-env-name "${api_key_env_name}" \
      --max-new-tokens "${stage1_max_new_tokens}" \
      --run-id "${stage1_run_id}" \
      --reasoning-effort "${reasoning_effort}" \
      --output-root "${output_root}" \
      --prompt "${prompt}"

  elif [[ "${stage1_local_engine}" == "vllm" ]]; then
    VLLM_WORKER_MULTIPROC_METHOD=spawn python generate_answers.py \
      --backend local \
      --local-engine vllm \
      --model-name "${model_to_evaluate}" \
      --model-ckpt "${model_to_evaluate_ckpt}" \
      --dataset-root "${dataset_root}" \
      --output-root "${output_root}" \
      --run-id "${stage1_run_id}" \
      --gpus "${stage1_gpus}" \
      --vllm-tensor-parallel-size "${stage1_vllm_tensor_parallel_size}" \
      --vllm-gpu-memory-utilization "${stage1_vllm_gpu_memory_utilization}" \
      --vllm-dtype "${stage1_vllm_dtype}" \
      --max-new-tokens "${stage1_max_new_tokens}" \
      --prompt "${prompt}"

  else
    CUDA_VISIBLE_DEVICES="${stage1_gpus}" python generate_answers.py \
      --backend local \
      --dataset-root "${dataset_root}" \
      --model-name "${model_to_evaluate}" \
      --model-ckpt "${model_to_evaluate_ckpt}" \
      --run-id "${stage1_run_id}" \
      --output-root "${output_root}" \
      --max-new-tokens "${stage1_max_new_tokens}" \
      --prompt "${prompt}"
  fi
else
  echo "[Stage1] Skipped."
fi

# ---------------------------
# Stage2
# ---------------------------
if (( start_stage_idx <= 2 && end_stage_idx >= 2 )); then
  echo "[Stage2] Running extraction and matching..."

  CUDA_VISIBLE_DEVICES="${stage2_gpus}" python extract_and_match.py \
    --run-root "${stage1_dir}" \
    --judge-model "${judge_model}" \
    --judge-ckpt "${judge_model_ckpt}" \
    --run-id "${stage2_run_id}" \
    --llm-backend "${stage2_llm_backend}" \
    --vllm-tensor-parallel-size "${stage2_vllm_tensor_parallel_size}" \
    --vllm-gpu-memory-utilization "${stage2_vllm_gpu_memory_utilization}" \
    --vllm-dtype "${stage2_vllm_dtype}" \
    --max-new-tokens "${stage2_max_new_tokens}" \
    "${stage2_priority_args[@]}"
else
  echo "[Stage2] Skipped."
fi

# ---------------------------
# Stage3
# ---------------------------
if (( start_stage_idx <= 3 && end_stage_idx >= 3 )); then
  echo "[Stage3] Running evaluation..."

  python evaluate.py \
    --run-root "${stage2_dir}" \
    --run-id "${stage3_run_id}" \
    --overwrite
else
  echo "[Stage3] Skipped."
fi
