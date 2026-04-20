"""Global configuration for the reasoning-redundancy project."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
TRACES_DIR = OUT_DIR / "traces"
TRUNC_DIR = OUT_DIR / "truncation"
ABL_DIR = OUT_DIR / "ablation"
FIG_DIR = OUT_DIR / "figures"

for d in [DATA_DIR, OUT_DIR, TRACES_DIR, TRUNC_DIR, ABL_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------- API endpoints ---------------- #
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------- Models ---------------- #
# (display_name, provider, model_id)
REASONING_MODELS = [
    ("deepseek_r1", "deepseek", "deepseek-reasoner"),
    ("qwq_32b",     "dashscope", "qwq-32b"),
    ("r1_distill_7b", "dashscope", "deepseek-r1-distill-qwen-7b"),
    ("qwen3_30b_thinking", "dashscope", "qwen3-30b-a3b-thinking-2507"),
]

JUDGE_SELF = ("deepseek", "deepseek-chat")        # cheap non-reasoning judge
JUDGE_CROSS = ("openai", "gpt-4o-mini")

# ---------------- Experiment knobs ---------------- #
# Scale knob: "pilot" (tiny smoke test) | "A" (150 MATH x 3) | "full" (500x10).
SCALE = os.getenv("SCALE", "A")
PILOT_MODE = SCALE == "pilot"

if SCALE == "pilot":
    N_MATH = 20
    N_GSM8K = 20
    M_SAMPLES = 2
    ABLATION_PROBLEMS_PER_LEVEL = 4
    REASONING_MODELS = [REASONING_MODELS[0]]
elif SCALE == "A":
    N_MATH = 150
    N_GSM8K = 60
    M_SAMPLES = 3
    ABLATION_PROBLEMS_PER_LEVEL = 10
    # All configured reasoning models (4 total: DS-R1, QwQ, R1-Distill-7B, Qwen3-30B-thinking)
else:  # "full"
    N_MATH = 500
    N_GSM8K = 500
    M_SAMPLES = 10
    ABLATION_PROBLEMS_PER_LEVEL = 20

TEMPERATURE = 0.7
MAX_TOKENS_REASONING = 8192
MAX_TOKENS_JUDGE = 256
SEED = 42

# Concurrency
MAX_CONCURRENT = 8
RETRY_ATTEMPTS = 3

# ---------------- File paths helper ---------------- #
def trace_path(model_name: str, dataset: str) -> Path:
    return TRACES_DIR / f"{model_name}__{dataset}.jsonl"

def segments_path(model_name: str, dataset: str) -> Path:
    return TRACES_DIR / f"{model_name}__{dataset}.segments.jsonl"

def truncation_path(model_name: str, dataset: str, judge: str) -> Path:
    # Prefer the v2 self-judge data (correct: primary judge = model itself,
    # cross judge = gpt-4o-mini uniformly). Fall back to legacy for back-compat.
    v2 = TRUNC_DIR / f"v2__{model_name}__{dataset}__judge-{judge}.jsonl"
    if v2.exists():
        return v2
    return TRUNC_DIR / f"{model_name}__{dataset}__judge-{judge}.jsonl"

def ablation_path(model_name: str, dataset: str) -> Path:
    return ABL_DIR / f"{model_name}__{dataset}.jsonl"
