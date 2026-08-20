#!/bin/bash
# Re-export $MODEL_PATH's tokenizer into $BENCH_TOKENIZER in a form that a plain
# transformers AutoTokenizer can load.
#
# Why: the checkpoint's tokenizer_config.json says `tokenizer_class:
# TokenizersBackend`, which is sglang's own backend name, not a transformers
# class. The server is fine -- srt/hf_transformers_utils.get_tokenizer strips
# that field into a temp dir -- but bench_serving.get_tokenizer only routes to
# that loader for gemma-4 paths (bench_serving.py:739-753) and otherwise calls
# AutoTokenizer directly, which dies with:
#   ValueError: Tokenizer class TokenizersBackend does not exist
#
# Doing the re-export here keeps the baseline tree unpatched. The vocab and
# merges are untouched, so token ids are identical -- the check below asserts it.
set -euo pipefail

cd "$(dirname "$0")"
source ./env.sh

"$PY" - "$MODEL_PATH" "$BENCH_TOKENIZER" <<'PYEOF'
import pathlib, sys

from transformers import AutoTokenizer

from sgl_jax.srt.hf_transformers_utils import get_tokenizer

src, dst = sys.argv[1], sys.argv[2]

ref = get_tokenizer(src, trust_remote_code=True)
pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
ref.save_pretrained(dst)

# Must load without sglang's workaround, and must tokenize identically.
new = AutoTokenizer.from_pretrained(dst, trust_remote_code=True)
probes = [
    "The quick brown fox jumps over the lazy dog.",
    "你好，世界！GLM-5.2 FP8 on TPU v7x.",
    "def f(x):\n    return x ** 2  # comment\n",
]
for p in probes:
    assert new.encode(p) == ref.encode(p), f"token ids differ for {p!r}"
assert new.vocab_size == ref.vocab_size

print(f"{dst}: {type(new).__name__}, vocab={new.vocab_size}, ids match source")
PYEOF
