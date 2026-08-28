#!/usr/bin/env python3
"""Stage 1 — the HF-port oracle.

Runs models/core's OWN shipped code (no NeMo involved) to produce the offline transcript that
every streaming run and the converted .nemo checkpoint are scored against. Two implementations
agreeing is the only verification of the conversion that actually means anything.

Two silent traps this guards against
------------------------------------
1. ``_init_weights`` must stay a no-op. transformers 5.x calls it on every module AFTER
   populating them from the checkpoint and sets no ``_is_hf_initialized`` marker, so an
   initializing body randomizes every weight *while reporting zero missing/unexpected keys*.
   Key counts are therefore NOT a valid load check. We verify by comparing loaded parameter
   VALUES against the raw safetensors payload, and by decoding real audio.

2. Never auto-LID. Measured top-1 accuracy: bho 0.047, hi 0.258, mai 0.356, ur 0.490. A wrong
   language yields the wrong *script* with no error raised, so ``lang`` is always explicit and
   this tool refuses to guess.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import textwrap
from pathlib import Path

import torch


def _safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _read_tensor(path: Path, name: str) -> torch.Tensor:
    """Read one tensor straight out of the file, bypassing every loader."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def verify_weights_by_value(model, model_dir: Path, n_probe: int = 6) -> dict:
    """Assert the loaded model actually carries the checkpoint's numbers.

    This is the guard against the silent `_init_weights` re-randomization. We do NOT count keys
    -- the failure mode reports a perfectly clean load. We compare actual values.
    """
    st = model_dir / "model.safetensors"
    hdr = _safetensors_header(st)
    names = [k for k in hdr if k != "__metadata__"]

    # A deterministic spread: the embedding (tied to lm_head, so the single most load-bearing
    # tensor), plus evenly spaced encoder/decoder tensors.
    probes = ["model.decoder.embedding.token_embedding.weight"]
    step = max(1, len(names) // n_probe)
    probes += [n for n in sorted(names)[::step][:n_probe] if n not in probes]

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    checked, mismatched = [], []
    for name in probes:
        ref = _read_tensor(st, name)
        got = params.get(name, buffers.get(name))
        if got is None:
            continue
        got = got.detach().cpu()
        # Compare in the LOADED dtype, and demand exact equality. Loading a fp32 checkpoint at
        # bf16 is a pure cast, so any difference at all is a real defect. Comparing in fp32
        # with a tolerance instead is wrong in both directions: bf16 carries only ~2^-8 (0.4%)
        # relative precision, so a tight rtol flags healthy large-magnitude tensors, while an
        # rtol loose enough to accommodate bf16 would wave through genuine corruption.
        same = torch.equal(got, ref.to(got.dtype))
        if same:
            checked.append(name)
        else:
            diff = (got.float() - ref.float()).abs()
            mismatched.append(
                f"{name}: max|d|={diff.max():.3e} mean|d|={diff.mean():.3e} "
                f"ref_rms={ref.float().pow(2).mean().sqrt():.3e}"
            )

    if mismatched:
        raise SystemExit(
            "WEIGHT VERIFICATION FAILED — loaded values differ from the checkpoint:\n  "
            + "\n  ".join(mismatched)
            + "\n\nIf max|d| is comparable to ref_rms the tensor is random: that is the "
              "_init_weights re-randomization signature, so confirm "
              "modeling_indic_canary.py:_init_weights is still a no-op."
        )

    # The embedding is tied to lm_head; if tying broke, generation silently degrades.
    emb = params["model.decoder.embedding.token_embedding.weight"]
    head = params.get("lm_head.weight")
    tied = head is None or head.data_ptr() == emb.data_ptr()

    return {"probes_checked": checked, "n_probes": len(checked), "lm_head_tied": bool(tied)}


def load(model_dir: Path, device: str, dtype: torch.dtype):
    sys.path.insert(0, str(model_dir))
    from indic_transcribe import LANGUAGES, IndicTranscribe  # noqa: F401
    from modeling_indic_canary import IndicCanaryForConditionalGeneration

    # Assert the no-op BEFORE loading 4.9 GB, so the failure is cheap and legible. The body is
    # read from source rather than trusted, because the corruption it causes is invisible to
    # every other check -- the load reports no missing/unexpected/mismatched keys while doing it.
    import ast
    import inspect
    import re

    src = inspect.getsource(IndicCanaryForConditionalGeneration._init_weights)
    fn = ast.parse(textwrap.dedent(src)).body[0]
    stmts = [ast.unparse(n) for n in fn.body
             if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                     and isinstance(n.value.value, str))]  # drop the docstring
    dangerous = [s for s in stmts
                 if re.search(r"normal_|uniform_|\binit\.|fill_|zero_|copy_", s)]
    if dangerous:
        raise SystemExit(
            "_init_weights has an initializing body — it will silently randomize the "
            f"checkpoint AFTER loading, reporting a clean load while doing so:\n  {dangerous}"
        )
    print(f"[verify] _init_weights is inert: {stmts or ['<empty>']}", file=sys.stderr)

    asr = IndicTranscribe.from_pretrained(str(model_dir), device=device, dtype=dtype)
    return asr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="*", help="audio file(s) to transcribe")
    ap.add_argument("--model-dir", default="/models/core", type=Path)
    ap.add_argument("--lang", required=True,
                    help="REQUIRED and never inferred — a wrong language yields wrong script "
                         "silently. See the module docstring.")
    ap.add_argument("--mode", default="native", choices=["native", "mixed", "romanised"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--out", type=Path, help="write results as JSON here")
    ap.add_argument("--verify-only", action="store_true",
                    help="load and verify weights by value, transcribe nothing")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    asr = load(args.model_dir, args.device, dtype)

    report = verify_weights_by_value(asr.model, args.model_dir)
    n_params = sum(p.numel() for p in asr.model.parameters())
    report["n_params"] = n_params
    report["n_params_b"] = round(n_params / 1e9, 4)
    print(f"[verify] {report['n_probes']} tensors match the checkpoint by value; "
          f"lm_head tied={report['lm_head_tied']}; params={report['n_params_b']}B",
          file=sys.stderr)

    if args.verify_only:
        print(json.dumps(report, indent=2))
        return 0

    if not args.audio:
        ap.error("give at least one audio file, or pass --verify-only")

    # The emitted TOKEN IDS are the real oracle. Text comparison is decode-convention
    # sensitive: this port's decode() does piece-join -> '_'->' ' -> .strip(), while NeMo's
    # Hypothesis.text keeps the leading space that the first SentencePiece '_' produces. Two
    # implementations can therefore agree perfectly and still differ by one leading U+0020, so
    # the ids are what Stage 2's gate compares.
    from indic_transcribe import MODES

    itn, romanized = MODES[args.mode]
    results = []
    for path in args.audio:
        wav = asr._load(path)
        feats, mask = asr._features(wav)
        prompt = asr.tokenizer.encode_prompt(args.lang, itn=itn, romanized=romanized)
        with torch.inference_mode():
            out = asr.model.generate(
                input_features=feats, attention_mask=mask,
                decoder_input_ids=torch.tensor([prompt], device=asr.device),
                max_new_tokens=256,
            )
        ids = asr.tokenizer.strip_prompt_and_trim(out[0].tolist(), prompt)
        text = asr.tokenizer.decode(ids)
        results.append({"audio": str(path), "lang": args.lang, "mode": args.mode,
                        "text": text, "token_ids": ids, "prompt_ids": prompt})
        print(f"{path}\t{text}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"verify": report, "results": results},
                                       ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
