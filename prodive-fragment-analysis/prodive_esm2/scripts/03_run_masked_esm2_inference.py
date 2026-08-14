#!/usr/bin/env python3
"""Predict masked-token distributions for frag_dom-formatted records.

Input format (3 lines per record, blank lines ignored; frag_dom line is ignored):
>HEADER
frag_dom=10-20,11-21,...
SEQUENCE

Example:
  python3 03_run_masked_esm2_inference.py \
    --model-dir pretrained_models/esm2_t36_3B_UR50D \
    --input representatives.full.with_frag_ranges.txt \
    --record-range 1:50 \
    --output representatives-1-50_3B.dat \
    --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from fragdom_utils import iter_fragdom_sequences

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
EXPECTED_MODEL_ID = "facebook/esm2_t36_3B_UR50D"
EXPECTED_NUM_HIDDEN_LAYERS = 36
EXPECTED_HIDDEN_SIZE = 2560


def _format_probs(probs: torch.Tensor) -> str:
    return " ".join(f"{aa}:{p:.5f}" for aa, p in zip(AA_ORDER, probs.tolist()))


def main() -> None:
    p = argparse.ArgumentParser(description="Masked-token inference for frag_dom ranges")
    p.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help=(
            "Local Transformers checkpoint for facebook/esm2_t36_3B_UR50D "
            "(must include tokenizer and masked-LM head weights)"
        ),
    )
    p.add_argument("--input", type=str, required=True, help="Path to frag_dom records (txt or .gz)")
    p.add_argument("--output", type=str, default=None, help="Probability text output path (default: stdout)")
    p.add_argument(
        "--record-range",
        type=str,
        default=None,
        help="1-indexed inclusive record range start:end (e.g., 1:1000)",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        from transformers.utils import logging as hf_logging
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "The transformers package is required for ESM2 masked-token inference. "
            "Install it before running this script."
        ) from e

    hf_logging.set_verbosity_error()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model-dir does not exist: {model_dir}")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input file does not exist: {input_path}")

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForMaskedLM.from_pretrained(str(model_dir))
    observed_layers = getattr(model.config, "num_hidden_layers", None)
    observed_hidden_size = getattr(model.config, "hidden_size", None)
    if (
        observed_layers != EXPECTED_NUM_HIDDEN_LAYERS
        or observed_hidden_size != EXPECTED_HIDDEN_SIZE
    ):
        raise ValueError(
            "The released ESM2 analysis requires "
            f"{EXPECTED_MODEL_ID} (num_hidden_layers={EXPECTED_NUM_HIDDEN_LAYERS}, "
            f"hidden_size={EXPECTED_HIDDEN_SIZE}); loaded checkpoint reports "
            f"num_hidden_layers={observed_layers}, hidden_size={observed_hidden_size}."
        )
    model.eval()

    device = torch.device(args.device)
    model.to(device)

    aa_ids = [tok.convert_tokens_to_ids(aa) for aa in AA_ORDER]
    if any(idx is None for idx in aa_ids):
        raise ValueError("Tokenizer is missing AA tokens required for probability extraction")
    aa_ids_tensor = torch.tensor(aa_ids, device=device, dtype=torch.long)

    output_f = open(args.output, "w") if args.output else None

    def _parse_record_range(raw: Optional[str]) -> Tuple[int, Optional[int]]:
        if not raw:
            return 1, None
        if ":" not in raw:
            raise ValueError("record-range must be start:end (1-indexed)")
        start_s, end_s = raw.split(":", 1)
        start = int(start_s) if start_s else 1
        end = int(end_s) if end_s else None
        if start <= 0:
            raise ValueError("record-range start must be >= 1")
        if end is not None and end < start:
            raise ValueError("record-range end must be >= start")
        return start, end

    start_idx, end_idx = _parse_record_range(args.record_range)
    processed = 0

    try:
        for idx, (header, seq) in enumerate(iter_fragdom_sequences(input_path), start=1):
            if idx < start_idx:
                continue
            if end_idx is not None and idx > end_idx:
                break
            seq_len = len(seq)

            inputs = tok(seq, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            expected_len = seq_len + 2
            if input_ids.size(1) != expected_len:
                raise ValueError(
                    f"Tokenizer length mismatch for {header}: got {input_ids.size(1)}, expected {expected_len}"
                )

            with torch.no_grad():
                line_header = f">{header}"
                if output_f is not None:
                    output_f.write(line_header + "\n")
                else:
                    print(line_header)
                for pos in range(1, seq_len + 1):
                    masked_ids = input_ids.clone()
                    masked_ids[0, pos] = int(tok.mask_token_id)
                    outputs = model(input_ids=masked_ids, attention_mask=attention_mask)
                    scores = outputs.logits[0, pos]
                    probs = torch.softmax(scores.float(), dim=-1)
                    aa_probs = probs.index_select(0, aa_ids_tensor).detach().cpu().double()
                    line = _format_probs(aa_probs)
                    if output_f is not None:
                        output_f.write(line + "\n")
                    else:
                        print(line)
            processed += 1
    finally:
        if output_f is not None:
            output_f.close()
    if processed == 0:
        raise ValueError("No records processed; check --record-range and input file")


if __name__ == "__main__":
    main()
