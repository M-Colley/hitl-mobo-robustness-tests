"""Convert NISQA_TRAIN_SIM + NISQA_VAL_SIM per-file metadata into the
pipeline observation format.

Design space = the published distortion parameters used to generate each
clip (transmission-chain tuning). Encoding decisions:

- Filter cutoffs: bp_low (highpass cutoff, fill 0 Hz = no highpass),
  bp_high (lowpass cutoff, fill 16000 Hz = no lowpass). The categorical
  'filter' column is fully captured by which cutoffs are set.
- arb_filter: random arbitrary frequency response; only the on/off flag is
  published, so it stays a binary param (realized shape -> residual noise).
- Time clipping: tc_fer (frame erasure %, fill 0 = off), tc_nburst (fill 0).
- Noise: wbgn_snr / bgn_snr in dB SNR, fill 60 dB = no audible noise
  (observed max 50). p50_q MNRU Q in dB, fill 45 (observed max 30 = no MNRU).
- Amplitude clipping: cl_th threshold, fill 1.0 = no clipping (observed
  0.01-0.5; lower = more clipping).
- Active speech level: asl_in_level (fill 0 = no level change) + asl_in_on
  flag; asl_out_level (fill -26 dBov nominal) + asl_out_on flag. Flags are
  kept because the neutral fill for the levels is an assumption.
- Codec chain (up to 3 tandemed codecs): per slot an ordinal codec id
  (0 none, 1 g711, 2 g722, 3 amrnb, 4 amrwb, 5 evs, 6 opus ~ ordered by
  bandwidth/era), bitrate mode bMode (fill 0 = no codec), packet-loss rate
  FER (fill 0 = no loss), and plc pattern flags plc_random / plc_bursty
  (noloss or no codec -> 0/0).

Objectives: mos, noi, col, dis, loud (all 1-5, higher = better quality /
less of the named degradation; per-file means over ~5 crowd votes).

Group_ID = source-speaker of the underlying clean clip (numeric code).
NOTE: this is a speech-CONTENT grouping, not a rater grouping - the corpus
publishes no rater ids for the SIM subsets. Grouped CV therefore measures
generalization to unseen source speakers, not unseen raters.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent / "raw" / "nisqa"
REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "external_datasets" / "nisqa_sim"

CODEC_ID = {"-": 0, "g711": 1, "g722": 2, "amrnb": 3, "amrwb": 4, "evs": 5, "opus": 6}

PARAM_COLUMNS = [
    "bp_low", "bp_high", "arb_filter_on",
    "tc_fer", "tc_nburst",
    "wbgn_snr", "bgn_snr", "p50_q", "cl_th",
    "asl_in_on", "asl_in_level", "asl_out_on", "asl_out_level",
    "codec1_id", "bMode1", "FER1", "plc1_random", "plc1_bursty",
    "codec2_id", "bMode2", "FER2", "plc2_random", "plc2_bursty",
    "codec3_id", "bMode3", "FER3", "plc3_random", "plc3_bursty",
]
OBJECTIVE_COLUMNS = ["mos", "noi", "col", "dis", "loud"]


def speaker_key(row: pd.Series) -> str:
    name = row["filename_ref"]
    src = row["source"]
    toks = name.replace(".wav", "").split("_")
    if src == "AusTalk":
        return f"austalk_{toks[0]}_{toks[1]}"
    if src == "DNS":
        i = toks.index("reader")
        return f"dns_reader_{toks[i + 1]}"
    if src == "TSP":
        return f"tsp_{toks[2]}"
    if src == "UKIRE":
        return f"ukire_{toks[0]}_{toks[1]}"
    raise ValueError(src)


def convert() -> pd.DataFrame:
    frames = []
    for sub in ["NISQA_TRAIN_SIM", "NISQA_VAL_SIM"]:
        frames.append(pd.read_csv(RAW / sub / f"{sub}_file.csv"))
    df = pd.concat(frames, ignore_index=True)

    out = pd.DataFrame()
    out["bp_low"] = pd.to_numeric(df["bp_low"], errors="coerce").fillna(0.0)
    out["bp_high"] = pd.to_numeric(df["bp_high"], errors="coerce").fillna(16000.0)
    out["arb_filter_on"] = (df["arb_filter"] == "x").astype(int)
    out["tc_fer"] = pd.to_numeric(df["tc_fer"], errors="coerce").fillna(0.0)
    out["tc_nburst"] = pd.to_numeric(df["tc_nburst"], errors="coerce").fillna(0.0)
    out["wbgn_snr"] = pd.to_numeric(df["wbgn_snr"], errors="coerce").fillna(60.0)
    out["bgn_snr"] = pd.to_numeric(df["bgn_snr"], errors="coerce").fillna(60.0)
    out["p50_q"] = pd.to_numeric(df["p50_q"], errors="coerce").fillna(45.0)
    out["cl_th"] = pd.to_numeric(df["cl_th"], errors="coerce").fillna(1.0)
    out["asl_in_on"] = (df["asl_in"] == "x").astype(int)
    out["asl_in_level"] = pd.to_numeric(df["asl_in_level"], errors="coerce").fillna(0.0)
    out["asl_out_on"] = (df["asl_out"] == "x").astype(int)
    out["asl_out_level"] = pd.to_numeric(df["asl_out_level"], errors="coerce").fillna(-26.0)

    for slot in (1, 2, 3):
        codec = df[f"codec{slot}"].fillna("-")
        plc = df[f"plcMode{slot}"].fillna("-")
        out[f"codec{slot}_id"] = codec.map(CODEC_ID).astype(int)
        out[f"bMode{slot}"] = pd.to_numeric(df[f"bMode{slot}"], errors="coerce").fillna(0.0)
        out[f"FER{slot}"] = pd.to_numeric(df[f"FER{slot}"], errors="coerce").fillna(0.0)
        out[f"plc{slot}_random"] = (plc == "random").astype(int)
        out[f"plc{slot}_bursty"] = (plc == "bursty").astype(int)

    for col in OBJECTIVE_COLUMNS:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    # metadata kept for noise calibration / provenance (not params/objectives)
    out["votes"] = df["votes"]
    for col in ["mos_std", "noi_std", "col_std", "dis_std", "loud_std"]:
        out[col] = df[col]
    out["db"] = df["db"]
    out["source"] = df["source"]

    speakers = df.apply(speaker_key, axis=1)
    codes = {s: i for i, s in enumerate(sorted(speakers.unique()))}
    out["Group_ID"] = speakers.map(codes).astype(int)
    out["speaker"] = speakers

    assert not out[PARAM_COLUMNS].isna().any().any()
    assert not out[OBJECTIVE_COLUMNS].isna().any().any()
    return out


def main() -> None:
    out = convert()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "ObservationsPerEvaluation.csv"
    out.to_csv(dest, sep=";", index=False)
    print(f"wrote {dest}: {len(out)} rows, {out['Group_ID'].nunique()} speakers")
    print("params:", PARAM_COLUMNS)
    print("objectives:", OBJECTIVE_COLUMNS)
    print(out[PARAM_COLUMNS].describe().T[["min", "max", "mean"]])


if __name__ == "__main__":
    main()
