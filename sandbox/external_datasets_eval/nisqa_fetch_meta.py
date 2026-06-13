"""Extract all non-audio members (csv/txt/xlsx/pdf) of the remote
NISQA_Corpus.zip on Zenodo via HTTP range requests. Total transfer ~25 MB
out of a 16 GB archive; audio is never downloaded.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import requests

URL = "https://zenodo.org/records/4728081/files/NISQA_Corpus.zip?download=1"
TOTAL_SIZE = 15_945_499_507
OUT_DIR = Path(__file__).resolve().parent / "raw" / "nisqa"
KEEP_EXT = (".csv", ".txt", ".xlsx", ".pdf")

session = requests.Session()
session.headers.update({"User-Agent": "metadata-extractor/0.1 (research use)"})


def fetch(start: int, end: int) -> bytes:
    r = session.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
    r.raise_for_status()
    assert r.status_code == 206
    return r.content


def central_directory() -> list[dict]:
    tail = fetch(TOTAL_SIZE - (1 << 16), TOTAL_SIZE - 1)
    loc_pos = tail.rfind(b"PK\x06\x07")
    (_, _, z64_eocd_offset, _) = struct.unpack("<IIQI", tail[loc_pos : loc_pos + 20])
    z64_pos = z64_eocd_offset - (TOTAL_SIZE - (1 << 16))
    z64 = tail[z64_pos : z64_pos + 56]
    (sig, _, _, _, _, _, _, n_total, cd_size, cd_offset) = struct.unpack("<IQHHIIQQQQ", z64[:56])
    assert sig == 0x06064B50
    cd = fetch(cd_offset, cd_offset + cd_size - 1)
    entries = []
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            break
        (
            sig, _, _, _, method, _, _, _, comp_size, uncomp_size,
            name_len, extra_len, comment_len, _, _, _, local_offset,
        ) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos : pos + 46])
        name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8", "replace")
        extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        if 0xFFFFFFFF in (uncomp_size, comp_size, local_offset):
            epos = 0
            while epos + 4 <= len(extra):
                eid, esz = struct.unpack("<HH", extra[epos : epos + 4])
                edata = extra[epos + 4 : epos + 4 + esz]
                if eid == 0x0001:
                    fpos = 0
                    if uncomp_size == 0xFFFFFFFF:
                        uncomp_size = struct.unpack("<Q", edata[fpos : fpos + 8])[0]; fpos += 8
                    if comp_size == 0xFFFFFFFF:
                        comp_size = struct.unpack("<Q", edata[fpos : fpos + 8])[0]; fpos += 8
                    if local_offset == 0xFFFFFFFF:
                        local_offset = struct.unpack("<Q", edata[fpos : fpos + 8])[0]; fpos += 8
                epos += 4 + esz
        entries.append(dict(name=name, method=method, comp_size=comp_size,
                            uncomp_size=uncomp_size, local_offset=local_offset))
        pos += 46 + name_len + extra_len + comment_len
    return entries


def extract(entry: dict) -> bytes:
    # local header: 30 fixed bytes + name + extra, then data
    head = fetch(entry["local_offset"], entry["local_offset"] + 29)
    assert head[:4] == b"PK\x03\x04", head[:4]
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    data_start = entry["local_offset"] + 30 + name_len + extra_len
    raw = fetch(data_start, data_start + entry["comp_size"] - 1)
    if entry["method"] == 0:
        return raw
    if entry["method"] == 8:
        return zlib.decompress(raw, -15)
    raise ValueError(f"unsupported method {entry['method']}")


def main() -> None:
    entries = central_directory()
    keep = [e for e in entries if e["name"].lower().endswith(KEEP_EXT)]
    total = sum(e["comp_size"] for e in keep)
    print(f"{len(keep)} files, {total/1e6:.1f} MB compressed transfer")
    for e in keep:
        rel = e["name"].split("NISQA_Corpus/", 1)[-1]
        dest = OUT_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == e["uncomp_size"]:
            print("skip", rel)
            continue
        data = extract(e)
        assert len(data) == e["uncomp_size"], (rel, len(data), e["uncomp_size"])
        dest.write_bytes(data)
        print("ok  ", rel, f"{e['uncomp_size']:,} B")


if __name__ == "__main__":
    main()
