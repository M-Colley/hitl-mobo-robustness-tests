"""List the contents of the remote NISQA_Corpus.zip on Zenodo via HTTP range
requests, without downloading the 16 GB archive.

Parses EOCD / Zip64 EOCD, fetches the central directory, and prints all
entries that are CSV files (plus a summary of everything else).
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import requests

URL = "https://zenodo.org/records/4728081/files/NISQA_Corpus.zip?download=1"
TOTAL_SIZE = 15_945_499_507
OUT_DIR = Path(__file__).resolve().parent / "raw" / "nisqa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "metadata-extractor/0.1 (research use)"})


def fetch(start: int, end: int) -> bytes:
    """Fetch inclusive byte range."""
    r = session.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    r.raise_for_status()
    assert r.status_code == 206, r.status_code
    return r.content


def main() -> None:
    tail_len = 1 << 20  # 1 MB tail: EOCD + zip64 locator
    tail = fetch(TOTAL_SIZE - tail_len, TOTAL_SIZE - 1)

    eocd_pos = tail.rfind(b"PK\x05\x06")
    assert eocd_pos >= 0, "EOCD not found"
    (sig, disk, cd_disk, n_disk, n_total, cd_size, cd_offset, comment_len) = struct.unpack(
        "<IHHHHIIH", tail[eocd_pos : eocd_pos + 22]
    )
    print(f"EOCD: entries={n_total} cd_size={cd_size} cd_offset={cd_offset:#x}")

    if cd_offset == 0xFFFFFFFF or n_total == 0xFFFF or cd_size == 0xFFFFFFFF:
        loc_pos = tail.rfind(b"PK\x06\x07")
        assert loc_pos >= 0, "zip64 locator not found"
        (_, _, z64_eocd_offset, _) = struct.unpack("<IIQI", tail[loc_pos : loc_pos + 20])
        z64_pos = z64_eocd_offset - (TOTAL_SIZE - tail_len)
        if z64_pos < 0:
            z64 = fetch(z64_eocd_offset, z64_eocd_offset + 56 - 1)
        else:
            z64 = tail[z64_pos : z64_pos + 56]
        (sig, _, _, _, _, _, _, n_total, cd_size, cd_offset) = struct.unpack(
            "<IQHHIIQQQQ", z64[:56]
        )
        assert sig == 0x06064B50
        print(f"Zip64 EOCD: entries={n_total} cd_size={cd_size} cd_offset={cd_offset:#x}")

    # Fetch central directory (may overlap the tail we already have)
    cd_start_in_tail = cd_offset - (TOTAL_SIZE - tail_len)
    if cd_start_in_tail >= 0:
        cd = tail[cd_start_in_tail : cd_start_in_tail + cd_size]
    else:
        print(f"Fetching central directory ({cd_size / 1e6:.1f} MB)...")
        cd = fetch(cd_offset, cd_offset + cd_size - 1)

    entries = []
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            break
        (
            sig, ver_made, ver_need, flags, method, mtime, mdate, crc,
            comp_size, uncomp_size, name_len, extra_len, comment_len2,
            disk_start, int_attr, ext_attr, local_offset,
        ) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos : pos + 46])
        name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8", "replace")
        extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        # zip64 extra field
        if uncomp_size == 0xFFFFFFFF or comp_size == 0xFFFFFFFF or local_offset == 0xFFFFFFFF:
            epos = 0
            while epos + 4 <= len(extra):
                eid, esz = struct.unpack("<HH", extra[epos : epos + 4])
                edata = extra[epos + 4 : epos + 4 + esz]
                if eid == 0x0001:
                    fields = []
                    fpos = 0
                    for needed, cur in ((8, uncomp_size), (8, comp_size), (8, local_offset)):
                        pass
                    fpos = 0
                    if uncomp_size == 0xFFFFFFFF:
                        uncomp_size = struct.unpack("<Q", edata[fpos : fpos + 8])[0]; fpos += 8
                    if comp_size == 0xFFFFFFFF:
                        comp_size = struct.unpack("<Q", edata[fpos : fpos + 8])[0]; fpos += 8
                    if local_offset == 0xFFFFFFFF:
                        local_offset = struct.unpack("<Q", edata[fpos : fpos + 8])[0]; fpos += 8
                epos += 4 + esz
        entries.append(
            dict(name=name, method=method, comp_size=comp_size,
                 uncomp_size=uncomp_size, local_offset=local_offset)
        )
        pos += 46 + name_len + extra_len + comment_len2

    print(f"Parsed {len(entries)} entries")
    csvs = [e for e in entries if e["name"].lower().endswith(".csv")]
    others: dict[str, int] = {}
    for e in entries:
        ext = e["name"].rsplit(".", 1)[-1].lower() if "." in e["name"] else "(dir)"
        others[ext] = others.get(ext, 0) + 1
    print("Extension counts:", others)
    print(f"CSV files: {len(csvs)}, total uncompressed {sum(e['uncomp_size'] for e in csvs)/1e6:.1f} MB")
    for e in csvs:
        print(f"  {e['name']}  comp={e['comp_size']:,}  uncomp={e['uncomp_size']:,}  off={e['local_offset']:,}")

    (OUT_DIR / "zip_csv_index.json").write_text(json.dumps(csvs, indent=1))
    print("Saved index to", OUT_DIR / "zip_csv_index.json")


if __name__ == "__main__":
    main()
