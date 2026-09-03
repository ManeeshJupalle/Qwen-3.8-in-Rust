"""Fetch the safetensors header (name -> dtype/shape/offsets) of every shard
listed in model.safetensors.index.json WITHOUT downloading the shards.

A safetensors file starts with an 8-byte little-endian u64 = header length,
followed by that many bytes of JSON. We issue an HTTP Range request for the
first 8 bytes, then for the header. Nothing else is fetched.

Writes: models/<repo>/shard_headers/<shard>.header.json (one per shard)
Usage:  python tools/st_headers.py Qwen/Qwen3.8-27B models/Qwen3.8-27B
Uses HF_TOKEN from the environment if set.
"""
import json
import os
import struct
import sys

import requests
from huggingface_hub import hf_hub_url


def fetch_range(url, start, end, token):
    headers = {"Range": f"bytes={start}-{end}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, allow_redirects=True, timeout=60)
    r.raise_for_status()
    assert r.status_code == 206, f"expected 206 Partial Content, got {r.status_code}"
    return r.content


def main():
    repo_id, local_dir = sys.argv[1], sys.argv[2]
    token = os.environ.get("HF_TOKEN")
    idx = json.load(open(os.path.join(local_dir, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    out_dir = os.path.join(local_dir, "shard_headers")
    os.makedirs(out_dir, exist_ok=True)
    for shard in shards:
        out_path = os.path.join(out_dir, shard + ".header.json")
        if os.path.exists(out_path):
            print(f"cached  {shard}")
            continue
        url = hf_hub_url(repo_id, shard)
        (hlen,) = struct.unpack("<Q", fetch_range(url, 0, 7, token))
        hdr = fetch_range(url, 8, 8 + hlen - 1, token)
        assert len(hdr) == hlen, f"{shard}: got {len(hdr)} of {hlen} header bytes"
        parsed = json.loads(hdr.decode("utf-8"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"header_len": hlen, "header": parsed}, f, indent=1)
        n = len([k for k in parsed if k != "__metadata__"])
        print(f"fetched {shard}: header {hlen} bytes, {n} tensors")


if __name__ == "__main__":
    main()
