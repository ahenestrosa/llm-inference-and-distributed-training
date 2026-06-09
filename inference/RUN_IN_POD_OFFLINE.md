# Running in a Pod When `uv sync` Stalls

If a fresh `uv sync` (or `uv pip install`) hangs on "Preparing packages..." with progress bars frozen at non-trivial percentages, the issue isn't PyPI throttling or pod-side throughput — it's uv's concurrent downloader interacting badly with the pod's network stack. This guide covers the symptom, the root cause, and the offline-install workaround that reliably gets a working `.venv` on the affected pods.

## Symptom

- `uv sync` resolves quickly but then sits in `Preparing packages... (0/N)` with progress bars frozen.
- Each in-flight stream stops at roughly the same partial-file size (observed: ~2,002,497 bytes / stream).
- Aggregate network throughput on the interface drops to ~0 even though uv is still alive.
- Lowering `UV_CONCURRENT_DOWNLOADS` (even to 1 or 2) does **not** help; uv keeps 50+ partial files open regardless.

## Root cause

uv opens ~50 concurrent HTTP/2 streams per host during package preparation. On affected RunPod containers (likely a conntrack / NAT-level issue, but not exhausted conntrack — observed 27 / 262,144), every stream hangs at the same point and the wall-clock stalls.

What ruled out the obvious suspects:

- **Not PyPI throttling.** Single-stream `curl` to `files.pythonhosted.org` ran at 12–90 MB/s.
- **Not pod-side bandwidth.** Single-stream `curl` to `download.pytorch.org` ran at ~170 MB/s.
- **Not big-package misrouting.** Even small packages stalled.
- **Not conntrack exhaustion.** `sudo conntrack -C` showed 27 / 262144.
- **Not the resolver.** uv's metadata fetches stalled the same way, so even `uv pip install --dry-run` couldn't complete fully online.

So: anything that multiplexes lots of concurrent streams over HTTP/2 to a single host stalls. Anything sequential is fine.

## Fix: resolve offline, fetch with curl, install offline

The plan: get one wheel locally to seed uv's cache, use that to drive a fully-offline resolution, then curl every pinned wheel sequentially with timeouts, then install offline.

### 1. Provision Python 3.12

System Python is often 3.11 on these images, but `flash-attn` ships `cp312` only and the project requires `>=3.12`:

```bash
uv python install 3.12
uv venv --python 3.12
```

### 2. Seed one wheel, then resolve fully offline

Download the torch wheel with plain curl, then let uv resolve the full pinned closure without network:

```bash
mkdir -p /workspace/wheels
curl -L --connect-timeout 15 --max-time 600 \
     --speed-time 20 --speed-limit 10000 \
     -o /workspace/wheels/torch-2.8.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl \
     "https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"

# Drives the resolver entirely off local + cached index metadata.
uv pip install --offline \
   --find-links /workspace/wheels \
   --dry-run \
   "torch==2.8.*" transformers accelerate "flash-attn==2.8.3"
```

With torch local and the index metadata already cached from earlier (failed) attempts, this produced the full 53-package pinned closure with **zero network**.

### 3. Curl each pin sequentially, with guards

Map every pinned `name==version` to a wheel URL (PyPI JSON API + the PyTorch index), then `curl` them one at a time:

```bash
curl -L \
     --connect-timeout 15 \
     --max-time 600 \
     --speed-time 20 --speed-limit 10000 \
     -o "/workspace/wheels/$FILENAME" \
     "$URL"
```

The `--speed-time 20 --speed-limit 10000` pair is essential: Fastly occasionally leaves a connection open after the file body is fully transferred (one `curl` blocked 5 min before the body terminated). The speed guard aborts after 20 s under 10 KB/s and retries.

### 4. Install offline

```bash
uv pip install --offline --no-index \
   --find-links /workspace/wheels \
   "torch==2.8.*" transformers accelerate "flash-attn==2.8.3"
```

No network, no concurrency stall, ~30 s.

## Gotchas worth remembering

- **`download.pytorch.org/whl/cu128/<file>` 403s for cudnn and triton.** The index hrefs actually point at `download-r2.pytorch.org`. Easiest fix: grab those two wheels from PyPI instead. Don't waste time chasing the R2 URL.
- **Reference flash-attn by `flash-attn==2.8.3` in the offline install, not the GitHub URL.** A `@ <url>` spec makes uv try that exact URL even under `--offline --find-links`, defeating the local cache.
- **Triton has two pins floating around.** Torch 2.8 pins `triton==3.4.0`; the repo's `[tool.uv]` `override-dependencies` says `triton>=3.0`, which uv reads from the project dir and bumps to whatever's current (3.7.0 at time of writing). Pre-fetch **both** wheels into `/workspace/wheels/` so resolution succeeds regardless of which uv picks.
- **HF gated downloads hit the same concurrency stall** because `hf_xet` multiplexes. Disable it:

  ```bash
  HF_HUB_DISABLE_XET=1 hf download meta-llama/Llama-3.1-8B \
      --exclude "original/*"
  ```

  Sequential downloads ran clean at ~294 MB/s. `--exclude "original/*"` skips the 16 GB redundant `.pth` — `transformers` only needs the safetensors.
- **One wheel was silently truncated** by an earlier stalled run. Sweep the wheel dir before trusting it:

  ```bash
  python -c "
  import sys, zipfile, pathlib
  for p in pathlib.Path('/workspace/wheels').glob('*.whl'):
      try:
          with zipfile.ZipFile(p) as z:
              bad = z.testzip()
              if bad: print('CORRUPT:', p, bad); sys.exit(1)
      except zipfile.BadZipFile:
          print('CORRUPT:', p); sys.exit(1)
  print('all wheels OK')
  "
  ```

## Net result

`/workspace/wheels/` holds the full 53-wheel offline set on persistent volume; the venv has `torch 2.8.0+cu128 / flash-attn 2.8.3 / transformers 5.10.2`. `pyproject.toml` and `uv.lock` untouched.

## When to use this guide

- `uv sync` or `uv pip install` stalls with progress bars frozen on a fresh pod.
- You've ruled out genuine PyPI/pod-side bandwidth issues via single-stream `curl`.
- You want a reproducible offline install that survives pod restarts (wheels on `/workspace/wheels` persist with the volume).

If `uv sync` works normally, ignore this file and follow [RUN_IN_POD.md](./RUN_IN_POD.md).
