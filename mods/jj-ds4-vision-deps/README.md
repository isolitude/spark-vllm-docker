# mods/jj-ds4-vision-deps

Runtime dependency stack for the DeepSeek-V4-Flash-Vision-Exp recipe (vLLM
`local-inference-lab` PR #634 on b12x). This mod is applied **at container
start** (not build time) so the tested `vllm-node-b12x:latest` image is used
as-is, and the open upstream PRs it depends on do not have to be merged or
baked into a new image.

## What it does (in `run.sh`, idempotent)

1. **b12x overlay** — copies 7 files over the installed `b12x` package
   (found via `import b12x`, not hardcoded). b12x is a pure Python/JIT
   package, so source overlays are enough; kernels recompile on first use.

   | file | source | PR |
   |---|---|---|
   | `attention/_shared/mla/prefill.py` | PR #301 head | FP8 dual-cache prefill, sparse topk-512 |
   | `attention/_shared/mla/prefill_mg.py` | PR #301 head | same |
   | `comm/pcie/_cute_intrinsics.py` | PR #246 head | TP2 peer-push fix |
   | `comm/pcie/_oneshot_cute.py` | PR #246 head | PIECEWISE shape binding |
   | `comm/pcie/pcie_allreduce.py` | PR #246 head | TP2 allreduce |
   | `comm/pcie/pcie_oneshot.py` | PR #246 head | TP2 oneshot |
   | `norm/mhc/_impl.py` | PR #306 head | `rms_eps=1e-20` allowance |

   (`local-inference-lab/b12x` PRs #301, #246, #306 — all open. #306's exact
   change is _also_ already present in `master`, but the baked image is on
   `b58f34ea` which predates it, so we include `_impl.py` too.)

   Post-overlay checks: `MHC_SUPPORTED_RMS_EPS` must contain `1e-20`, and all
   7 modules import clean.

2. **LMCache install (PR #44, vendored tarball)** — `lmcache-44-src.tar.gz`
   (1.9M) is a vendored, gzip'd copy of the `refs/pull/44/head` source
   (`lmcache/` + `csrc/` + build scaffolding only — tests/docs/examples
   excluded; `__pycache__`/`*.pyc`/`build/`/`*.egg-info` stripped), pinned at
   `PR44_COMMIT = 273ed7f7` inside the archive. `run.sh` extracts it to a
   writable `/tmp` dir (setuptools writes `build/` into the source root, and
   mod trees may be mounted read-only), then installs with
   ```bash
   uv pip install --no-build-isolation --no-deps <staged>
   ```
   `--no-build-isolation` is required: the repo's pyproject build-system pins
   `torch==2.11.0` (incompatible with the image torch 2.13) — skipping build
   isolation uses the image's own torch/toolchain (verified present: torch
   2.13.0+cu130, ninja, gcc/g++, nvcc, wheel, cuda-python). `--no-deps` avoids
   re-resolving deps the image already provides (redis/cufile are optional
   storage backends, intentionally not pulled; vLLM uses the gpu/disk
   backend). `SETUPTOOLS_SCM_PRETEND_VERSION` is exported because
   setuptools_scm wants a git tree and the vendored dir is not git. Verified
   afterwards that the #44-introduced `async_engine_driven` module and
   `EngineDrivenContextPickle` are importable (i.e. it is really PR #44, not a
   dev-branch / 0.5.x wheel LMCache).

   > Why not the official `lmcache==0.5.4` wheel? It has an aarch64 cp312
   > wheel, but it does NOT contain PR #44's engine-driven path — e.g.
   > `v1/multiprocess/transfer_context/async_engine_driven.py` exists there
   > but lacks `EngineDrivenContextPickle` and `_submit_store_multigroup_async`,
   > and ~190 other files differ. #44 is based on
   > `release/v0.5.2-glm52-dcp-base`, a divergent lineage from 0.5.4.

3. **flashinfer — no-op.** The baked flashinfer `0.6.18`
   (git commit `18e5811d`) already contains the SM120 sparse-MLA topk-512
   fallback plus the `extra_page_block_size==2` dispatch that PR/commit
   `803c4664` (flashinfer-ai/flashinfer) adds. Verified: `csrc/sparse_mla_sm120_prefill.cu`
   in `18e5811d` already has the `extra_page_block_size == 2` branch.

## Provenance / verification

- b12x PR heads: `#301=223f88c2`, `#246=ea76030d`, `#306=3f6896dc`
  (local-inference-lab/b12x, all open, base on master history).
- LMCache: `refs/pull/44/head` = `273ed7f7` (base `release/v0.5.2-glm52-dcp-base`).
  Note the `dev` branch is a different lineage (Δ ~300 commits) and does NOT
  contain #44 → do not install `dev`; the recipe/IP depends on #44's
  engine-driven path.
- flashinfer commit `803c4664` (flashinfer-ai/flashinfer, not a branch head).

## Testing

- Overlay logic verified in a temp container (`vllm-node-b12x:latest`):
  `MHC_SUPPORTED_RMS_EPS` goes `(1e-06,1e-05)` → `(1e-20,1e-06,1e-05)`, 7
  modules import clean.
- LMCache install path: exercise via `launch-cluster.sh` with this mod applied
  (needs network + HTTP(S)_PROXY, GPU image). See the recipe's
  `deepseek-v4-flash-vision-exp.yaml`.
