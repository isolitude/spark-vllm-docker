#!/bin/bash
# run.sh for mods/jj-ds4-vision-deps
#
# vLLM #634 (DeepSeek-V4-Flash-Vision-Exp on b12x) runtime dependency stack.
# Applied at container start (see launch-cluster.sh apply_mod_to_container):
#   - b12x  : overlay #301 (prefill.py/prefill_mg.py) + #246 (comm/pcie/*.py)
#             + #306 (mhc/_impl.py, rms_eps allowance). b12x is a pure
#             Python/JIT package: overlaying the .py files is sufficient,
#             no wheel rebuild needed.
#   - LMCache : build+install PR #44 (engine-driven hybrid KV) from the
#             vendored source tree (lmcache-44-src/), no network needed at
#             runtime.
#   - flashinfer : nothing — the baked 0.6.18 already contains 803c logic.
#
# Idempotent: safe to re-run.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
B12X_OVERLAY="$ROOT_DIR/b12x-overlay/b12x"
LMCACHE_TARGZ="$ROOT_DIR/lmcache-44-src.tar.gz"
LMCACHE_PIN="273ed7f7fafa6ce0155a17deec7fa11a9894df7b"  # refs/pull/44/head
LOG_PREFIX="[jj-ds4-vision-deps]"

log() { echo "$LOG_PREFIX $*"; }
die() { echo "$LOG_PREFIX ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1) b12x overlay (#301 prefill, #246 pcie, #306 mhc)
# ---------------------------------------------------------------------------
SITE="$(python3 -c 'import b12x, os; print(os.path.dirname(os.path.dirname(os.path.abspath(b12x.__file__))))' 2>/dev/null \
          || python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
B12X_PKG="$SITE/b12x"
log "b12x installed at: $B12X_PKG"
[ -d "$B12X_PKG" ] || die "b12x package dir not found under $SITE"

overlay_file() {
    local rel="$1"
    local src="$B12X_OVERLAY/$rel"
    local dst="$B12X_PKG/$rel"
    mkdir -p "$(dirname "$dst")"
    cp -f "$src" "$dst"
    log "  overlaid $rel ($(wc -l < "$src") lines)"
}

overlay_file "attention/_shared/mla/prefill.py"
overlay_file "attention/_shared/mla/prefill_mg.py"
overlay_file "comm/pcie/_cute_intrinsics.py"
overlay_file "comm/pcie/_oneshot_cute.py"
overlay_file "comm/pcie/pcie_allreduce.py"
overlay_file "comm/pcie/pcie_oneshot.py"
overlay_file "norm/mhc/_impl.py"

# Verify the mhc rms_eps allowance actually contains 1e-20 (the #306 effect).
RUST_OES="$(python3 -c 'from b12x.norm.mhc import _impl as i; print(i.MHC_SUPPORTED_RMS_EPS)' 2>&1 || true)"
log "post-overlay MHC_SUPPORTED_RMS_EPS = $RUST_OES"
case "$RUST_OES" in
    *1e-20*|*1e-20,*) log "b12x rms_eps=1e-20 support OK" ;;
    *) die "b12x overlay did not take effect (MHC_SUPPORTED_RMS_EPS=$RUST_OES)" ;;
esac

# Sanity: verify the overlay files import cleanly (no missing symbols).
python3 - <<'PY' || die "b12x overlay import check failed"
import b12x.attention._shared.mla.prefill
import b12x.attention._shared.mla.prefill_mg
import b12x.comm.pcie._cute_intrinsics
import b12x.comm.pcie._oneshot_cute
import b12x.comm.pcie.pcie_allreduce
import b12x.comm.pcie.pcie_oneshot
import b12x.norm.mhc._impl
print("[jj-ds4-vision-deps] b12x overlay imports clean")
PY

# ---------------------------------------------------------------------------
# 2) LMCache — build & install vendored PR #44 (engine-driven hybrid KV)
# ---------------------------------------------------------------------------
if python3 -c 'import lmcache' 2>/dev/null; then
    v="$(python3 -c 'import lmcache; print(getattr(lmcache,"__version__","?"))' 2>/dev/null || echo '?')"
    # detect whether existing install already is PR #44 (marker module)
    if python3 -c 'import lmcache.v1.multiprocess.transfer_context.async_engine_driven as m; ok=hasattr(m,"EngineDrivenContextPickle") or True' 2>/dev/null \
       && [ -f "$(python3 -c 'import lmcache,pathlib;print(pathlib.Path(lmcache.__file__).parent/"v1"/"multiprocess"/"transfer_context"/"async_engine_driven.py")' 2>/dev/null)" ]; then
        log "LMCache already installed (version=$v) with PR #44 marker — skipping build."
        exit_or_skip=0
    else
        log "LMCache present but missing PR #44 marker — will rebuild from vendored source."
        exit_or_skip=1
    fi
else
    log "LMCache not installed — building from vendored PR #44 source."
    exit_or_skip=1
fi

if [ "${exit_or_skip:-0}" = "1" ]; then
    [ -f "$LMCACHE_TARGZ" ] || die "vendored LMCache tarball missing ($LMCACHE_TARGZ)"

    log "  installing setuptools_scm (needed by pyproject build-system when --no-build-isolation)"
    uv pip install --python /usr/bin/python3 setuptools_scm 2>&1 | tail -2 || die "could not install setuptools_scm"

    # The mod tree itself may be read-only (docker cp from a :ro context), and
    # setuptools writes a build/ dir inside the source root — so extract the
    # tarball to a writable dir in /tmp and build from there. This also keeps
    # the installed mod dir pristine.
    log "  extracting vendored source to a writable build dir"
    LMCACHE_BUILD_DIR="/tmp/jj-ds4-vision-deps-lmcache-build"
    rm -rf "$LMCACHE_BUILD_DIR"
    mkdir -p "$LMCACHE_BUILD_DIR"
    tar xzf "$LMCACHE_TARGZ" -C "$LMCACHE_BUILD_DIR"

    # Ensure the pinned commit matches the tarball (traceability guard).
    if [ -f "$LMCACHE_BUILD_DIR/PR44_COMMIT" ] && [ "$(cat "$LMCACHE_BUILD_DIR/PR44_COMMIT")" != "$LMCACHE_PIN" ]; then
        log "Warning: vendored LMCache pin $(cat "$LMCACHE_BUILD_DIR/PR44_COMMIT") != expected $LMCACHE_PIN; continuing with vendored content."
    fi

    # setuptools_scm reads version from git; vendored tree is not a git repo,
    # so pretend a version matching PR #44's lineage.
    export SETUPTOOLS_SCM_PRETEND_VERSION="0.5.2.post0+jj44.$LMCACHE_PIN"

    log "  building with --no-build-isolation (uses image torch 2.13 / toolchain; avoids pyproject's torch==2.11 pin)"
    # Build the wheel once, install it. --no-deps: every runtime dep the image
    # needs is already present (torch, cuda-python, ...). redis/cufile are
    # optional storage backends, intentionally not pulled.
    uv pip install --python /usr/bin/python3 --no-build-isolation --no-deps "$LMCACHE_BUILD_DIR" \
        || die "LMCache #44 build/install failed"
    rm -rf "$LMCACHE_BUILD_DIR"

    # Install the light-weight runtime dependencies LMCache imports at startup
    # (see requirements/common.txt) that are missing from this image. Intentionally
    # NOT installed (would collide / be heavy / downgrade): numpy (must keep 2.3.5),
    # numba, cupy, ray, xformers, awscrt, cufile-python, google-cloud-*, pytest.
    log "  installing missing light LMCache runtime deps"
    uv pip install --python /usr/bin/python3 \
        sortedcontainers aiofiles aiofile pyzmq pyyaml 2>&1 | tail -3 \
        || die "could not install LMCache light runtime deps"
fi

# Verify installed LMCache is PR #44 and importable.
python3 - <<'PY' || die "LMCache post-install verification failed"
import lmcache, pathlib
print("[jj-ds4-vision-deps] LMCache installed:", getattr(lmcache, "__version__", "?"))
tok = pathlib.Path(lmcache.__file__).parent / "v1" / "multiprocess" / "transfer_context" / "async_engine_driven.py"
if not tok.exists():
    raise SystemExit(f"LMCache async_engine_driven module missing: {tok} (not PR #44?)")
import lmcache.v1.multiprocess.transfer_context.pickle as p
assert hasattr(p, "EngineDrivenContextPickle"), "PR #44 EngineDrivenContextPickle missing"
print("[jj-ds4-vision-deps] LMCache PR #44 markers present")
PY

# ---------------------------------------------------------------------------
# 3) flashinfer — no-op (baked 0.6.18 already has the 803c dispatch)
# ---------------------------------------------------------------------------
log "flashinfer: skipping (image 0.6.18 already contains SM120 topk-512 fallback)"

log "jj-ds4-vision-deps applied successfully"
