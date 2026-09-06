#!/bin/bash
# verify.sh — standalone verification for mods/jj-ds4-vision-deps
#
# Checks, against the *installed* environment, that the mod's three claims hold:
#   1. b12x overlay applied (MHC_SUPPORTED_RMS_EPS contains 1e-20)
#   2. b12x overlay modules import clean
#   3. LMCache is PR #44 (async_engine_driven + EngineDrivenContextPickle)
#
# Usage:
#   apt/container:  bash /path/to/mods/jj-ds4-vision-deps/verify.sh
# Exits 0 on all-pass, 1 otherwise.
set -u

fail=0
ok() { echo "  PASS: $*"; }
bad() { echo "  FAIL: $*"; fail=1; }

echo "== verify 1: b12x rms_eps (PR #306) =="
RUST_OES="$(python3 -c 'from b12x.norm.mhc import _impl as i; print(i.MHC_SUPPORTED_RMS_EPS)' 2>&1 || true)"
case "$RUST_OES" in
    *1e-20*) ok "MHC_SUPPORTED_RMS_EPS=$RUST_OES contains 1e-20" ;;
    *) bad "MHC_SUPPORTED_RMS_EPS=$RUST_OES missing 1e-20 (overlay not applied?)" ;;
esac

echo "== verify 2: b12x overlay modules import =="
if python3 - <<'PY' 2>&1
import b12x.attention._shared.mla.prefill
import b12x.attention._shared.mla.prefill_mg
import b12x.comm.pcie._cute_intrinsics
import b12x.comm.pcie._oneshot_cute
import b12x.comm.pcie.pcie_allreduce
import b12x.comm.pcie.pcie_oneshot
import b12x.norm.mhc._impl
PY
then
    ok "all 7 overlay modules import clean"
else
    bad "overlay module import failed"
fi

echo "== verify 3: LMCache is PR #44 =="
if python3 - <<'PY' 2>&1
import lmcache, pathlib
p = pathlib.Path(lmcache.__file__).parent
tok = p / "v1" / "multiprocess" / "transfer_context" / "async_engine_driven.py"
assert tok.exists(), f"async_engine_driven missing: {tok}"
from lmcache.v1.multiprocess.transfer_context.pickle import EngineDrivenContextPickle
from lmcache.v1.multiprocess.transfer_context.async_engine_driven import AsyncEngineDrivenTransferContext
print("lmcache", getattr(lmcache, "__version__", "?"), "@", lmcache.__file__)
PY
then
    ok "LMCache PR #44 markers present (async_engine_driven + EngineDrivenContextPickle)"
else
    bad "LMCache PR #44 markers missing — not PR #44?"
fi

echo
if [ "$fail" = "0" ]; then
    echo "verify: ALL PASS"
    exit 0
else
    echo "verify: FAILURES DETECTED"
    exit 1
fi
