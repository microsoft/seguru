#!/usr/bin/env bash
# Verification harness for the SeGuRu case studies (v2).
#
#   ./verify.sh                     # all five case studies
#   ./verify.sh aes gpusorting      # a subset
#
# For each selected case study this runs `cargo test --release -p <package>
# -- --test-threads=1`, then audits the whole tree for `unsafe`, ignoring the
# deliberately-unsafe `cuda_ffi.rs` FFI shims that bind the CUDA C++ reference
# baselines behind the `bench` feature.
#
# Note: the workspace `target/` directory may be in use by another build; cargo
# will block on the file lock until it is free. That wait is legitimate and this
# script deliberately imposes no timeout.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"

ALL_CRATES=(aes polybench kernelbench gpusorting heongpu)

if [ "$#" -gt 0 ]; then
    CRATES=("$@")
else
    CRATES=("${ALL_CRATES[@]}")
fi

# Directory name -> cargo package name, read out of the crate's Cargo.toml.
package_of() {
    local dir="$1"
    sed -n 's/^name[[:space:]]*=[[:space:]]*"\(.*\)"/\1/p' "$ROOT/$dir/Cargo.toml" | head -n 1
}

declare -A STATUS
declare -A DETAIL

run_crate() {
    local dir="$1"

    if [ ! -f "$ROOT/$dir/Cargo.toml" ]; then
        STATUS["$dir"]="SKIP"
        DETAIL["$dir"]="not implemented yet (no Cargo.toml)"
        return 0
    fi

    local pkg
    pkg="$(package_of "$dir")"
    if [ -z "$pkg" ]; then
        STATUS["$dir"]="SKIP"
        DETAIL["$dir"]="not implemented yet (no package name in Cargo.toml)"
        return 0
    fi

    if [ ! -f "$ROOT/$dir/src/lib.rs" ] && [ ! -d "$ROOT/$dir/src" ]; then
        STATUS["$dir"]="SKIP"
        DETAIL["$dir"]="not implemented yet (no src/)"
        return 0
    fi

    echo "=== $dir ($pkg): cargo test --release -p $pkg -- --test-threads=1"
    local log="$ROOT/.verify-$dir.log"
    if (cd "$ROOT" && cargo test --release -p "$pkg" -- --test-threads=1) 2>&1 | tee "$log"; then
        STATUS["$dir"]="PASS"
        DETAIL["$dir"]="$(grep -c '^test .* ok$' "$log" || true) test(s) passed"
    else
        STATUS["$dir"]="FAIL"
        DETAIL["$dir"]="see $log"
    fi
    echo
}

for c in "${CRATES[@]}"; do
    run_crate "$c"
done

# ---------------------------------------------------------------------------
# no-unsafe audit
# ---------------------------------------------------------------------------
echo "=== no-unsafe audit (excluding cuda_ffi.rs)"
UNSAFE_HITS=""
while IFS= read -r -d '' f; do
    [ "$(basename "$f")" = "cuda_ffi.rs" ] && continue
    # Strip `//` line comments first, so that prose mentioning the word does not
    # trip the audit; only real `unsafe` code counts.
    hits="$(sed 's|//.*||' "$f" | grep -n '\bunsafe\b' || true)"
    if [ -n "$hits" ]; then
        UNSAFE_HITS+="$(echo "$hits" | sed "s|^|${f#"$ROOT/"}:|")"$'\n'
    fi
done < <(find "$ROOT" -mindepth 3 -path "$ROOT/target" -prune -o -type f -name '*.rs' -path '*/src/*' -print0)

if [ -n "$UNSAFE_HITS" ]; then
    echo 'FAIL: unsafe found outside cuda_ffi.rs:'
    printf '%s' "$UNSAFE_HITS"
    UNSAFE_STATUS="FAIL"
else
    echo 'PASS: no unsafe outside cuda_ffi.rs'
    UNSAFE_STATUS="PASS"
fi
echo

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
echo "=== summary"
printf '%-14s %-6s %s\n' "CASE STUDY" "STATUS" "DETAIL"
printf '%-14s %-6s %s\n' "----------" "------" "------"
EXIT=0
for c in "${CRATES[@]}"; do
    printf '%-14s %-6s %s\n' "$c" "${STATUS[$c]}" "${DETAIL[$c]}"
    if [ "${STATUS[$c]}" = "FAIL" ]; then EXIT=1; fi
done
printf '%-14s %-6s %s\n' "unsafe-audit" "$UNSAFE_STATUS" "*/src/**/*.rs excluding cuda_ffi.rs"
if [ "$UNSAFE_STATUS" = "FAIL" ]; then EXIT=1; fi

exit "$EXIT"
