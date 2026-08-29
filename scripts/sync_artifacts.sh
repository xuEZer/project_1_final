#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  scripts/sync_artifacts.sh pack [archive.tar.gz]
  scripts/sync_artifacts.sh unpack <archive.tar.gz> [--force]

Commands:
  pack      Archive only successfully completed experiment logs and their
            matching checkpoints. By default, the archive is created next to
            the project directory.
  unpack    Extract an archive into the project. Existing files are preserved
            unless --force is specified.
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

pack_artifacts() {
    local archive="${1:-}"
    local timestamp
    local log_path
    local ckpt_path
    local exp_name
    local artifacts=()
    local completed_count=0
    local checkpoint_count=0

    timestamp="$(date '+%Y%m%d_%H%M%S')"
    if [[ -z "$archive" ]]; then
        archive="$(dirname "$PROJECT_ROOT")/$(basename "$PROJECT_ROOT")_artifacts_${timestamp}.tar.gz"
    elif [[ "$archive" != /* ]]; then
        archive="$PWD/$archive"
    fi

    [[ "$archive" == *.tar.gz || "$archive" == *.tgz ]] \
        || die "archive must end in .tar.gz or .tgz"

    if [[ -d "$PROJECT_ROOT/logs" ]]; then
        while IFS= read -r -d '' log_path; do
            grep -qF "Experiment completed successfully!" "$log_path" \
                || continue
            artifacts+=("${log_path#"$PROJECT_ROOT/"}")
            ((completed_count += 1))

            exp_name="$(basename "$log_path" .log)"
            ckpt_path="$PROJECT_ROOT/cache/ckpt/$exp_name.pt"
            if [[ -f "$ckpt_path" ]]; then
                artifacts+=("${ckpt_path#"$PROJECT_ROOT/"}")
                ((checkpoint_count += 1))
            fi
        done < <(find "$PROJECT_ROOT/logs" -maxdepth 1 -type f -name '*.log' \
            -print0 | sort -z)
    fi
    ((completed_count > 0)) \
        || die "no successfully completed experiment logs were found"

    mkdir -p "$(dirname "$archive")"
    tar -C "$PROJECT_ROOT" -czf "$archive" "${artifacts[@]}"

    printf 'Created: %s\n' "$archive"
    printf 'Completed experiments: %d\n' "$completed_count"
    printf 'Matching checkpoints: %d\n' "$checkpoint_count"
    printf 'Size:    %s\n' "$(du -h "$archive" | cut -f1)"
    printf 'SHA256:  %s\n' "$(sha256sum "$archive" | cut -d' ' -f1)"
}

validate_archive() {
    local archive="$1"
    local entry

    while IFS= read -r entry; do
        entry="${entry#./}"
        [[ -z "$entry" ]] && continue
        case "$entry" in
            logs|logs/*|cache|cache/*) ;;
            *) die "archive contains an unexpected path: $entry" ;;
        esac
        [[ "$entry" != /* && "$entry" != *"../"* && "$entry" != ".." ]] \
            || die "archive contains an unsafe path: $entry"
    done < <(tar -tzf "$archive")
}

unpack_artifacts() {
    local archive="$1"
    local force="${2:-}"
    local tar_options=(-xzf "$archive" -C "$PROJECT_ROOT")
    local entry

    [[ -f "$archive" ]] || die "archive not found: $archive"
    [[ -z "$force" || "$force" == "--force" ]] || die "unknown option: $force"

    validate_archive "$archive"
    if [[ "$force" != "--force" ]]; then
        while IFS= read -r entry; do
            entry="${entry#./}"
            [[ -z "$entry" || "$entry" == */ ]] && continue
            [[ ! -e "$PROJECT_ROOT/$entry" && ! -L "$PROJECT_ROOT/$entry" ]] \
                || die "destination already exists: $entry (use --force to overwrite)"
        done < <(tar -tzf "$archive")
    fi

    if ! tar "${tar_options[@]}"; then
        die "extraction failed"
    fi

    printf 'Extracted into: %s\n' "$PROJECT_ROOT"
}

case "${1:-}" in
    pack)
        [[ $# -le 2 ]] || { usage >&2; exit 1; }
        pack_artifacts "${2:-}"
        ;;
    unpack)
        [[ $# -ge 2 && $# -le 3 ]] || { usage >&2; exit 1; }
        unpack_artifacts "$2" "${3:-}"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
