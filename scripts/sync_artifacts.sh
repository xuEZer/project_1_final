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
  pack      Archive the project's logs/ and cache/ directories. By default, the
            archive is created next to the project directory.
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
    local sources=()

    timestamp="$(date '+%Y%m%d_%H%M%S')"
    if [[ -z "$archive" ]]; then
        archive="$(dirname "$PROJECT_ROOT")/$(basename "$PROJECT_ROOT")_artifacts_${timestamp}.tar.gz"
    elif [[ "$archive" != /* ]]; then
        archive="$PWD/$archive"
    fi

    [[ "$archive" == *.tar.gz || "$archive" == *.tgz ]] \
        || die "archive must end in .tar.gz or .tgz"

    [[ -d "$PROJECT_ROOT/logs" ]] && sources+=(logs)
    [[ -d "$PROJECT_ROOT/cache" ]] && sources+=(cache)
    ((${#sources[@]} > 0)) || die "neither logs/ nor cache/ exists"

    mkdir -p "$(dirname "$archive")"
    tar -C "$PROJECT_ROOT" -czf "$archive" "${sources[@]}"

    printf 'Created: %s\n' "$archive"
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
