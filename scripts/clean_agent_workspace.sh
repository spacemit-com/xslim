#!/usr/bin/env bash
#
# Enforce the agent-run retention policy from .agents/shared/run.md.
#
# Deletes `.agents/workspace/<prefix>-<slug>-<yyyymmdd>/` directories that are
# older than --days AND are not among the --keep newest runs of their prefix.
# A directory that is still a registered git worktree is removed with
# `git worktree remove --force` first. Dry run unless --apply is given.
#
# Only prefixes declared by AGENTS.md Rule 14 are ever touched; anything else is
# reported and skipped.

set -euo pipefail

days=14
keep=3
apply=0

usage() {
    cat <<'EOF'
Usage: scripts/clean_agent_workspace.sh [--days N] [--keep N] [--apply]

  --days N   Delete runs older than N days (default 14)
  --keep N   Always keep the N newest runs per prefix (default 3)
  --apply    Actually delete; without it the script only prints the selection
  -h, --help Show this help
EOF
}

while (($#)); do
    case "$1" in
        --days)
            days="${2:?--days requires a value}"
            shift 2
            ;;
        --keep)
            keep="${2:?--keep requires a value}"
            shift 2
            ;;
        --apply)
            apply=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "$days" =~ ^[0-9]+$ ]] || ! [[ "$keep" =~ ^[0-9]+$ ]]; then
    echo "--days and --keep must be non-negative integers" >&2
    exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$repo_root/.agents/workspace"
prefixes=(pass quant acc review maint)
cutoff="$(date -d "-${days} days" +%Y%m%d)"

if [[ ! -d "$workspace" ]]; then
    echo "no agent workspace at $workspace"
    exit 0
fi

shopt -s nullglob

declare -a delete_dirs=()
declare -a keep_dirs=()
declare -a unknown_dirs=()

for prefix in "${prefixes[@]}"; do
    candidates=("$workspace/${prefix}"-*-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9])
    ((${#candidates[@]})) || continue

    # Sort by the trailing yyyymmdd, then by name.
    sorted=()
    while IFS= read -r name; do
        [[ -n "$name" ]] && sorted+=("$name")
    done < <(
        for path in "${candidates[@]}"; do
            name="$(basename -- "$path")"
            [[ "$name" =~ ^${prefix}-.+-[0-9]{8}$ ]] || continue
            printf '%s\t%s\n' "${name: -8}" "$name"
        done | sort -k1,1 -k2,2 | cut -f2-
    )

    total=${#sorted[@]}
    keep_count=$keep
    ((keep_count > total)) && keep_count=$total

    for ((i = 0; i < total; i++)); do
        name="${sorted[$i]}"
        date_suffix="${name: -8}"
        if ((i >= total - keep_count)); then
            keep_dirs+=("$name")
        elif [[ "$date_suffix" < "$cutoff" ]]; then
            delete_dirs+=("$name")
        else
            keep_dirs+=("$name")
        fi
    done
done

# Report run-shaped directories whose prefix Rule 14 does not declare.
for path in "$workspace"/*; do
    [[ -d "$path" ]] || continue
    name="$(basename -- "$path")"
    if [[ "$name" =~ ^([a-z]+)-.+-[0-9]{8}$ ]]; then
        prefix="${BASH_REMATCH[1]}"
        declared=0
        for known in "${prefixes[@]}"; do
            if [[ "$prefix" == "$known" ]]; then
                declared=1
                break
            fi
        done
        ((declared)) || unknown_dirs+=("$name")
    else
        unknown_dirs+=("$name")
    fi
done

echo "workspace: $workspace"
echo "policy: delete runs older than $days days, keep $keep newest per prefix"
echo "cutoff: $cutoff"

for name in "${keep_dirs[@]}"; do
    echo "keep              $name"
done
for name in "${unknown_dirs[@]}"; do
    echo "skip (unrecognized) $name"
done

for name in "${delete_dirs[@]}"; do    path="$workspace/$name"
    if git -C "$repo_root" worktree list --porcelain 2>/dev/null | grep -Fxq "worktree $path"; then
        command_kind="worktree"
    else
        command_kind="rm"
    fi

    if ((apply)); then
        echo "delete            $name ($command_kind)"
        if [[ "$command_kind" == "worktree" ]]; then
            git -C "$repo_root" worktree remove --force "$path"
        else
            rm -rf -- "$path"
        fi
    else
        echo "would delete      $name ($command_kind)"
    fi
done

if ((${#delete_dirs[@]} == 0)); then
    echo "nothing to delete"
elif ((!apply)); then
    echo "dry run: re-run with --apply after archiving the selection"
fi

exit 0
