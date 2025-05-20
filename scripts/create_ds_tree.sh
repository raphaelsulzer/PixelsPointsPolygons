#!/bin/bash

ROOT_DIR="/data/rsulzer/PixelsPointsPolygons/data/224"
MAX_FILES=${1:-5}
MAX_DIRS_LVL4=${2:-5}

# Custom sort orders
CUSTOM_SORT_ORDER_LVL1=("annotations" "images" "lidar" "ffl")
CUSTOM_SORT_ORDER_LVL2=("train" "val" "test")
CUSTOM_SORT_ORDER_LVL4=("0" "5000" "10000")

declare -A CUSTOM_ORDER_IDX_LVL1 CUSTOM_ORDER_IDX_LVL2 CUSTOM_ORDER_IDX_LVL4
for i in "${!CUSTOM_SORT_ORDER_LVL1[@]}"; do CUSTOM_ORDER_IDX_LVL1["${CUSTOM_SORT_ORDER_LVL1[$i]}"]=$i; done
for i in "${!CUSTOM_SORT_ORDER_LVL2[@]}"; do CUSTOM_ORDER_IDX_LVL2["${CUSTOM_SORT_ORDER_LVL2[$i]}"]=$i; done
for i in "${!CUSTOM_SORT_ORDER_LVL4[@]}"; do CUSTOM_ORDER_IDX_LVL4["${CUSTOM_SORT_ORDER_LVL4[$i]}"]=$i; done

get_sorted_entries() {
    local dir="$1"
    local depth="$2"
    local entries=()
    mapfile -t all_entries < <(ls "$dir" 2>/dev/null)

    local -a ordered
    local -A custom_map
    if [ "$depth" -eq 1 ]; then
        for entry in "${all_entries[@]}"; do
            [[ -n "${CUSTOM_ORDER_IDX_LVL1[$entry]}" ]] && ordered[CUSTOM_ORDER_IDX_LVL1[$entry]]="$entry"
        done
    elif [ "$depth" -eq 2 ]; then
        for entry in "${all_entries[@]}"; do
            [[ -n "${CUSTOM_ORDER_IDX_LVL2[$entry]}" ]] && ordered[CUSTOM_ORDER_IDX_LVL2[$entry]]="$entry"
        done
    elif [ "$depth" -eq 4 ]; then
        for entry in "${all_entries[@]}"; do
            [[ -n "${CUSTOM_ORDER_IDX_LVL4[$entry]}" ]] && ordered[CUSTOM_ORDER_IDX_LVL4[$entry]]="$entry"
        done
    fi

    local result=()
    for ((i = 0; i < 100; i++)); do
        [[ -n "${ordered[$i]}" ]] && result+=("${ordered[$i]}")
    done

    local custom_set="${result[*]}"
    local others=()
    for entry in "${all_entries[@]}"; do
        [[ "$custom_set" != *"$entry"* ]] && others+=("$entry")
    done
    IFS=$'\n' sorted_others=($(printf "%s\n" "${others[@]}" | sort))

    result+=("${sorted_others[@]}")
    printf "%s\n" "${result[@]}"
}

print_tree() {
    local dir="$1"
    local prefix="$2"
    local depth="$3"

    local subdirs=()
    local files=()
    local entries=()

    mapfile -t entries < <(get_sorted_entries "$dir" "$depth")

    for entry in "${entries[@]}"; do
        [ -d "$dir/$entry" ] && subdirs+=("$entry")
        [ -f "$dir/$entry" ] && files+=("$entry")
    done

    local show_subdirs=("${subdirs[@]}")
    if [ "$depth" -eq 4 ] && [ "${#subdirs[@]}" -gt "$MAX_DIRS_LVL4" ]; then
        show_subdirs=("${subdirs[@]:0:$MAX_DIRS_LVL4}")
    fi

    for i in "${!show_subdirs[@]}"; do
        local is_last=$(( i == ${#show_subdirs[@]} - 1 && ${#files[@]} == 0 ))
        local connector="├──"
        local next_prefix="$prefix│   "
        [ "$is_last" = 1 ] && connector="└──" && next_prefix="$prefix    "
        echo "${prefix}${connector} ${show_subdirs[$i]}"
        print_tree "$dir/${show_subdirs[$i]}" "$next_prefix" $((depth + 1))
    done

    if [ "$depth" -eq 4 ] && [ "${#subdirs[@]}" -gt "$MAX_DIRS_LVL4" ]; then
        echo "${prefix}    ... (${#subdirs[@]} dirs total)"
    fi

    local file_limit="${#files[@]}"
    [ "$file_limit" -gt "$MAX_FILES" ] && file_limit=$MAX_FILES

    for ((i=0; i<file_limit; i++)); do
        local connector="├──"
        [ "$i" -eq "$((file_limit - 1))" ] && connector="└──"
        echo "${prefix}${connector} ${files[$i]}"
    done

    if [ "${#files[@]}" -gt "$MAX_FILES" ]; then
        echo "${prefix}    ... (${#files[@]} files total)"
    fi
}

# Output markdown header
echo '```text'
echo "$(basename "$ROOT_DIR")"
print_tree "$ROOT_DIR" "" 1
echo '```'
