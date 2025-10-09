#!/bin/bash
# Compare benchmark JSON files and SRT outputs
# Usage: bash scripts/compare_benchmarks.sh [pattern]
# Example: bash scripts/compare_benchmarks.sh "Weighted_Scorecard"

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# Helper function to colorize output
colorize() {
    local color="$1"
    shift
    echo -e "${color}$@${RESET}"
}

PATTERN="${1:-*}"
FILES=($(ls -1tr benchmarks/${PATTERN}*.json 2>/dev/null))

if [ ${#FILES[@]} -eq 0 ]; then
    colorize "$RED" "❌ No benchmark files found matching pattern: $PATTERN"
    exit 1
fi

colorize "$BOLD$CYAN" "🔍 Found ${#FILES[@]} benchmark files to compare"
echo ""

# Get timestamp type from config
get_timestamp_type() {
    local file="$1"
    jq -r '.config.timestamp_type // .config.return_timestamps // "unknown"' "$file" 2>/dev/null
}

# Auto-detect variants based on preprocessing config
get_variant_name() {
    local file="$1"
    local stab=$(jq -r '.config.stabilize // false' "$file" 2>/dev/null)
    local dem=$(jq -r '.config.demucs // false' "$file" 2>/dev/null)
    local vad=$(jq -r '.config.vad // false' "$file" 2>/dev/null)
    
    local parts=()
    [ "$stab" = "true" ] && parts+=("stabilize")
    [ "$dem" = "true" ] && parts+=("demucs")
    [ "$vad" = "true" ] && parts+=("vad")
    
    if [ ${#parts[@]} -eq 0 ]; then
        echo "baseline"
    else
        IFS='+'; echo "${parts[*]}"
    fi
}

# Group files by timestamp type
declare -A files_by_ts_type

for f in "${FILES[@]}"; do
    ts_type=$(get_timestamp_type "$f")
    if [ -z "${files_by_ts_type[$ts_type]}" ]; then
        files_by_ts_type[$ts_type]="$f"
    else
        files_by_ts_type[$ts_type]="${files_by_ts_type[$ts_type]} $f"
    fi
done

colorize "$BOLD$BLUE" "📊 Benchmark Groups by Timestamp Type"
echo ""
for ts_type in "${!files_by_ts_type[@]}"; do
    count=$(echo "${files_by_ts_type[$ts_type]}" | wc -w)
    colorize "$CYAN" "  • ${ts_type}: ${count} benchmarks"
done
echo ""

# Process each timestamp type group
for ts_type in "${!files_by_ts_type[@]}"; do
    colorize "$DIM" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    colorize "$BOLD$MAGENTA" "🎯 Timestamp Type: ${ts_type}"
    echo ""
    
    # Convert space-separated string to array
    group_files=(${files_by_ts_type[$ts_type]})
    
    colorize "$BOLD$YELLOW" "📈 Quality Metrics Comparison (${ts_type})"
    echo ""
    echo "| Variant | Score | Max Dur | Avg Dur | Min Dur | Segments | Too Short | Within | Too Long |"
    echo "|---------|-------|---------|---------|---------|----------|-----------|--------|----------|"
    
    # Print metrics for each file in this group
    for f in "${group_files[@]}"; do
        variant=$(get_variant_name "$f")
        
        score=$(jq -r '.format_quality.srt.score' "$f" 2>/dev/null | cut -c1-5)
        max_dur=$(jq -r '.format_quality.srt.details.duration_stats.max_seconds' "$f" 2>/dev/null | cut -c1-4)
        avg_dur=$(jq -r '.format_quality.srt.details.duration_stats.average_seconds' "$f" 2>/dev/null | cut -c1-4)
        min_dur=$(jq -r '.format_quality.srt.details.duration_stats.min_seconds' "$f" 2>/dev/null | cut -c1-4)
        too_short=$(jq -r '.format_quality.srt.details.boundary_counts.too_short' "$f" 2>/dev/null)
        within=$(jq -r '.format_quality.srt.details.boundary_counts.within_range' "$f" 2>/dev/null)
        too_long=$(jq -r '.format_quality.srt.details.boundary_counts.too_long' "$f" 2>/dev/null)
        
        # Count segments from SRT
        srt_file="transcripts-srt/$(basename "$f" .json).srt"
        seg_count=$(grep -c "^[0-9]\+$" "$srt_file" 2>/dev/null || echo "N/A")
        
        printf "| %-20s | %5s | %7ss | %7ss | %7ss | %8s | %9s | %6s | %8s |\n" \
            "$variant" "$score" "$max_dur" "$avg_dur" "$min_dur" "$seg_count" "$too_short" "$within" "$too_long"
    done
    
    echo ""
    colorize "$BOLD$CYAN" "🔄 SRT Content Differences (${ts_type})"
    echo ""
    
    # Get baseline (first file in this group)
    baseline_file="${group_files[0]}"
    baseline_srt="transcripts-srt/$(basename "$baseline_file" .json).srt"
    baseline_variant=$(get_variant_name "$baseline_file")
    
    colorize "$DIM" "Baseline: $baseline_variant"
    echo ""
    
    # Compare each variant to baseline within this group
    for f in "${group_files[@]:1}"; do
        variant=$(get_variant_name "$f")
        srt_file="transcripts-srt/$(basename "$f" .json).srt"
        
        if [ -f "$srt_file" ] && [ -f "$baseline_srt" ]; then
            if diff -q "$baseline_srt" "$srt_file" > /dev/null 2>&1; then
                colorize "$GREEN" "  ✅ ${variant}: Identical to baseline"
            else
                diff_lines=$(diff "$baseline_srt" "$srt_file" 2>/dev/null | wc -l)
                colorize "$YELLOW" "  🔧 ${variant}: ${diff_lines} diff lines (timestamp/text differences)"
            fi
        else
            colorize "$RED" "  ❌ ${variant}: SRT file not found"
        fi
    done
    
    echo ""
    colorize "$BOLD$BLUE" "🔍 Sample Differences (${ts_type})"
    echo ""
    
    # Show sample differences for first non-identical variant in this group
    found_diff=false
    for f in "${group_files[@]:1}"; do
        variant=$(get_variant_name "$f")
        srt_file="transcripts-srt/$(basename "$f" .json).srt"
        
        if [ -f "$srt_file" ] && [ -f "$baseline_srt" ]; then
            if ! diff -q "$baseline_srt" "$srt_file" > /dev/null 2>&1; then
                colorize "$YELLOW" "Sample: ${baseline_variant} vs ${variant}"
                echo ""
                echo '```diff'
                diff "$baseline_srt" "$srt_file" 2>/dev/null | head -30
                echo '```'
                echo ""
                found_diff=true
                break  # Only show first difference
            fi
        fi
    done
    
    if [ "$found_diff" = false ]; then
        colorize "$GREEN" "  ✨ All variants identical to baseline"
        echo ""
    fi
done

colorize "$DIM" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
colorize "$BOLD$GREEN" "✅ Comparison complete - Analyzed ${#FILES[@]} benchmarks across ${#files_by_ts_type[@]} timestamp type(s)"
