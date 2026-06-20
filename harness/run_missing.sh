#!/usr/bin/env bash
#
# Batch-run Devin against missing AutomationBench tasks.
#
# Usage:
#   ./harness/run_missing.sh                  # run all 45 missing tasks
#   ./harness/run_missing.sh --limit 5        # run first 5 only
#   ./harness/run_missing.sh --start 10       # start from task #10
#   ./harness/run_missing.sh --dry-run        # print what would run
#
# Each task gets its own Devin session (non-interactive, -p mode).
# Results are saved to data/enriched/results.jsonl
# Traces are extracted from sessions.db after each run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MISSING_JSON="$REPO_ROOT/data/missing_tasks.json"
RESULTS_DIR="$REPO_ROOT/data/enriched"
TRACES_DIR="$REPO_ROOT/data/enriched/traces"
RESULTS_FILE="$RESULTS_DIR/results.jsonl"
SESSIONS_DB="$HOME/.local/share/devin/cli/sessions.db"
TMP_CONFIG="$REPO_ROOT/.devin/config.json"

LIMIT=9999
START=0
DRY_RUN=false
MODEL="claude-opus-4.8"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit) LIMIT="$2"; shift 2 ;;
        --start) START="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR" "$TRACES_DIR"

# Read tasks from JSON
TASKS=$(python3 -c "
import json
with open('$MISSING_JSON') as f:
    data = json.load(f)
for t in data['tasks']:
    print(f\"{t['domain']}\t{t['task_index']}\t{t['task']}\t{t['example_id']}\t{t['best_partial_credit']}\")
")

TOTAL=$(echo "$TASKS" | wc -l | tr -d ' ')
echo "=== AutomationBench Missing Tasks Runner ==="
echo "Tasks: $TOTAL total, starting at #$START, limit $LIMIT, model: $MODEL"
echo "Results: $RESULTS_FILE"
echo "Traces:  $TRACES_DIR/"
echo ""

IDX=0
DONE=0

while IFS=$'\t' read -r DOMAIN TASK_INDEX TASK_NAME EXAMPLE_ID BEST_PC; do
    # Skip/limit logic
    if [ "$IDX" -lt "$START" ]; then
        IDX=$((IDX + 1))
        continue
    fi
    if [ "$DONE" -ge "$LIMIT" ]; then
        break
    fi

    echo "[$((DONE + 1))/$LIMIT] Task: $TASK_NAME (domain=$DOMAIN, index=$TASK_INDEX, best_pc=$BEST_PC)"

    if [ "$DRY_RUN" = true ]; then
        echo "  [dry-run] Would launch: devin -p with domain=$DOMAIN task_index=$TASK_INDEX"
        IDX=$((IDX + 1))
        DONE=$((DONE + 1))
        continue
    fi

    # Write per-task MCP config
    echo "  [$(date +%H:%M:%S)] Writing MCP config: domain=$DOMAIN task_index=$TASK_INDEX"
    cat > "$TMP_CONFIG" << CONFIGEOF
{
  "mcpServers": {
    "automationbench": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "./benchmarks",
        "python", "../harness/mcp_server.py",
        "--domain", "$DOMAIN",
        "--task-index", "$TASK_INDEX"
      ]
    }
  }
}
CONFIGEOF
    echo "  [$(date +%H:%M:%S)] Config written: $TMP_CONFIG"

    # Record timestamp before launching (to find session after)
    BEFORE_TS=$(date +%s)

    # Launch Devin in non-interactive mode
    LOG_FILE="$RESULTS_DIR/logs/${TASK_NAME}.log"
    echo "  [$(date +%H:%M:%S)] Launching Devin (-p mode, --model $MODEL, --permission-mode dangerous)..."
    echo "  [$(date +%H:%M:%S)] Log: $LOG_FILE"
    echo "  [$(date +%H:%M:%S)] Waiting for Devin to finish (this may take several minutes)..."

    EXPORT_FILE="$TRACES_DIR/${TASK_NAME}.json"
    MAX_SECONDS=600
    START_TIME=$(date +%s)
    # Launch devin in background, kill after MAX_SECONDS if still running
    devin -p "Use the tools provided by the automationbench MCP to complete the task. Call get_task first to see what you need to do, then solve it using tools provided by this MCP. When you're done, call score to check your result. After scoring, report the final score and stop." \
        --model "$MODEL" \
        --permission-mode dangerous \
        --export "$EXPORT_FILE" \
        2>&1 | tee "$LOG_FILE" &
    DEVIN_PID=$!
    ( sleep "$MAX_SECONDS" && kill "$DEVIN_PID" 2>/dev/null && echo "  [$(date +%H:%M:%S)] TIMEOUT after ${MAX_SECONDS}s (safety limit)" ) &
    TIMER_PID=$!
    wait "$DEVIN_PID" 2>/dev/null
    kill "$TIMER_PID" 2>/dev/null
    wait "$TIMER_PID" 2>/dev/null
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo "  [$(date +%H:%M:%S)] Devin finished (${ELAPSED}s elapsed)"

    # Find the session that was just created (match by timestamp + prompt content)
    echo "  [$(date +%H:%M:%S)] Looking up session in sessions.db..."
    SESSION_ID=$(sqlite3 "$SESSIONS_DB" \
        "SELECT id FROM sessions WHERE created_at >= $BEFORE_TS AND title LIKE '%automationbench%' ORDER BY created_at DESC LIMIT 1;" 2>/dev/null)
    # Fallback: any session created after our timestamp
    if [ -z "$SESSION_ID" ]; then
        SESSION_ID=$(sqlite3 "$SESSIONS_DB" \
            "SELECT id FROM sessions WHERE created_at >= $BEFORE_TS ORDER BY created_at DESC LIMIT 1;" 2>/dev/null || echo "unknown")
    fi
    SESSION_TITLE=$(sqlite3 "$SESSIONS_DB" \
        "SELECT title FROM sessions WHERE id = '$SESSION_ID';" 2>/dev/null || echo "unknown")

    echo "  [$(date +%H:%M:%S)] Session: $SESSION_ID ($SESSION_TITLE)"
    echo "  [$(date +%H:%M:%S)] Resume:  devin -r $SESSION_ID"

    # Count messages in session
    MSG_COUNT=$(sqlite3 "$SESSIONS_DB" \
        "SELECT COUNT(*) FROM message_nodes WHERE session_id = '$SESSION_ID';" 2>/dev/null || echo "0")
    echo "  [$(date +%H:%M:%S)] Messages in session: $MSG_COUNT"

    # Extract the score from the session (look for the score tool result)
    echo "  [$(date +%H:%M:%S)] Extracting score..."
    SCORE=$(sqlite3 "$SESSIONS_DB" \
        "SELECT chat_message FROM message_nodes WHERE session_id = '$SESSION_ID' AND chat_message LIKE '%Score:%' ORDER BY node_id DESC LIMIT 1;" 2>/dev/null \
        | python3 -c "
import sys, json, re
try:
    msg = json.loads(sys.stdin.read())
    content = msg.get('content', '')
    if isinstance(content, list):
        content = ' '.join(c.get('text', '') for c in content if isinstance(c, dict))
    match = re.search(r'Score:\s*([\d.]+)%', content)
    if match:
        print(round(float(match.group(1)) / 100, 4))
    else:
        print('unknown')
except:
    print('unknown')
" 2>/dev/null || echo "unknown")

    echo "  [$(date +%H:%M:%S)] Score: $SCORE"

    # Export trace
    TRACE_FILE="$TRACES_DIR/${TASK_NAME}.jsonl"
    echo "  [$(date +%H:%M:%S)] Exporting trace to $TRACE_FILE..."
    sqlite3 "$SESSIONS_DB" \
        "SELECT chat_message FROM message_nodes WHERE session_id = '$SESSION_ID' ORDER BY node_id;" \
        > "$TRACE_FILE" 2>/dev/null || true

    TRACE_LINES=$(wc -l < "$TRACE_FILE" 2>/dev/null | tr -d ' ')
    echo "  [$(date +%H:%M:%S)] Trace: $TRACE_LINES messages"

    # Append result
    python3 -c "
import json, sys
result = dict(
    task=sys.argv[1],
    domain=sys.argv[2],
    task_index=int(sys.argv[3]),
    example_id=int(sys.argv[4]),
    session_id=sys.argv[5],
    score=sys.argv[6],
    best_leaderboard_pc=float(sys.argv[7]),
    trace_file=sys.argv[8],
    trace_messages=int(sys.argv[9]),
    elapsed_seconds=int(sys.argv[10]),
    model=sys.argv[11],
    resume_cmd='devin -r ' + sys.argv[5],
)
print(json.dumps(result))
" "$TASK_NAME" "$DOMAIN" "$TASK_INDEX" "$EXAMPLE_ID" "$SESSION_ID" "$SCORE" "$BEST_PC" "${TASK_NAME}.jsonl" "$TRACE_LINES" "$ELAPSED" "$MODEL" >> "$RESULTS_FILE"

    echo "  [$(date +%H:%M:%S)] Result appended to $RESULTS_FILE"
    echo "  [$(date +%H:%M:%S)] Done. (score=$SCORE, ${ELAPSED}s, $TRACE_LINES messages)"
    echo ""

    IDX=$((IDX + 1))
    DONE=$((DONE + 1))
done <<< "$TASKS"

echo "=== Completed $DONE tasks ==="
echo "Results: $RESULTS_FILE"

# Print summary
if [ -f "$RESULTS_FILE" ]; then
    echo ""
    echo "Summary:"
    python3 -c "
import json
results = [json.loads(l) for l in open('$RESULTS_FILE')]
print(f'  Total runs: {len(results)}')
scores = [float(r['score']) for r in results if r['score'] != 'unknown']
if scores:
    improved = sum(1 for r in results if r['score'] != 'unknown' and float(r['score']) >= 0.5)
    print(f'  Scored: {len(scores)}')
    print(f'  Score >= 0.5: {improved}')
    print(f'  Avg score: {sum(scores)/len(scores):.3f}')
"
fi
