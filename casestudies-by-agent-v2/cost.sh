#!/usr/bin/env bash
# Report the agent/token cost of building the case studies in this folder.
#
# Reads the Copilot CLI local session store, which records one row per model
# request with token counts and the billed cost in nano-AIU (AI Units).
#
# Usage:  ./cost.sh [SESSION_ID]
# With no argument it uses the session that produced casestudies-by-agent-v2.

set -euo pipefail

SESSION="${1:-0886181b-ac49-474e-aaaf-0f8ecde70e1d}"
STORE="${COPILOT_SESSION_STORE:-$HOME/.copilot/session-store.db}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MAP="$HERE/cost-agents.tsv"

if [[ ! -f "$STORE" ]]; then
  echo "session store not found at $STORE" >&2
  exit 1
fi

# Turn the agent-id -> readable-name table into SQL inserts.
inserts=""
if [[ -f "$MAP" ]]; then
  while IFS=$'\t' read -r id name || [[ -n "${id:-}" ]]; do
    [[ -z "${id:-}" || "${id:0:1}" == "#" ]] && continue
    inserts+="INSERT INTO agent_map VALUES('${id}','${name//\'/\'\'}');"$'\n'
  done < "$MAP"
fi

sqlite3 -readonly "$STORE" <<SQL
.mode box
.headers on

CREATE TEMP TABLE agent_map(agent_id TEXT PRIMARY KEY, name TEXT);
${inserts}

SELECT
  COALESCE(m.name, COALESCE(u.agent_id,'lead (interactive)')) AS agent,
  COUNT(*)                                   AS reqs,
  SUM(u.input_tokens - u.cache_read_tokens)  AS fresh_in,
  SUM(u.cache_read_tokens)                   AS cache_read,
  SUM(u.cache_write_tokens)                  AS cache_write,
  SUM(u.output_tokens)                       AS output_tok,
  SUM(u.reasoning_tokens)                    AS reasoning,
  ROUND(SUM(u.total_nano_aiu)/1e9, 1)        AS aiu,
  ROUND(SUM(u.duration_ms)/60000.0, 1)       AS model_min
FROM assistant_usage_events u
LEFT JOIN agent_map m ON m.agent_id = u.agent_id
WHERE u.session_id = '${SESSION}'
GROUP BY 1
ORDER BY aiu DESC;

SELECT
  COUNT(*)                                               AS total_reqs,
  SUM(input_tokens) AS total_prompt_tok,
  SUM(cache_read_tokens) AS total_cached,
  SUM(output_tokens)                                     AS total_output_tok,
  ROUND(SUM(total_nano_aiu)/1e9, 1)                      AS total_aiu,
  ROUND(SUM(duration_ms)/60000.0, 1)                     AS total_model_min
FROM assistant_usage_events
WHERE session_id = '${SESSION}';
SQL
