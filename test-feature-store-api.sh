#!/usr/bin/env bash
# Functional smoke tests for xyz-feature-store-service (port 8115)
# Usage: ./test-feature-store-api.sh [BASE_URL]
# Exit code: 0 = all passed, 1 = one or more failures
#
# 2026-09-11: counters no longer use ((X++)) (returns status 1 when X=0 → set -e
# aborted after the first PASS); applied doc/TEST-REPORT.md fixes R1–R7.

set -euo pipefail

BASE="${1:-http://localhost:8115}"
PASS=0; FAIL=0

AUTH_HEADERS=(
  -H "x-verified-tenant-id: 00000000-0000-0000-0000-000000000001"
  -H "x-verified-user-id: test-user-001"
  -H "x-verified-roles: ADMIN"
)
TENANT_ID="00000000-0000-0000-0000-000000000001"
# Unique per run: online features are cached in Redis by entity-id pair (24h TTL),
# so reusing ids would return a previous run's vector.
RUN_ID="$(date +%s)_${RANDOM}"

pass() { echo "  PASS: $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL: $1 — $2"; FAIL=$((FAIL+1)); }

check_status() {
  local label="$1" expected="$2"
  shift 2
  local actual
  actual=$(curl -s -o /dev/null -w "%{http_code}" "$@")
  if [[ "$actual" == "$expected" ]]; then pass "$label"; else fail "$label" "expected $expected, got $actual"; fi
}

check_json_field() {
  local label="$1" expr="$2"
  shift 2
  local body
  body=$(curl -s "$@")
  if echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); assert $expr" 2>/dev/null; then
    pass "$label"
  else
    fail "$label" "assertion '$expr' failed in: $(echo "$body" | head -c 200)"
  fi
}

echo "=== xyz-feature-store-service functional tests: $BASE ==="
echo ""

# ── TC-HEALTH ─────────────────────────────────────────────────────────────────
echo "--- TC-HEALTH ---"
check_status "TC-HEALTH-01: GET /api/v1/health returns 2xx"           200 "$BASE/api/v1/health"
check_json_field "TC-HEALTH-02: health has status field" "'status' in d" "$BASE/api/v1/health"

# ── TC-AUTH ───────────────────────────────────────────────────────────────────
echo "--- TC-AUTH ---"
check_status "TC-AUTH-01: no headers → 401" 401 \
  "$BASE/api/v1/features/e1/e2?tenant_id=$TENANT_ID"
check_status "TC-AUTH-02: missing user-id → 401" 401 \
  -H "x-verified-tenant-id: $TENANT_ID" -H "x-verified-roles: ADMIN" \
  "$BASE/api/v1/features/e1/e2?tenant_id=$TENANT_ID"
check_status "TC-AUTH-03: valid headers → not 401" 200 \
  "${AUTH_HEADERS[@]}" \
  "$BASE/api/v1/features/ent_001/ent_002?tenant_id=$TENANT_ID&name1=Alice&name2=Alise"

# ── TC-FEATURES ───────────────────────────────────────────────────────────────
echo "--- TC-FEATURES ---"
check_json_field "TC-FEAT-01: GET /features/{id1}/{id2} returns features dict" "'features' in d" \
  "${AUTH_HEADERS[@]}" \
  "$BASE/api/v1/features/ent_001/ent_002?tenant_id=$TENANT_ID&name1=Alice+Johnson&name2=Alise+Jonson"
check_json_field "TC-FEAT-02: features dict has exactly 50 features" "len(d.get('features',{})) == 50" \
  "${AUTH_HEADERS[@]}" \
  "$BASE/api/v1/features/ent_001/ent_002?tenant_id=$TENANT_ID&name1=Alice+Johnson&name2=Alise+Jonson"
check_json_field "TC-FEAT-03: all feature values in [0,1]" \
  "all(0.0 <= v <= 1.0 for v in d.get('features',{}).values())" \
  "${AUTH_HEADERS[@]}" \
  "$BASE/api/v1/features/ent_001/ent_002?tenant_id=$TENANT_ID&name1=Alice&name2=Alice"
# R1: ids not used by any other test (and unique per run) so the vector isn't served from cache
check_json_field "TC-FEAT-04: identical names → ss_levenshtein_name=1.0" \
  "d.get('features',{}).get('ss_levenshtein_name',0) == 1.0" \
  "${AUTH_HEADERS[@]}" \
  "$BASE/api/v1/features/ent_same_a_${RUN_ID}/ent_same_b_${RUN_ID}?tenant_id=$TENANT_ID&name1=John+Smith&name2=John+Smith"

# ── TC-BATCH ──────────────────────────────────────────────────────────────────
echo "--- TC-BATCH ---"
check_status "TC-BATCH-01: POST /features/batch valid request → 200" 200 \
  -X POST "${AUTH_HEADERS[@]}" \
  -H "Content-Type: application/json" \
  -d "{\"pairs\":[{\"entity_id_1\":\"e1\",\"entity_id_2\":\"e2\"}],\"tenant_id\":\"$TENANT_ID\"}" \
  "$BASE/api/v1/features/batch"
check_status "TC-BATCH-02: POST /features/batch no auth → 401" 401 \
  -X POST -H "Content-Type: application/json" \
  -d "{\"pairs\":[{\"entity_id_1\":\"e1\",\"entity_id_2\":\"e2\"}],\"tenant_id\":\"$TENANT_ID\"}" \
  "$BASE/api/v1/features/batch"
check_status "TC-BATCH-03: POST /features/batch >100 pairs → 422" 422 \
  -X POST "${AUTH_HEADERS[@]}" \
  -H "Content-Type: application/json" \
  -d "{\"pairs\":[$(python3 -c "import json; print(','.join(json.dumps({'entity_id_1':f'e{i}','entity_id_2':f'e{i+100}'}) for i in range(101)))")],\"tenant_id\":\"$TENANT_ID\"}" \
  "$BASE/api/v1/features/batch"

# ── TC-PUSH ────────────────────────────────────────────────────────────────────
echo "--- TC-PUSH ---"
# R2: FeaturePushRequest schema is {tenant_id, entity_type, fields}
check_status "TC-PUSH-01: PUT /features/{id} no auth → 401" 401 \
  -X PUT -H "Content-Type: application/json" \
  -d "{\"entity_type\":\"customer\",\"fields\":{\"name\":\"Alice\"},\"tenant_id\":\"$TENANT_ID\"}" \
  "$BASE/api/v1/features/e1"
check_status "TC-PUSH-02: PUT /features/{id} with auth → 2xx" 200 \
  -X PUT "${AUTH_HEADERS[@]}" \
  -H "Content-Type: application/json" \
  -d "{\"entity_type\":\"customer\",\"fields\":{\"name\":\"Alice\"},\"tenant_id\":\"$TENANT_ID\"}" \
  "$BASE/api/v1/features/e1"

# ── TC-DEFINITIONS ──────────────────────────────────────────────────────────────
echo "--- TC-DEFINITIONS ---"
# R3: response is {"total": 50, "version": ..., "features": [...]}
check_status "TC-DEF-01: GET /features/definitions no auth → 401" 401 \
  "$BASE/api/v1/features/definitions"
check_json_field "TC-DEF-02: GET /features/definitions returns list" \
  "isinstance(d, list) or 'features' in d or 'definitions' in d" \
  "${AUTH_HEADERS[@]}" "$BASE/api/v1/features/definitions"
check_json_field "TC-DEF-03: definitions count = 50" \
  "len(d if isinstance(d, list) else d.get('features', d.get('definitions', []))) == 50" \
  "${AUTH_HEADERS[@]}" "$BASE/api/v1/features/definitions"

# ── TC-OFFLINE ──────────────────────────────────────────────────────────────────
echo "--- TC-OFFLINE ---"
# R4: OfflineFeatureRequest requires as_of_timestamp
check_status "TC-OFF-01: POST /features/offline no auth → 401" 401 \
  -X POST -H "Content-Type: application/json" \
  -d "{\"tenant_id\":\"$TENANT_ID\",\"entity_pairs\":[[\"e1\",\"e2\"]],\"as_of_timestamp\":\"2026-01-01T00:00:00Z\"}" \
  "$BASE/api/v1/features/offline"
check_status "TC-OFF-02: POST /features/offline with auth → 200" 200 \
  -X POST "${AUTH_HEADERS[@]}" \
  -H "Content-Type: application/json" \
  -d "{\"tenant_id\":\"$TENANT_ID\",\"entity_pairs\":[[\"e1\",\"e2\"]],\"as_of_timestamp\":\"2026-01-01T00:00:00Z\"}" \
  "$BASE/api/v1/features/offline"

# ── TC-MATERIALIZE ──────────────────────────────────────────────────────────────
echo "--- TC-MATERIALIZE ---"
check_status "TC-MAT-01: POST /materialize no auth → 401" 401 \
  -X POST -H "Content-Type: application/json" \
  -d "{\"tenant_id\":\"$TENANT_ID\"}" \
  "$BASE/api/v1/materialize"
# R5: capture the real job_id from the POST so TC-MAT-03 queries a valid UUID
MAT_RESP=$(curl -s -w "\n%{http_code}" -X POST "${AUTH_HEADERS[@]}" \
  -H "Content-Type: application/json" \
  -d "{\"tenant_id\":\"$TENANT_ID\"}" \
  "$BASE/api/v1/materialize")
MAT_CODE=$(echo "$MAT_RESP" | tail -n 1)
MAT_BODY=$(echo "$MAT_RESP" | sed '$d')
if [[ "$MAT_CODE" == "202" ]]; then
  pass "TC-MAT-02: POST /materialize with auth → 202"
else
  fail "TC-MAT-02: POST /materialize with auth → 202" "expected 202, got $MAT_CODE"
fi
JOB_ID=$(echo "$MAT_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null || true)
check_json_field "TC-MAT-03: GET /materialize/{id} returns job status" "'status' in d or 'job_id' in d" \
  "${AUTH_HEADERS[@]}" "$BASE/api/v1/materialize/${JOB_ID:-00000000-0000-0000-0000-000000000000}"

# ── TC-DRIFT ──────────────────────────────────────────────────────────────────
echo "--- TC-DRIFT ---"
check_status "TC-DRIFT-01: GET /drift no auth → 401" 401 \
  "$BASE/api/v1/drift"
# R6: endpoint returns a list of reports ([] when none computed yet)
check_json_field "TC-DRIFT-02: GET /drift with auth returns report" \
  "isinstance(d, list) or 'drift_detected' in d or 'features' in d or 'tenant_id' in d" \
  "${AUTH_HEADERS[@]}" "$BASE/api/v1/drift"

# ── TC-STATS ──────────────────────────────────────────────────────────────────
echo "--- TC-STATS ---"
# R7: response is {"online_store": {...}, "service_version": ..., "timestamp": ...}
check_json_field "TC-STATS-01: GET /stats returns cache info" \
  "'online_store' in d or 'total_features' in d or 'cache' in d or 'status' in d" \
  "${AUTH_HEADERS[@]}" "$BASE/api/v1/stats"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[[ "$FAIL" -eq 0 ]] || exit 1
