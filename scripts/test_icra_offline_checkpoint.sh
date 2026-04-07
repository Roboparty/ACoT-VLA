#!/usr/bin/env bash
set -euo pipefail

# Offline evaluation helper for an OpenPI checkpoint on ICRA GenieSim tasks.
# It starts a local policy server, runs genie_sim/scripts/run_icra_tasks.sh,
# and stores logs for later comparison.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_DIR="${REPO_ROOT}/checkpoints/acot_icra_simulation_challenge_reasoning_to_action/v03/20000"
POLICY_CONFIG="acot_icra_simulation_challenge_reasoning_to_action"
PORT="8000"
INFER_HOST=""
GENIE_ROOT="${REPO_ROOT}/genie_sim"
RUN_NAME="offline_eval_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/output/offline_eval/${RUN_NAME}"
KEEP_SERVER=false
READY_TIMEOUT="300"
ISAAC_PYTHON="${ISAAC_PYTHON:-/isaac-sim/python.sh}"

usage() {
  cat <<'EOF'
Usage: scripts/test_icra_offline_checkpoint.sh [options]

Options:
  --checkpoint-dir DIR   Checkpoint step directory (default: checkpoints/.../v03/20000)
  --policy-config NAME   Training config name (default: acot_icra_simulation_challenge_reasoning_to_action)
  --port PORT            Local policy server port (default: 8000)
  --infer-host HOST:PORT GenieSim infer host override (default: 127.0.0.1:<port>)
  --genie-root DIR       GenieSim repository root (default: <repo>/genie_sim)
  --log-dir DIR          Output log directory (default: output/offline_eval/<timestamp>)
  --ready-timeout SEC    Wait time for policy server health (default: 300)
  --isaac-python PATH    Isaac Sim python launcher (default: /isaac-sim/python.sh)
  --keep-server          Do not stop policy server after evaluation
  -h, --help             Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --policy-config)
      POLICY_CONFIG="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --infer-host)
      INFER_HOST="$2"
      shift 2
      ;;
    --genie-root)
      GENIE_ROOT="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --ready-timeout)
      READY_TIMEOUT="$2"
      shift 2
      ;;
    --isaac-python)
      ISAAC_PYTHON="$2"
      shift 2
      ;;
    --keep-server)
      KEEP_SERVER=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${INFER_HOST}" ]]; then
  INFER_HOST="127.0.0.1:${PORT}"
fi

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] Checkpoint directory not found: ${CHECKPOINT_DIR}"
  exit 1
fi

if [[ ! -d "${CHECKPOINT_DIR}/params" ]]; then
  echo "[ERROR] Missing params directory: ${CHECKPOINT_DIR}/params"
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/scripts/serve_policy.py" ]]; then
  echo "[ERROR] Missing serve script: ${REPO_ROOT}/scripts/serve_policy.py"
  exit 1
fi

if [[ ! -f "${GENIE_ROOT}/scripts/run_icra_tasks.sh" ]]; then
  echo "[ERROR] Missing genie runner: ${GENIE_ROOT}/scripts/run_icra_tasks.sh"
  exit 1
fi

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  echo "[ERROR] Isaac Sim python launcher not found or not executable: ${ISAAC_PYTHON}"
  echo "[ERROR] Pass --isaac-python /abs/path/to/python.sh"
  exit 1
fi

mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/policy_server.log"
EVAL_LOG="${LOG_DIR}/genie_eval.log"

echo "[INFO] Repo root      : ${REPO_ROOT}"
echo "[INFO] Checkpoint dir : ${CHECKPOINT_DIR}"
echo "[INFO] Policy config  : ${POLICY_CONFIG}"
echo "[INFO] Genie root     : ${GENIE_ROOT}"
echo "[INFO] Infer host     : ${INFER_HOST}"
echo "[INFO] Isaac python   : ${ISAAC_PYTHON}"
echo "[INFO] Log dir        : ${LOG_DIR}"
echo "[INFO] Ready timeout  : ${READY_TIMEOUT}s"

if [[ ! -f "${CHECKPOINT_DIR}/assets/norm_stats.json" ]] && [[ ! -f "${CHECKPOINT_DIR}/assets/all_tasks_merged/norm_stats.json" ]]; then
  echo "[WARN] norm_stats.json not found under checkpoint assets."
  echo "[WARN] Expected one of:"
  echo "       ${CHECKPOINT_DIR}/assets/norm_stats.json"
  echo "       ${CHECKPOINT_DIR}/assets/all_tasks_merged/norm_stats.json"
fi

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    if [[ "${KEEP_SERVER}" == true ]]; then
      echo "[INFO] Keeping policy server alive (PID=${SERVER_PID})"
    else
      echo "[INFO] Stopping policy server (PID=${SERVER_PID})"
      kill "${SERVER_PID}" >/dev/null 2>&1 || true
      wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT INT TERM

echo "[INFO] Starting local policy server..."
(
  cd "${REPO_ROOT}"
  uv run python scripts/serve_policy.py \
    --env G2SIM \
    --port "${PORT}" \
    policy:checkpoint \
    --policy.config "${POLICY_CONFIG}" \
    --policy.dir "${CHECKPOINT_DIR}"
) >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "[INFO] Policy server PID: ${SERVER_PID}"
echo "[INFO] Waiting for health endpoint http://${INFER_HOST}/healthz ..."

HEALTH_HOST="${INFER_HOST%:*}"
HEALTH_PORT="${INFER_HOST##*:}"

check_server_ready() {
  # 1) Ensure TCP port is open.
  if ! python - <<PY >/dev/null 2>&1
import socket
host = ${HEALTH_HOST@Q}
port = int(${HEALTH_PORT@Q})
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
ok = (s.connect_ex((host, port)) == 0)
s.close()
raise SystemExit(0 if ok else 1)
PY
  then
    return 1
  fi

  # 2) Health endpoint check (bypass proxy to avoid localhost proxy issues).
  curl --noproxy '*' -fsS "http://${INFER_HOST}/healthz" >/dev/null 2>&1
}

READY=false
for _ in $(seq 1 "${READY_TIMEOUT}"); do
  if check_server_ready; then
    READY=true
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] Policy server exited early. Last log lines:"
    tail -n 80 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 1
done

if [[ "${READY}" != true ]]; then
  echo "[ERROR] Policy server did not become ready within ${READY_TIMEOUT}s"
  echo "[ERROR] Last log lines:"
  tail -n 80 "${SERVER_LOG}" || true
  exit 1
fi

echo "[INFO] Server is ready. Running GenieSim offline benchmark..."
set +e
(
  cd "${GENIE_ROOT}"
  bash scripts/run_icra_tasks.sh --infer-host "${INFER_HOST}" --isaac-python "${ISAAC_PYTHON}"
) 2>&1 | tee "${EVAL_LOG}"
EVAL_STATUS=${PIPESTATUS[0]}
set -e

echo "[INFO] Evaluation exit code: ${EVAL_STATUS}"
echo "[INFO] Policy log: ${SERVER_LOG}"
echo "[INFO] Eval log  : ${EVAL_LOG}"
echo "[INFO] Benchmark dir: ${GENIE_ROOT}/output/benchmark"

if [[ ${EVAL_STATUS} -ne 0 ]]; then
  echo "[WARN] Offline benchmark reported failures. Check ${EVAL_LOG} for failed configs."
fi

exit ${EVAL_STATUS}
