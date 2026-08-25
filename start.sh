#!/usr/bin/env bash
# Startet die ACE-Step Gradio-UI: killt vorher den Zielport, startet den Server
# und öffnet danach automatisch den Browser.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# PORT/SERVER_NAME aus .env übernehmen (gleiche Logik wie start_gradio_ui.sh),
# damit wir den richtigen Port killen, bevor der eigentliche Server startet.
PORT="${PORT:-}"
SERVER_NAME="${SERVER_NAME:-}"
if [[ -f .env ]]; then
    while IFS='=' read -r key value; do
        key="$(echo "$key" | xargs)"
        value="$(echo "$value" | xargs)"
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        case "$key" in
            PORT) [[ -n "$value" ]] && PORT="$value" ;;
            SERVER_NAME) [[ -n "$value" ]] && SERVER_NAME="$value" ;;
        esac
    done < .env
fi
: "${PORT:=7860}"
: "${SERVER_NAME:=127.0.0.1}"

# Fix: /usr/local/cuda/lib64 (System-CUDA-12.9) in LD_LIBRARY_PATH kollidiert
# mit den von Torch mitgelieferten cuBLAS-Libs im venv (-> CUBLAS_STATUS_INVALID_VALUE).
# Diesen Pfad entfernen, sonstige Einträge behalten.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^/usr/local/cuda' | paste -sd: -)"
    export LD_LIBRARY_PATH
fi

# RTX 5060 Ti (Blackwell, schneller als die 4060 Ti) deterministisch als GPU 0
# wählen: CUDA sortiert Geräte sonst nicht nach PCI-Bus-Reihenfolge.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "Prüfe Port $PORT..."
# lsof erfasst den Prozess hier manchmal nicht zuverlässig (z.B. bei uv-Wrapper-
# Prozessen) -- zusätzlich über ss/fuser nach dem tatsächlichen Listener suchen.
PIDS="$(lsof -ti tcp:"$PORT" 2>/dev/null || true)"
SS_PID="$(ss -ltnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p"$" {print $0}' | grep -oP 'pid=\K[0-9]+' || true)"
FUSER_PIDS="$(fuser -n tcp "$PORT" 2>/dev/null || true)"
PIDS="$(printf '%s\n%s\n%s\n' "$PIDS" "$SS_PID" "$FUSER_PIDS" | grep -E '^[0-9]+$' | sort -u)"
if [[ -n "$PIDS" ]]; then
    echo "Port $PORT ist belegt, beende Prozess(e): $(echo "$PIDS" | tr '\n' ' ')"
    kill -9 $PIDS 2>/dev/null || true
    sleep 1
else
    echo "Port $PORT ist frei."
fi

URL="http://${SERVER_NAME}:${PORT}"

open_browser() {
    # Warten, bis der Server tatsächlich antwortet, dann Browser öffnen.
    for _ in $(seq 1 120); do
        if curl -sf -o /dev/null "$URL"; then
            break
        fi
        sleep 1
    done

    if command -v xdg-open &>/dev/null; then
        xdg-open "$URL" &>/dev/null &
    elif command -v open &>/dev/null; then
        open "$URL" &>/dev/null &
    else
        echo "Konnte keinen Browser-Opener finden. Bitte manuell öffnen: $URL"
    fi
}

open_browser &

export PORT SERVER_NAME
exec ./start_gradio_ui.sh
