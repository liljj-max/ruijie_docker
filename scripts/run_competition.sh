#!/usr/bin/env bash
set -eo pipefail

baseline_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
cd "$baseline_root"

python3 -m competition_client.perception &
perception_pid=$!

cleanup() {
  kill "$perception_pid" 2>/dev/null || true
  wait "$perception_pid" 2>/dev/null || true
}
trap cleanup EXIT

python3 -m competition_client.competition_client
