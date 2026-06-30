#!/usr/bin/env bash
#
# One-shot setup for the RAMMP-Kinova workspace on the Jetson.
# Clones ros2_kortex (Humble), imports its dependency repos, resolves rosdeps,
# and builds the whole colcon workspace (upstream driver + our adl_primitives).
#
# REQUIREMENTS: Ubuntu 22.04 + ROS 2 Humble installed at /opt/ros/humble.
# This script is LINUX ONLY. Do not run it on Windows/macOS.
#
# Usage:
#   bash scripts/setup_ros2_kortex.sh
#   COLCON_WORKERS=2 bash scripts/setup_ros2_kortex.sh   # fewer parallel jobs (less RAM)

set -euo pipefail

: "${ROS_DISTRO:=humble}"
: "${COLCON_WORKERS:=3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS="${REPO_ROOT}/ros2_ws"
SRC="${WS}/src"

echo "==> Repo root : ${REPO_ROOT}"
echo "==> Workspace : ${WS}"
echo "==> ROS_DISTRO: ${ROS_DISTRO}"

if [ "${ROS_DISTRO}" != "humble" ]; then
  echo "WARNING: this project targets ROS 2 Humble, but ROS_DISTRO=${ROS_DISTRO}" >&2
fi

if [ ! -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  echo "ERROR: /opt/ros/${ROS_DISTRO}/setup.bash not found. Install ROS 2 ${ROS_DISTRO} first." >&2
  exit 1
fi

# ROS setup scripts reference unbound vars; relax 'nounset' while sourcing.
set +u
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

echo "==> Installing build tooling (sudo)..."
sudo apt-get update
sudo apt-get install -y \
  git \
  python3-colcon-common-extensions \
  python3-vcstool \
  python3-rosdep

mkdir -p "${SRC}"

# 1) Clone ros2_kortex (humble branch) if not already present.
if [ ! -d "${SRC}/ros2_kortex" ]; then
  echo "==> Cloning ros2_kortex (${ROS_DISTRO})..."
  git clone -b "${ROS_DISTRO}" https://github.com/Kinovarobotics/ros2_kortex.git "${SRC}/ros2_kortex"
else
  echo "==> ros2_kortex already present, skipping clone."
fi

# 2) Import dependency repos declared by ros2_kortex.
echo "==> Importing dependency repositories with vcs..."
vcs import "${SRC}" --skip-existing --input "${SRC}/ros2_kortex/ros2_kortex.${ROS_DISTRO}.repos"
NOT_RELEASED="${SRC}/ros2_kortex/ros2_kortex-not-released.${ROS_DISTRO}.repos"
if [ -f "${NOT_RELEASED}" ]; then
  vcs import "${SRC}" --skip-existing --input "${NOT_RELEASED}"
fi

# 3) Resolve dependencies with rosdep.
echo "==> Resolving rosdeps..."
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init || true
fi
rosdep update
rosdep install --ignore-src --from-paths "${SRC}" -y -r

# 4) Build.
echo "==> Building (this can take a while on the Jetson)..."
cd "${WS}"
colcon build \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --parallel-workers "${COLCON_WORKERS}"

echo
echo "==> Done."
echo "    Source the overlay before using ROS:"
echo "      source /opt/ros/${ROS_DISTRO}/setup.bash"
echo "      source ${WS}/install/setup.bash"
