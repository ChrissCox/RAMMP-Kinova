#!/usr/bin/env zsh
source ~/.zshrc 2>/dev/null
cd ~/RAMMP-Kinova/ros2_ws || exit 1
source install/setup.zsh
# Kill the previous stack COMPLETELY before launching. Every ROS node has
# --ros-args on its cmdline — that is the only pattern that catches them
# all (killing by node names missed the planner, whose binary is just
# 'planner'; a half-dead planner once answered a fresh session with the
# old session's held-bottle state). The [-] bracket keeps pkill from
# matching THIS script's own cmdline and killing the ssh session.
pkill -f 'mujoco_bringup[.]launch' 2>/dev/null && sleep 6
pkill -9 -f 'ros[-]args' 2>/dev/null
pkill -9 -f 'xvfb[-]run' 2>/dev/null
sleep 2
left=$(pgrep -f 'ros[-]args' | wc -l)
if [ "$left" -ne 0 ]; then
  echo "REFUSING to launch: $left stale ROS processes survived the kill"
  pgrep -af 'ros[-]args'
  exit 1
fi
nohup ros2 launch mujoco_sim mujoco_bringup.launch.py > /tmp/bringup.log 2>&1 &
disown
echo "LAUNCHED pid $!"
