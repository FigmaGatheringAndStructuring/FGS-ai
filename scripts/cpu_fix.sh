#!/bin/sh
# LXC 컨테이너에서 PyTorch(aarch64)가 import torch 중 cpuinfo에서 segfault 나는 문제 우회.
# sysfs의 CPU 목록을 /proc/cpuinfo와 같은 0..N-1 로 덮어씌운다.
# 컨테이너가 재시작되면 풀리므로 서버 시작 전에 매번 실행한다. 여러 번 실행해도 안전하다.
set -e

N=$(grep -c ^processor /proc/cpuinfo)
echo "0-$((N - 1))" > /root/cpu-fix

for f in online possible present; do
  target=/sys/devices/system/cpu/$f
  mountpoint -q "$target" || mount --bind /root/cpu-fix "$target"
done

echo "cpu fix: $(cat /sys/devices/system/cpu/online)"
