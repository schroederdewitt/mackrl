#!/bin/bash
HASH=$1
name=${USER}_fastmarl_mackrl4neurips_GPU_${GPU}_${HASH}

echo "Launching container named '${name}' on GPU '${GPU}'"
# Launches a docker container using our image, and runs the provided command

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -z "$FASTMARL_MACKRL4NEURIPS_RESULTS_PATH" ]; then
    RESULTS_PATH="${FASTMARL_MACKRL4NEURIPS_PATH}/results"
    mkdir -p $RESULTS_PATH
else
    RESULTS_PATH=$FASTMARL_MACKRL4NEURIPS_RESULTS_PATHs
fi

echo "HASH: ${HASH}"
echo "REST: ${@:1}"

echo "RESULTS_PATH: ${RESULTS_PATH}"
${cmd} run -d --rm \
    --name $name \
    --security-opt="apparmor=unconfined" --cap-add=SYS_PTRACE \
    --net host \
    --user $(id -u) \
    -v $SCRIPT_PATH:/fastmarl \
    -v $RESULTS_PATH:/fastmarl/results \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=unix$DISPLAY \
    -t fastmarl/mackrl4neurips \
    ${@:2}
