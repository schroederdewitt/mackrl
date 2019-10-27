#!/bin/bash

if [ -z "$3" ]; then
    echo "target 'local' selected automatically."
    target="local"
    tag=$1
    reps=$2
else
   target=$1
   tag=$2
   reps=$3
fi 

config="mackrl/neurips_2s3z"
with_args="t_max=5000000  use_hdf_logger=False save_episode_samples=False  save_model=True save_model_interval=100000 use_tensorboard=False fix_partition_size=3 "
with_args_local=""

${FASTMARL_PATH}/exp_scripts/run.sh "${target}" "${config}" "${with_args}" "${with_args_local}" "${tag}" ${reps}
