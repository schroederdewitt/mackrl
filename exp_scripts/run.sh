#!/bin/bash

target=$1
config=$2
with_args=$3
with_args_local=$4
tag=$5
reps=$6

filename=$(basename -- "$0")
extension="${filename##*.}"
filename="${filename%.*}"
timestamp=`date "+%d_%m_%y-%H_%M_%S"`
name=${tag}__${filename}__${timestamp}

# set up experiment summary file
spath="${FASTMARL_MACKRL4NEURIPS_PATH}/exp_summaries"
mkdir -p $spath
sfilepath="$spath/${tag}.summary"
if [ -f $sfilepath ]; then
    echo "FATAL: experiment summary already exists. Overwrite?"
    read -p "Are you sure? " -n 1 -r
    echo    # (optional) move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        echo "Aborting run."
        exit
    else
        echo "Deleting old experiment summary..."
        rm -rf $sfilepath
    fi
fi

# set fastmarl results path
if [ -z "$FASTMARL_MACKRL4NEURIPS_RESULTS_PATH" ]; then
    RESULTS_PATH="${FASTMARL_MACKRL4NEURIPS_PATH}/results"
    mkdir -p $RESULTS_PATH
else
    RESULTS_PATH=$FASTMARL_MACKRL4NEURIPS_RESULTS_PATH
fi

if [ $target == "local" ] ; then

    echo "launching locally on "`hostname`"..."
    export PYTHONPATH=$PYTHONPATH:/fastmarl/src

    # enter general run information into summary file
    echo "hostname: "`hostname`" "
    echo "fastmarl_path: ${FASTMARL_MACKRL4NEURIPS_PATH}" >> $sfilepath
    echo "python_path: ${PYTHONPATH}" >> $sfilepath
    echo "results_path: ${RESULTS_PATH}" >> $sfilepath

    #if [ `hostname` == "octavia" | `hostname` ==  ]

    n_gpus=`nvidia-smi -L | wc -l`
    n_upper=`expr $n_gpus - 1`

    for i in $(seq 1 $reps); do
        gpu_id=`shuf -i0-${n_upper} -n1`
        echo "Starting repeat number $i on GPU $gpu_id"
        HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
        echo "NV_GPU=${gpu_id} ${FASTMARL_MACKRL4NEURIPS_PATH}/docker.sh ${HASH} python3 /fastmarl/src/main.py --exp_name=${config} with ${with_args} ${with_args_local} name=${name}__repeat${i} &"
        NV_GPU=${gpu_id} ${FASTMARL_MACKRL4NEURIPS_PATH}/docker.sh ${HASH} python3 /fastmarl/src/main.py --exp_name=${config} with ${with_args} ${with_args_local} name=${name}__repeat${i} &
        echo "repeat: ${i}"
        echo "    name: ${name}__repeat${i}" >> $sfilepath
        echo "    gpu: ${gpu_id}" >> $sfilepath
        echo "    docker_hash: ${HASH}" >> $sfilepath
        sleep 5s
    done

else

    echo "Target ${target} not supported!"
    exit

fi

echo "Finished experiment launch on "`hostname`"."
