#!/bin/bash
echo "Killing all docker containers with a name  matching ${USER}_fastmarl_mackrl4neurips_GPU_*"
docker rm $(docker stop $(docker ps -a -q --filter name=${USER}_fastmarl_mackrl4neurips_GPU_ --format="{{.ID}}"))

