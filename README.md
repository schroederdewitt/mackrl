# Fast MVC-style deep MARL framework

Includes implementations of algorithms:
- MACKRL
- CENTRAL-V

## Installation instructions

Build the Dockerfile:

```bash
$ ./build.sh
```

### StarCraft II

Set up StarCraft II. Download this specific version here:
(SC2.3.16.1)[http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.16.1.zip]
move to 
> coma/3rdparty/StarCraftII

(and unzip of course using password `iagreetotheeula`)

and then copy 
> /src/envs/starcraft2/maps

to the 

> 3rdparty/StarCraftII/Maps/Melee 
maps folder (which you will have to create first).

## Run an experiment 

Run one of the EXPERIMENTs from the folder `src/config/experiments`
on a specific GPU using some special PARAMETERS:
```
cd fastmarl/src
../run.sh <GPU> python3 main.py --exp_name=<EXPERIMENT> with <PARAMETERS>
```

Keep an eye on your docker containers, they will be named
`<USER>_fastmarl_GPU_<GPU>_<RANDOM>`:
```
docker ps
```

If you do not want them anymore, kill a container named `NAME` with
```
docker kill <NAME>
docker rm <NAME>
```

If you want to get rid of *all* your containers, execute 
```
fastmarl/kill.sh
```

## Run SC2 baselines

Run 

> exp_scripts/coma_baselines/run.sh <Number of runs per scenario, e.g. 5>

Results are automatically logged to both tensorboard

> tensorboard --logdir=./results/tb_logs

and, if MongoDB has been set up (see main.py for config details), a database will be created.

If MongoDB is not available, then Sacred will produce output files under

> ./results/sacred
