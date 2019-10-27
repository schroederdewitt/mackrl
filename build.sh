echo 'Building Dockerfile with image name fastmarl`'
if [ ! -f .dockerignore ]; then
    cp .dockerignore.local .dockerignore
    docker build -t fastmarl/mackrl4neurips . &
    sleep 20
    rm .dockerignore
else
   echo ".dockerignore file exists - please remove manually after verifying that no other build process is currently in progress."
fi
