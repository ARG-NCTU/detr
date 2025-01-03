#!/usr/bin/env bash
#
# Typical usage: ./join.bash subt
#

BASH_OPTION=bash

IMG=argnctu/detr:jetson-orin

xhost +
containerid=$(docker ps -aqf "ancestor=${IMG}") && echo $containerid
docker exec -it \
    --privileged \
    -e LINES="$(tput lines)" \
    ${containerid} \
    $BASH_OPTION
xhost -