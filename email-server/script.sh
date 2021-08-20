#!/bin/bash

export HOME='/home/amathur'
# source virtualenv here
timeout 360s python /mnt/volume_nyc3_01/script.py $1
exit 0
