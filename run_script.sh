#!/bin/bash

if [[ $(sudo fpga-describe-local-image -S 0 -H) != *"$(cat afi.txt)"* ]]; then
  echo "Loading AFI..."
  ./load_image.sh > /dev/null
fi

args="${@:2}"
sudo $(which python3) $1 $args
