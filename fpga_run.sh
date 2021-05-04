#!/bin/bash

if [[ $(sudo fpga-describe-local-image -S 0 -H) != *"$(cat fpga_afi.txt)"* ]]; then
  echo "Loading AFI..."
  sudo fpga-load-local-image -S 0 -I $(cat fpga_afi.txt) > /dev/null
fi

args="${@:2}"
sudo $(which python3) $1 $args
