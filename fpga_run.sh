#!/bin/bash

AGFI=agfi-023ca8e2707196e49

if [[ $(sudo fpga-describe-local-image -S 0 -H) != *"$AGFI"* ]]; then
  echo "Loading AFI..."
  sudo fpga-load-local-image -S 0 -I $AGFI > /dev/null
fi

args="${@:2}"
sudo $(which python3) $1 $args
