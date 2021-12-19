#!/bin/bash

echo "Warning: Please check requirements in 'requirements.txt' file"
echo "Usage: ./configure.sh"
read -r -p "Continue? [y/N] " response;
if [ -z "$(echo $response | grep -E "^([yY][eE][sS]|y)$")" ]; then
  exit;
fi;

cd build >/dev/null 2>&1 && cmake .. "$@";
