#!/bin/bash
sudo apt-get update && sudo apt-get install locales -y

sudo locale-gen en_US.UTF-8

{
echo "LANG=en_US.UTF-8"
echo "LC_COLLATE=en_US.UTF-8"
echo "LC_CTYPE=en_US.UTF-8"
echo "LC_MESSAGES=en_US.UTF-8"
echo "LC_MONETARY=en_US.UTF-8"
echo "LC_NUMERIC=en_US.UTF-8"
echo "LC_TIME=en_US.UTF-8"
echo "LC_ALL=en_US.UTF-8"
} | sudo tee /etc/default/locale

