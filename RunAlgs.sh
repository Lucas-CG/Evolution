#!/usr/bin/env bash

runs=25
runList=(1 2 3 4 5 6 7 8 9 10)
commands=()

for i in $runList; do
        commands+="python3 GeneticAlgorithm.py & > run$i.txt"
done

echo $commands
echo "${commands[@]}"

parallel < $commands
