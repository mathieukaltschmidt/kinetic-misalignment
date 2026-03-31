#!/bin/bash

for i in 0 50 100 150 ; do
nmdir="$(printf "i%03d" "$i")"
echo "$nmdir"
mkdir -p "$nmdir"
cp simgen_exe.py simgen_mpcdf.py measfile.dat analysis.py inco.py "$nmdir"
cd "$nmdir"
ls
pwd
python3 simgen_exe.py $i
sbatch "$nmdir"
cd ..
done
