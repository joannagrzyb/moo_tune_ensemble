#!/bin/bash
#PBS -q plgrid
#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=2:mem=2048MB
#PBS -A plgjoannagrzyb2021a

# wejscie do katalogu, z ktorego zostalo wstawione zadania
cd $PBS_O_WORKDIR
ls

# instalowanie potrzebnych bibliotek
module add /usr/local/Modules/python/3.6.8-gcc7.3.0
pip install -r requirements.txt --user
pip install -U stream-learn

# uruchom program
# z PRZEKIEROWANIEM ZAPISYWANIA WYNIKOW -- BARDZO WAZNE
python3 experiment1_9higher_part1.py > wynik.txt
