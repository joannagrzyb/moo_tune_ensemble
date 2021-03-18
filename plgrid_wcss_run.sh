#!/bin/bash
#PBS -q plgrid
#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=2:mem=2048MB

# wejscie do katalogu, z ktorego zostalo wstawione zadania
cd $PBS_O_WORKDIR

# instalowanie potrzebnych bibliotek
pip3 install -r requirements.txt --user
pip3 install -U stream-learn

# uruchom program
# z PRZEKIEROWANIEM ZAPISYWANIA WYNIKOW -- BARDZO WAZNE
python3 experiment1_9higher_part1 >& wynik.txt
