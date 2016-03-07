#!/bin/sh
#PBS -l select=1:ncpus=1
#PBS -l place=pack
#PBS -j oe
#PBS -N main

cd $PBS_O_WORKDIR

./main.py
