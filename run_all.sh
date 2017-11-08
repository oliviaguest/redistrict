#!/bin/bash
for i in `seq 1 56`;
do
  python run.py $i &
done
