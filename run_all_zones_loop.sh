#!/bin/bash

for ITER in {1..30}
do
    python main.py configs/baseline_zone1_tolerance"$1"_week.yaml $ITER
    python main.py configs/baseline_zone2_tolerance"$1"_week.yaml $ITER
    python main.py configs/baseline_zones_tolerance"$1"_week.yaml $ITER
    echo "iteration $ITER done"
done
