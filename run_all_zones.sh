#!/bin/bash

python main.py configs/baseline_zone1_tolerance"$1"_week.yaml
python main.py configs/baseline_zone2_tolerance"$1"_week.yaml
python main.py configs/baseline_zones_tolerance"$1"_week.yaml
