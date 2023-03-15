#!/bin/bash
for var in {1..5}
do
   python src/run_best_model.py
done

for var in {1..5}
do
   python main.py --model random_forest
done