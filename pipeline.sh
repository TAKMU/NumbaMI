#!/bin/bash
var_path=$PWD  
python3 preprocess_mi.py --input data/example.tsv --output data/simplified.csv --separation "t" --axis 1
Rscript R_discretize.R --input data/simplified.csv --output data/discretized.csv --working_dir $var_path
python3 generate_matrix.py --input data/discretized.csv --output data/mi_matrix_1.csv
