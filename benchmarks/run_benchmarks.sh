#!/bin/bash 

filename="benchmark_results.json" 
directory="results" 

echo "writing to file: ${directory}/${filename}" 
mkdir ${directory}
../install/Release/bin/benchmarks --benchmark_out="${directory}/${filename}" --benchmark_counters_tabular=true --benchmark_out_format=json 
