#! /usr/bin/env bash

benchmark_dir="benchmarks-results"

echo "running go/asm arithmetic benchmarks"
#(cd mont-arith && go test -run=^$ -bench=. | python3 ../format-geth-benchmark-output-as-csv.py) > $benchmark_dir/go-arith-benchmarks.csv
(cd mont-arith && go test -run=^$ -bench=AddModNonUnrolled | python3 ../format-geth-benchmark-output-as-csv.py) >> $benchmark_dir/go-arith-benchmarks.csv
(cd mont-arith && go test -run=^$ -bench=SubModNonUnrolled | python3 ../format-geth-benchmark-output-as-csv.py) >> $benchmark_dir/go-arith-benchmarks.csv
