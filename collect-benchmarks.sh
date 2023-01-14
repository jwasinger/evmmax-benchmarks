#! /usr/bin/env bash

benchmark_dir="benchmarks-results"

echo "running eip5843 go benchmarks"
(cd evmmax-arith-eip5843/ && go test -timeout=4h -run=^$ -bench=Op/"non-unrolled*") > $benchmark_dir/go-arith-benchmarks-eip5843.txt
echo " running little endian go benchmarks"
(cd evmmax-arith-little-endian/ && go test -timeout=4h -run=^$ -bench=Op/"non-unrolled*") > $benchmark_dir/go-arith-benchmarks-le.txt
echo "running asm384 go benchmarks"
(cd evmmax-arith-little-endian/ && go test -timeout=4h -run=^$ -bench=Op/"asm384_.*384") > $benchmark_dir/go-arith-benchmarks-asm384.txt

echo "running asm384 evm benchmarks"
(cd evm-benchmarks-asm384/evmmax_generator && python3 benchmark.py MULMONTX 6 6) > $benchmark_dir/geth-evm-asm384.csv
(cd evm-benchmarks-asm384/evmmax_generator && python3 benchmark.py ADDMODX 6 6) >> $benchmark_dir/geth-evm-asm384.csv
(cd evm-benchmarks-asm384/evmmax_generator && python3 benchmark.py SUBMODX 6 6) >> $benchmark_dir/geth-evm-asm384.csv

echo "running little-endian benchmarks"
(cd evm-benchmarks-le/evmmax_generator && python3 benchmark.py) > $benchmark_dir/geth-evm-little-endian.csv
echo "running big-endian EVM benchmarks"
(cd evm-benchmarks-eip-5843/evmmax_generator && python3 benchmark.py) > $benchmark_dir/geth-evm-eip-5843.csv
