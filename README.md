# evmmax-bench-plot

This repository contains scripts for benchmarking [EVMMAX](http://github.com/ethereum/EIPs)  Geth implementation arithmetic and EVM overhead.

## Usage

Download and build the submodule dependencies:
```
git submodule update --init --recursive --depth=1
(cd evm-benchmarks-asm384/evmmax_generator/go-ethereum && make all)
(cd evm-benchmarks-le/evmmax_generator/go-ethereum && make all)
(cd evm-benchmarks-eip-5843/evmmax_generator/go-ethereum && make all)
```

run benchmarks and store the raw data to files in `benchmarks-results`:
```
./collect-benchmarks.sh
```

Create benchmark charts under `charts`:
```
python3 plot.py
```
