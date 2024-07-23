# evmmax-bench-plot

This repository contains scripts for benchmarking [EIP-7747](https://github.com/ethereum/EIPs/pull/8743).

Benchmarks include:
* standalone EVM implementations of G1/G2 bls12381 point multiplication.
* arithmetic-only benchmarks for `ADDMODX`/`SUBMODX`/`MULMODX`/`SETMOD`.
* standalone EVM benchmarks for `ADDMODX`/`SUBMODX`/`MULMODX` which execute the target operation in a large loop, supplying random inputs.

Standalone EVM benchmarks can be executed against several Geth implementation presets:
* `mulmont384_asm` - x86_64 assembly for `MULMODX` at 321-384 bit widths
* `arith384_asm` - x86_64 assembly for `MULMODX`/`ADDMODX`/`SUBMODX` at 321-384 bit widths
* `fallback` - Go reference implementation (basis for the EIP cost model and Geth implementation)

Also provided is a script to generate graphs from the output of the benchmarks.

## Usage

### Setup

Download and build the submodule dependencies:
```
git submodule update --init
(cd go-ethereum-preset-arith-asm384 && make all)
(cd go-ethereum-preset-fallback && make all)
(cd go-ethereum-preset-mulmodx-asm384 && make all)
```

Setup virtual python environment:
```
python -m venv ./venv
. ./venv/bin/activate
pip install -r requirements.txt
```

### Collect Benchmarks

* Run standalone EVM g1/g2 mul benchmarks:
    1. `PRESET="fallback" GETH_EVM=./go-ethereum-preset-fallback/build/bin/evm python3 evmmax-bls12-381/test.py > benchmarks-results/geth-bls12381-benchmarks.csv`
    2. `PRESET="mulmont384_asm" GETH_EVM=./go-ethereum-preset-mulmodx-asm384/build/bin/evm python3 evmmax-bls12-381/test.py >> benchmarks-results/geth-bls12381-benchmarks.csv`
    3. `PRESET="arith384_asm" GETH_EVM=./go-ethereum-preset-arith-asm384/build/bin/evm python3 evmmax-bls12-381/test.py >> benchmarks-results/geth-bls12381-benchmarks.csv`
* standalone evm op benchmarks:
    1. `PRESET="fallback" GETH_EVM=./go-ethereum-preset-fallback/build/bin/evm python3 evm-benchmarks/evmmax_generator/generate.py > benchmarks-results/evm-op-benchmarks.csv`
    2. `PRESET="arith384_asm" GETH_EVM=./go-ethereum-preset-arith-asm384/build/bin/evm python3 evm-benchmarks/evmmax_generator/generate.py 6 6 >> benchmarks-results/evm-op-benchmarks.csv`
* gnark bls12381 g1/g2 mul (double-and-add) benchmarks (used in the graphs)
    1. `(cd gnark-crypto/ecc/bls12-381 && go test -bench=G1JacScalarMultiplication -run=^$`.  Benchmark results from my machine currently-hardcoded in the graph plotting script.
* arithmetic-only benchmarks for `ADDMODX`/`SUBMODX`/`MULMODX`:
    1. `(cd evmmax-arith/ && go test -bench=.) | python3 format-go-benchmark-output-as-csv.py > benchmarks-results/arith.csv`

### Plot Charts
```
python3 plot.py
```
