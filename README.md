# irf_simulation

```bash
$ python script_test_bench.py
```


## dependencies
- python=3.10
- numpy==1.26.3
- torch==2.1.2

## test reuslts
- test1, nboot = 399
```bash
# python
Using nboot: 399
Loaded inputs
original script took 1.5724365711212158 seconds
Using device: cuda
gpu array script took 1.5913772583007812 seconds

# matlab
Using nboot:
   399

Original Script, Compile on Execution
Elapsed time is 1.335735 seconds.
GPU Script, Compile on Execution
Elapsed time is 4.970961 seconds.

```

- test2, nboot = 10_000
```bash
Using nboot: 10000
Loaded inputs
original script took 39.14391088485718 seconds
Using device: cuda
gpu array script took 23.954208850860596 seconds

# matlab
Using nboot:
       10000

Original Script, Compile on Execution
Elapsed time is 36.517406 seconds.
GPU Script, Compile on Execution
Elapsed time is 110.462466 seconds.
```


-test3, nboot = 100_000 (8GB)
```bash
Using nboot: 100000
Loaded inputs
original script took 386.21000719070435 seconds
Using device: cuda
gpu array script took 235.99200344085693 seconds

# matlab
Using nboot:
      100000

Original Script, Compile on Execution
Elapsed time is 274.250803 seconds.
GPU Script, Compile on Execution
(out of memory)

```


p.s.:
1. nboot = 1_000_000: gpu out of memory (75GB)
2. the default formate of the matrix is float64, which is the same for both cpu and gpu.
3. the random number generator is not the same for cpu and gpu.






