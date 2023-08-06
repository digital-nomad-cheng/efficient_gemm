
# Instructions

To compile:

```
make
```

To execute:

```
./mm <M> <N> <K> <verify>   # Multiply an MxK matrix with a KxN matrix. Verify the result if verify != 0.
./mm                        # Use default configutations (M=1000, N=1200, K=1100, verify=1)
```

# Result

CPU time: 1306.735039 ms
Allocation time: 0.440000 ms
Copy to GPU time: 1.304000 ms
Navie kernel time: 22.225000 ms
Copy from GPU time: 3.325000 ms
Coalesing kernel time: 3.036000 ms
Copy from GPU time: 1.325000 ms
Shared tiling kernel time: 2.236000 ms
Copy from GPU time: 1.039000 ms
Shared coalesing kernel time: 2.204000 ms
Copy from GPU time: 0.753000 ms
Thread tiling kernel time: 0.955000 ms
Copy from GPU time: 1.333000 ms
Deallocation time: 0.553000 ms
