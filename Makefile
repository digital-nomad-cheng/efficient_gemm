
NVCC        = nvcc
NVCC_FLAGS  = -O3 -lineinfo
OBJ         = timer.o main.o kernel_launch.o 00_navie_gemm.o 01_coalesing_gemm.o \
	02_shared_mem_tiling_gemm.o 03_shared_mem_coalesing_gemm.o 04_shared_mem_thread_tiling_gemm.o
EXE         = mm


default: $(EXE)

%.o: %.cpp
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

