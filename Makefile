
NVCC        = nvcc
NVCC_FLAGS  = -O3
OBJ         = main.o kernel_launch.o 00_navie_gemm.o 01_coalesing_gemm.o
EXE         = mm


default: $(EXE)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

