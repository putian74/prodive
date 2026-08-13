# Makefile for KL divergence computation with CUDA

# Compiler settings
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -O3 -arch=sm_86 -std=c++14 -Xcompiler -fPIC
CXX_FLAGS = -O3 -std=c++14 -fPIC -fopenmp

# Include directories (include current directory for cnpy.h)
INCLUDES = -I/usr/local/cuda/include -I/usr/include/eigen3 -I.

# Library directories and libraries (ADD -lgomp for OpenMP support)
LIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas -lz -lgomp

# Source files (add multi_gpu.cpp and cnpy.cpp to local sources)
CUDA_SOURCES = kernels.cu
CPP_SOURCES = main.cpp command_line.cpp matrix_inversion.cpp util.cpp multi_gpu.cpp cnpy.cpp

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
ALL_OBJECTS = $(CUDA_OBJECTS) $(CPP_OBJECTS)

# Target executable
TARGET = kl_divergence

# Default target
all: $(TARGET)

# Link all objects into final executable
$(TARGET): $(ALL_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -fopenmp -o $@ $^ $(LIBS)

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ source files
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f *.o $(TARGET)

# Clean all generated files including results
clean-all: clean
	rm -rf results/ debug/ *.txt *.npy

# Create necessary directories
setup:
	mkdir -p results
	mkdir -p data

# Debug build with additional flags
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: CXX_FLAGS += -g -DDEBUG
debug: $(TARGET)

# Profile build with profiling support
profile: NVCC_FLAGS += -lineinfo
profile: CXX_FLAGS += -pg
profile: $(TARGET)

# Check if cnpy files exist
check-cnpy:
	@if [ ! -f cnpy.h ]; then echo "Error: cnpy.h not found in current directory"; exit 1; fi
	@if [ ! -f cnpy.cpp ]; then echo "Error: cnpy.cpp not found in current directory"; exit 1; fi
	@echo "cnpy files found: cnpy.h and cnpy.cpp"

# Build with cnpy check
build-with-check: check-cnpy all

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build the main executable (default)"
	@echo "  build-with-check - Check for cnpy files and build"
	@echo "  check-cnpy       - Verify cnpy.h and cnpy.cpp exist"
	@echo "  debug            - Build with debug symbols and flags"
	@echo "  profile          - Build with profiling support"
	@echo "  clean            - Remove object files and executable"
	@echo "  clean-all        - Remove all generated files including results"
	@echo "  setup            - Create necessary directories"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Note: Requires cnpy.h and cnpy.cpp in the current directory"
	@echo "      Uses OpenMP for parallel matrix inversion"

# Phony targets
.PHONY: all clean clean-all setup debug profile help check-cnpy build-with-check
