# Compiler and flags
CXX = nvcc
CXXFLAGS = -Xcompiler -Wall,-Wno-sign-compare,-Wno-endif-labels,-Wno-unused-variable -O3 -std=c++14

# Include directories for CUDA and other headers
INCLUDES = -I/usr/include -I/usr/local/cuda/include

# Source files
SRCS = constant.cu dynamics.cu polar.cu TD_hamiltonian.cu param.cu utilize.cu

# Header files
HDRS = constant.h dynamics.h polar.h TD_hamiltonian.h param.h utilize.h

# Object files
OBJS = $(SRCS:.cu=.o)

# Static library name
LIB = libcpu2des.a

# Default target
all: $(LIB)

# Compile object files
%.o: %.cu $(HDRS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -dc $< -o $@

# Link object files and create static library
$(LIB): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -lib -o $(LIB) -lcublas

# Clean up the build directory
clean:
	rm -f $(OBJS) $(LIB) *.a

# Phony targets
.PHONY: all clean
