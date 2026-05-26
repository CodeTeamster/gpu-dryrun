CXX ?= nvcc
TARGET := gpu-dryrun
SRC := gpu-dryrun.cpp.cu

ARCH_FLAGS := -gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_89,code=sm_89
CONDA_FLAGS :=
ifneq ($(CONDA_PREFIX),)
CONDA_FLAGS += -I$(CONDA_PREFIX)/include -L$(CONDA_PREFIX)/lib
endif

CXXFLAGS := -std=c++17 -O2 $(ARCH_FLAGS) $(CONDA_FLAGS)
LDFLAGS := -ldl

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
