CXX ?= nvcc
TARGET := gpu-dryrun
SRC := gpu-dryrun.cpp.cu

CXXFLAGS := -std=c++17 -O2
LDFLAGS := -ldl

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
