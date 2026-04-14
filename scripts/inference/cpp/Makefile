REPO_ROOT := $(realpath ../../..)
ORT_DIR   := $(REPO_ROOT)/lib/onnxruntime

CXX    := g++
CXXFLAGS := -O2 -std=c++17 -Wall \
             -I$(ORT_DIR)/include \
             -L$(ORT_DIR)/lib \
             -Wl,-rpath,$(ORT_DIR)/lib
LIBS   := -lonnxruntime

TARGET := infer

.PHONY: all clean

all: $(TARGET)

$(TARGET): infer.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)
