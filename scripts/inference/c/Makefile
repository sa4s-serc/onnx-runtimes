REPO_ROOT := $(realpath ../../..)
ORT_DIR   := $(REPO_ROOT)/lib/onnxruntime

CC     := gcc
CFLAGS := -O2 -std=c11 -Wall \
           -I$(ORT_DIR)/include \
           -L$(ORT_DIR)/lib \
           -Wl,-rpath,$(ORT_DIR)/lib
LIBS   := -lonnxruntime -lm

TARGET := infer

.PHONY: all clean

all: $(TARGET)

$(TARGET): infer.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)
