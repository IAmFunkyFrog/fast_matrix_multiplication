CC=gcc
MPICC=mpic++
RM=rm -rf
BUILD_DIR=build
SRC=./src
FLAGS=-O3 -fopenmp

all: main

build_dir:
	mkdir -p $(BUILD_DIR)

experiment: $(SRC)/main.c $(SRC)/matrix.c $(SRC)/matrix.h
	$(CC) $(FLAGS) -I$(SRC) $(SRC)/main.c $(SRC)/matrix.c -o $(BUILD_DIR)/experiment

main: experiment

clean:
	$(RM) $(BUILD_DIR)