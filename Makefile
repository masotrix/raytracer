CC= nvcc
#CFLAGS1= -gencode arch=compute_35,code=sm_35 -std=c++11
#CFLAGS1+= -g -I. -lGLEW -lGL -lglfw -G
CFLAGS= -arch=sm_52 -std=c++14 -lpng
CFLAGS+= -g -I. -G #--ptxas-options=-v
CCPP= g++
CPPFLAGS= -std=c++14 -lpng -g -I.

SRC= $(wildcard *.cu)
OBJ= $(patsubst %.cu, %.o, $(SRC)) display.o

# Link

all: run

run: $(OBJ)
	$(CC) $^ -o $@ $(CFLAGS)

# Compile

%.o: %.cu %.cuh
	$(CC) $< -dc -o $@ $(CFLAGS)

%.o: %.cpp %.h
	$(CCPP) $< -c -o $@ $(CPPFLAGS)

# Clean

.PHONY: clean

clean:
	rm -f *.o

