CC1= nvcc
CC2= g++
#CFLAGS1= -gencode arch=compute_35,code=sm_35 -std=c++11
#CFLAGS1+= -g -I. -lGLEW -lGL -lglfw -G
CFLAGS1= -arch=sm_35 -std=c++11 -lpng
CFLAGS1+= -g -I. -G #--ptxas-options=-v
CFLAGS2= -std=c++14 -g -I.

# Link

all: run

run: main.o display.o parsing.o model.o
	$(CC1) $^ -o $@ $(CFLAGS1)

# Compile

display.o: display.cpp display.h
	$(CC2) $< -c -o $@ $(CFLAGS2)

parsing.o: parsing.cpp parsing.h model.h
	$(CC2) $< -c -o $@ $(CFLAGS2)

model.o: model.cu model.h display.h
	$(CC1) $< -dc -o $@ $(CFLAGS1)

main.o: main.cu parsing.h display.h model.h
	$(CC1) $< -dc -o $@ $(CFLAGS1)


# Clean

.PHONY: clean

clean:
	rm -f *.o

