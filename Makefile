FLAGS = -Wall -O3

CC = mpicc

all: pyhmm

train: train.c hmm.c
	$(CC) $(FLAGS) -lgsl -lm train.c hmm.c -o train

pyhmm: pyhmm.o hmm.o
	$(CC) $(FLAGS) -shared pyhmm.o hmm.o -o pyhmm.so

pyhmm.o: pyhmm.c
	$(CC) $(FLAGS) -c -fpic pyhmm.c -o pyhmm.o

hmm.o: hmm.c
	$(CC) $(FLAGS) -c -fpic hmm.c -o hmm.o

clean:
	rm -f hmm.o pyhmm.o pyhmm.so train
