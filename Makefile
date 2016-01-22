FLAGS = -Wall -O3

all: pyhmm

pyhmm: pyhmm.o hmm.o
	gcc $(FLAGS) -shared pyhmm.o hmm.o -o pyhmm.so

pyhmm.o: pyhmm.c
	gcc $(FLAGS) -c -fpic pyhmm.c -o pyhmm.o

hmm.o: hmm.c
	gcc $(FLAGS) -c -fpic hmm.c -o hmm.o

clean:
	rm -f hmm.o pyhmm.o pyhmm.so
