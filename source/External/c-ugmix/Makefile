.SUFFIXES: .cpp

CFLAGS = -Wall -O3

libs = -lm -lnewmat

.cpp.o:
	$(CXX) -c $(CFLAGS) $*.cpp

.c.o:
	$(CXX) -c $(CFLAGS) $*.c

a.out: main.o unsupervised.h unsupervised.o k_mean.h k_mean.o mem.h mem.o
	$(CXX) main.o unsupervised.o k_mean.o mem.o -o a.out ${libs}

clean: 
	rm -f *.o