CC          := cc
CFLAGS      := -c -O3 -Wall
LFLAGS      := -O3 -Wall -lstdc++
ALL         := genome_index.exe

all : $(ALL)

%.exe : %.o  data_source.o
	$(CC) $(LFLAGS) -o genome_index data_source.o $<


%.o : %.cpp
	$(CC) $(CFLAGS) $<

clean :
	rm -f *.o *.out *.err genome_index

