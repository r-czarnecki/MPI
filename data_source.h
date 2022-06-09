#ifndef DATASOURCE_H
#define DATASOURCE_H

#include <string>

class DataSource {
public:
    DataSource(char *genome_in);

    uint64_t getTotalGenomeSize(int i);  // Returns total length of i-th genome
    uint64_t getNodeGenomeSize(int i);   // Returns the length of i-th genome part
                                         // assigned to the current node
    uint64_t getNodeGenomeOffset(int i); // Returns the first index of i-th genome that is
                                         // assigned to the current node
    void getNodeGenomeValues(int i, char *buffer); // Reads data of length :getNodeGenomeSize(i)
                                                   // at offset :getNodeGenomeOffset(i)
                                                   // of the i-th genome using MPI I/O,
                                                   // into the buffer.
private:
    std::string genome_in;
    std::string getGenomeFilename(int i);
};

#endif // DATASOURCE_H
