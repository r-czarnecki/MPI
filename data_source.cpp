#include <math.h>
#include <mpi.h>
#include <cstdlib>
#include <cassert>
#include <string>

#include "data_source.h"

DataSource::DataSource(char *genome_in) : genome_in(genome_in) {};

uint64_t DataSource::getTotalGenomeSize(int i) {
    MPI_File fh;
    MPI_Offset filesize;

    std::string filename = getGenomeFilename(i);

    assert(MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) == 0);
    assert(MPI_File_get_size(fh, &filesize) == 0);
    assert(MPI_File_close(&fh) == 0);

    assert(filesize > 0);

    return filesize;
}

uint64_t DataSource::getNodeGenomeSize(int i) {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    MPI_Offset totalSize = getTotalGenomeSize(i);
    MPI_Offset nodeSize = (totalSize + nprocs - rank - 1) / nprocs;
    assert(nodeSize > 0);

    return nodeSize;
}

uint64_t DataSource::getNodeGenomeOffset(int i) {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    MPI_Offset totalSize = getTotalGenomeSize(i);

    MPI_Offset offset = rank * (totalSize / nprocs) + std::min(totalSize % nprocs, (MPI_Offset) rank);

    return offset;
}

void DataSource::getNodeGenomeValues(int i, char *buffer) {
    int rank, nprocs;

    MPI_File fh;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Offset nodeSize = getNodeGenomeSize(i);
    std::string filename = getGenomeFilename(i);

    assert(MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) == 0);
    assert(MPI_File_seek(fh, getNodeGenomeOffset(i), MPI_SEEK_SET) == 0);

    MPI_Status status;
    assert(MPI_File_read(fh, buffer, nodeSize, MPI_CHAR, &status) == 0);
    assert(MPI_File_close(&fh) == 0);

    buffer[nodeSize] = 0;
}

std::string DataSource::getGenomeFilename(int i) {
    return genome_in + "_" + std::to_string(i);
}


