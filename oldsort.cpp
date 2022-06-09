
template<typename T>
void sort(DataSource &dataSource, int i, int rank, MPI_Offset start, MPI_Offset end, T *buffer) {
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Offset totalSize = dataSource.getTotalGenomeSize(i);
    MPI_Offset myOffset = dataSource.getNodeGenomeOffset(i);
    int my_start = getOffset(totalSize, i, rank);
    int my_end = getOffset(totalSize, i, rank + 1) - 1;
    int procs_start = getRank(nprocs, totalSize, start);
    int procs_end = getRank(nprocs, totalSize, end);
    MPI_Comm *myComm;

    MPI_Comm_split(MPI_COMM_WORLD, (rank < procs_start || rank > procs_end), rank, myComm);
    
    if (rank < procs_start || rank > procs_end) {
        MPI_Comm_free(myComm);
        return;
    }

    if (start > my_end || end < my_start) {
        return;
    }

    if (my_start < start && start <= my_end) {
        my_start = start;
    }

    if (my_end > end && end >= my_start) {
        my_end = end;
    }


    std::sort(buffer + my_start - myOffset, buffer + my_end - myOffset);

    if (procs_start == procs_end) {
        return;
    }

    int num_procs = procs_end - procs_start + 1;
    T *sendBuf = new T[num_procs];
    int cnt = 0;

    for (int j = my_start - myOffset; j <= my_end - myOffset; j += std::max(1, (my_end - my_start + 1) / num_procs)) {
        sendBuf[cnt] = buffer[j];
        cnt++;
    }

    for (int j = cnt; j < num_procs; j++) {
        sendBuf[j] = sendBuf[j - 1];
    }

    T *recvBuf = (rank == procs_start) ? new T[num_procs * num_procs] : nullptr;
    MPI_Gather(sendBuf, num_procs * sizeof(T), MPI_BYTE, recvBuf, num_procs * num_procs * sizeof(T), MPI_BYTE, 0, myComm);
    delete[] sendBuf;

    int *splitters = new T[num_procs];

    if (rank == procs_start) {
        std::sort(recvBuf, recvBuf + num_procs * num_procs);
        cnt = 0;
        for (int j = num_procs - 1; j < num_procs * num_procs; j += num_procs) {
            splitters[cnt] = recvBuf[j];
            cnt++;
        }

        delete[] recvBuf;
    }

    MPI_Bcast(splitters, num_procs * sizeof(T), MPI_BYTE, 0, myComm);

    std::vector<
}