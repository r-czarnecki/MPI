#include <algorithm>
#include <fstream>
#include <utility>
#include <cassert>
#include <iostream>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "data_source.h"

using KMer = unsigned long long int;
using Rank = long long int;


MPI_Offset getOffset(MPI_Offset totalSize, long long int i, long long int rank) {
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    return rank * (totalSize / nprocs) + std::min(totalSize % nprocs, (MPI_Offset) rank);
}

long long int getRank(long long int nprocs, long long int totalSize, MPI_Offset offset) {
    long long int chunkSize = (totalSize / nprocs);
    if (offset <= (totalSize % nprocs) + (totalSize % nprocs) * chunkSize) {
        return offset / (chunkSize + 1);
    }
    else {
        return (offset - (totalSize % nprocs)) / chunkSize;
    }
}


template<typename T>
void sort(MPI_Offset totalSize, MPI_Offset myOffset, long long int bufferSize, long long int i, int rank, MPI_Offset start, MPI_Offset end, T *buffer) {
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    long long int my_start = getOffset(totalSize, i, rank);
    long long int my_end = getOffset(totalSize, i, rank + 1) - 1;
    long long int procs_start = getRank(nprocs, totalSize, start);
    long long int procs_end = getRank(nprocs, totalSize, end);
    MPI_Comm myComm;

    MPI_Comm_split(MPI_COMM_WORLD, (rank < procs_start || rank > procs_end) || (start > my_end || end < my_start), rank, &myComm);
    
    if (rank < procs_start || rank > procs_end) {
        MPI_Comm_free(&myComm);
        return;
    }

    if (start > my_end || end < my_start) {
        MPI_Comm_free(&myComm);
        return;
    }

    if (my_start < start && start <= my_end) {
        my_start = start;
    }

    if (my_end > end && end >= my_start) {
        my_end = end;
    }

    std::sort(buffer + my_start - myOffset, buffer + my_end + 1 - myOffset);

    if (procs_start == procs_end) {
        MPI_Comm_free(&myComm);
        return;
    }

    long long int num_procs = procs_end - procs_start + 1;
    long long int communicateWith = -1;
    T *newBuffer = new T[my_end - my_start + 1];
    MPI_Comm_rank(myComm, &rank);

    for (long long int phase = 0; phase < num_procs; phase++) {
        if (phase % 2 == 0) {
            communicateWith = (rank % 2 == 0) ? rank - 1 : rank + 1;
        }
        else {
            communicateWith = (rank % 2 == 0) ? rank + 1 : rank - 1;
        }

        if (communicateWith >= 0 && communicateWith < num_procs) {
            long long int recvSize = (totalSize + nprocs - communicateWith - 1) / nprocs;
            if (communicateWith == 0) {
                recvSize = getOffset(totalSize, i, procs_start + 1) - start;
            }
            else if (communicateWith == num_procs - 1) {
                recvSize = end - getOffset(totalSize, i, procs_end) + 1;
            }

            T *recvBuf = new T[recvSize];
            MPI_Sendrecv(buffer + my_start - myOffset, (my_end - my_start + 1) * sizeof(T), MPI_BYTE, communicateWith, 0,
                         recvBuf, recvSize * sizeof(T), MPI_BYTE, communicateWith, 0, myComm, MPI_STATUS_IGNORE);

            if (communicateWith > rank) {
                long long int ptr1 = my_start - myOffset, ptr2 = 0;
                for (long long int i = 0; i < (my_end - my_start + 1); i++) {
                    if (ptr1 > my_end - myOffset) {
                        newBuffer[i] = recvBuf[ptr2];
                        ptr2++;
                    }
                    else if (ptr2 == recvSize || buffer[ptr1] < recvBuf[ptr2]) {
                        newBuffer[i] = buffer[ptr1];
                        ptr1++;
                    }
                    else {
                        newBuffer[i] = recvBuf[ptr2];
                        ptr2++;
                    }
                }
            }
            else {
                long long int ptr1 = my_end - myOffset, ptr2 = recvSize - 1;
                for (long long int i = my_end - my_start; i >= 0; i--) {
                    if (ptr1 < my_start - myOffset) {
                        newBuffer[i] = recvBuf[ptr2];
                        ptr2--;
                    }
                    else if (ptr2 < 0 || buffer[ptr1] > recvBuf[ptr2]) {
                        newBuffer[i] = buffer[ptr1];
                        ptr1--;
                    }
                    else {
                        newBuffer[i] = recvBuf[ptr2];
                        ptr2--;
                    }
                }
            }

            memcpy(buffer + my_start - myOffset, newBuffer, (my_end - my_start + 1) * sizeof(T));
            delete[] recvBuf;
        }
    }
    delete[] newBuffer;
    MPI_Comm_free(&myComm);
}

unsigned long long int charToInt(char c) {
    switch (c) {
        case 'A':
            return 1;
        case 'C':
            return 2;
        case 'G':
            return 3;
        case 'T':
            return 4;
        default:
            return 0;
    }
}

unsigned long long int getNextChar(long long int &ptr, long long int &ptr2, long long int bufferSize, long long int nprocs, long long int k, char* buffer, char *endBuffer) {
    unsigned long long int nextChar;
    if (ptr < bufferSize) {
        nextChar = charToInt(buffer[ptr]);
        ptr++;
    }
    else if (ptr2 >= nprocs * k) {
        nextChar = 0;
    }
    else if (endBuffer[ptr2] == 0) {
        ptr2++;
        return getNextChar(ptr, ptr2, bufferSize, nprocs, k, buffer, endBuffer);
    }
    else {
        nextChar = charToInt(endBuffer[ptr2]);
        ptr2++;
    }

    return nextChar;
}

std::pair<KMer, MPI_Offset> *sortKmers(MPI_Offset totalSize, MPI_Offset myOffset, long long int bufferSize, long long int rank, long long int nprocs, char *buffer, long long int k) {
    char *endBuffer = new char[nprocs * k];
    char *sendBuffer = new char[k];
    if (bufferSize >= k) {
        memcpy(sendBuffer, buffer, k * sizeof(char));
    }
    else {
        memcpy(sendBuffer, buffer, bufferSize * sizeof(char));
        for (long long int i = 0; i < k - bufferSize; i++) {
            sendBuffer[bufferSize + i] = 0;
        }
    }
    MPI_Allgather(sendBuffer, k, MPI_CHAR, endBuffer, k, MPI_CHAR, MPI_COMM_WORLD);

    KMer kmer = 0;
    long long int ptr = 0, ptr2 = (rank + 1) * k;
    for (long long int i = 0; i < k; i++) {
        unsigned long long int nextChar = getNextChar(ptr, ptr2, bufferSize, nprocs, k, buffer, endBuffer);
        kmer <<= 4;
        kmer |= nextChar;
    }

    std::pair<KMer, MPI_Offset> *B = new std::pair<KMer, MPI_Offset>[bufferSize];
    B[0] = std::make_pair(kmer, myOffset);

    for (long long int i = 1; i < bufferSize; i++) {
        auto nextChar = getNextChar(ptr, ptr2, bufferSize, nprocs, k, buffer, endBuffer);
        kmer <<= 4;
        kmer &= ~15;
        kmer |= nextChar;

        B[i] = std::make_pair(kmer, myOffset + i);
    }

    delete[] endBuffer;
    delete[] sendBuffer;
    
    sort(totalSize, myOffset, bufferSize, 0, rank, 0, totalSize - 1, B);
    
    return B;
}

template<typename T>
Rank *rebucket(MPI_Offset totalSize, MPI_Offset myOffset, long long int bufferSize, long long int rank, long long int nprocs, std::pair<T, MPI_Offset> *B, Rank *B2 = nullptr) {
    Rank *result = new Rank[bufferSize];
    std::pair<T, Rank> *prevs = new std::pair<T, Rank>[nprocs];
    std::pair<T, Rank> prevSend = std::make_pair(B[bufferSize - 1].first, (B2 == nullptr) ? 0 : B2[bufferSize - 1]);

    MPI_Allgather(&prevSend, sizeof(std::pair<T, Rank>), MPI_BYTE, prevs, sizeof(std::pair<T, Rank>), MPI_BYTE, MPI_COMM_WORLD);

    Rank maxRank = 0;
    std::pair<T, Rank> prev;
    if (rank == 0) {
        prev = std::make_pair(B[0].first, (B2 == nullptr) ? 0 : B2[0]);
    }
    else {
        prev = prevs[rank - 1];
    }

    for (long long int i = 0; i < bufferSize; i++) {
        if (B[i].first != prev.first ||
            (B2 != nullptr && B2[i] != prev.second)) {
            result[i] = myOffset + i;
            maxRank = result[i];
        }
        else {
            result[i] = 0;
        }

        prev = std::make_pair(B[i].first, (B2 == nullptr) ? 0 : B2[i]);
    }

    Rank *counts = new Rank[nprocs];

    MPI_Allgather(&maxRank, 1, MPI_LONG_LONG, counts, 1, MPI_LONG_LONG, MPI_COMM_WORLD);

    Rank prevMaxRank = 0;
    for (long long int i = 0; i < rank; i++) {
        prevMaxRank = std::max(prevMaxRank, counts[i]);
    }

    if (result[0] == 0) {
        result[0] = prevMaxRank;
    }

    for (long long int i = 1; i < bufferSize; i++) {
        result[i] = std::max(result[i - 1], result[i]);
    }

    delete[] prevs;
    delete[] counts;

    return result;
}

template<typename T>
Rank *reorder(MPI_Offset totalSize, MPI_Offset myOffset, long long int bufferSize, long long int rank, long long int nprocs, std::pair<T, MPI_Offset> *B, Rank *ranks) {
    std::vector<std::pair<MPI_Offset, Rank>> destinations;

    for (long long int i = 0; i < bufferSize; i++) {
        destinations.push_back(std::make_pair(B[i].second, ranks[i]));
    }

    std::vector<int> counts(nprocs, 0);
    std::vector<int> counts2(nprocs, 0);
    std::vector<int> displacemets(nprocs, 0);
    std::vector<std::pair<MPI_Offset, Rank>> send(bufferSize);
    std::pair<MPI_Offset, Rank> *recv = new std::pair<MPI_Offset, Rank>[bufferSize];

    for (long long int i = 0; i < destinations.size(); i++) {
        auto dest = destinations[i];
        long long int destRank = getRank(nprocs, totalSize, dest.first);
        counts[destRank] += sizeof(std::pair<MPI_Offset, Rank>);
    }

    displacemets[0] = 0;
    for (long long int i = 1; i < nprocs; i++) {
        displacemets[i] = displacemets[i - 1] + counts[i - 1];
    }

    for (long long int i = 0; i < destinations.size(); i++) {
        auto dest = destinations[i];
        long long int destRank = getRank(nprocs, totalSize, dest.first);
        int idx = displacemets[destRank] / sizeof(std::pair<MPI_Offset, Rank>) + counts2[destRank];
        send[idx] = dest;
        counts2[destRank]++;
    }

    int *recvCounts = new int[nprocs];
    int *recvDispl = new int[nprocs];

    MPI_Alltoall(counts.data(), 1, MPI_INT, recvCounts, 1, MPI_INT, MPI_COMM_WORLD);

    recvDispl[0] = 0;
    for (long long int i = 1; i < nprocs; i++) {
        recvDispl[i] = recvDispl[i - 1] + recvCounts[i - 1];
    }

    MPI_Alltoallv(send.data(), counts.data(), displacemets.data(), MPI_BYTE, recv, recvCounts, recvDispl, MPI_BYTE, MPI_COMM_WORLD);

    Rank *result = new Rank[bufferSize];
    for (long long int i = 0; i < bufferSize; i++) {
        result[recv[i].first - myOffset] = recv[i].second;
    }

    delete[] recvCounts;
    delete[] recvDispl;
    delete[] recv;

    return result;
}

bool allSingletons(MPI_Offset totalSize, MPI_Offset myOffset, long long int bufferSize, long long int rank, long long int nprocs, Rank *ranks) {
    bool allSingletonsLocal = true;

    for (long long int i = 0; i < bufferSize; i++) {
        if (ranks[i] != myOffset + i) {
            allSingletonsLocal = false;
        }
    }

    bool allSingletons;
    MPI_Allreduce(&allSingletonsLocal, &allSingletons, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);

    return allSingletons;
}

void sortRanks(MPI_Offset totalSize, MPI_Offset myOffset, long long int bufferSize, long long int rank, long long int nprocs, Rank *B, std::pair<Rank, std::pair<Rank, MPI_Offset>> *B2) {
    long long int send[2] = {-1, -1};

    int tmp = -1;

    bool putFirst = true;
    Rank last = B[bufferSize - 1];
    Rank *lastRanks = new Rank[nprocs];
    MPI_Allgather(&last, 1, MPI_LONG_LONG, lastRanks, 1, MPI_LONG_LONG, MPI_COMM_WORLD);

    Rank prev = (rank == 0) ? B[0] + 1 : lastRanks[rank - 1];

    for (long long int i = 0; i < bufferSize; i++) {
        if (B[i] != prev) {
            if (putFirst) {
                send[0] = i + myOffset - 1;
                putFirst = false; 
            }

            if (send[1] != -1 && i + myOffset - send[1] > 1) {
                std::sort(B2 + send[1] - myOffset, B2 + i);
            }

            send[1] = i + myOffset;
        }

        prev = B[i];
    }

    if (rank == nprocs - 1 && send[1] != -1) {
        std::sort(B2 + send[1] - myOffset, B2 + bufferSize);
        send[1] = totalSize - 1;
    }

    long long int *recv = new long long int[2 * nprocs];
    MPI_Allgather(send, 2, MPI_LONG_LONG, recv, 2, MPI_LONG_LONG, MPI_COMM_WORLD);

    tmp = -1;
    for (int i = 0; i <= 2 * nprocs; i++) {
        long long int r = (i == 2 * nprocs) ? totalSize - 1 : recv[i];
        if (r == -1) {
            continue;
        }

        if (tmp == -1) {
            tmp = r;
        }
        else {
            if (r - tmp != 0) {
                sort(totalSize, myOffset, bufferSize, 0, rank, tmp, r, B2);
            }

            tmp = -1;
        }
    }

    delete[] lastRanks;
    delete[] recv;
}

void getSA(MPI_Offset totalSize, MPI_Offset offset, long long int size, long long int rank, long long int nprocs, std::pair<KMer, MPI_Offset> *B, Rank *ranks, long long int k) {
    long long int h = k;
    bool done = false;
    Rank *B3 = new Rank[size];
    auto B2 = reorder(totalSize, offset, size, rank, nprocs, B, ranks);

    while (true) {
        std::pair<Rank, MPI_Offset> *M = new std::pair<Rank, MPI_Offset>[size];
        std::pair<Rank, MPI_Offset> *M2 = new std::pair<Rank, MPI_Offset>[size];
        std::vector<int> count(nprocs + 1, 0);
        std::vector<int> count2(nprocs + 1, 0);
        std::vector<int> displacements(nprocs + 1, 0);
        for (int i = 0; i < size; i++) {
            M[i] = std::make_pair(B[i].second + h, offset + i);
            int r = (B[i].second + h >= totalSize) ? nprocs : getRank(nprocs, totalSize, B[i].second + h);
            count[r] += sizeof(std::pair<Rank, MPI_Offset>);
        }

        displacements[0] = 0;
        for (int i = 1; i <= nprocs; i++) {
            displacements[i] = displacements[i - 1] + count[i - 1];
        }

        for (int i = 0; i < size; i++) {
            int r = std::min(nprocs, getRank(nprocs, totalSize, M[i].first));
            M2[displacements[r] / sizeof(std::pair<Rank, MPI_Offset>) + count2[r]] = M[i];
            count2[r]++; 
        }

        int *recvCount = new int[nprocs];
        std::vector<int> recvDispl(nprocs + 1, 0);

        MPI_Alltoall(count.data(), 1, MPI_INT, recvCount, 1, MPI_INT, MPI_COMM_WORLD);

        recvDispl[0] = 0;
        int sumRecv = recvCount[0] / sizeof(std::pair<Rank, MPI_Offset>);
        int sumCount = count[0] / sizeof(std::pair<Rank, MPI_Offset>);
        for (int i = 1; i < nprocs; i++) {
            sumRecv += recvCount[i] / sizeof(std::pair<Rank, MPI_Offset>);
            sumCount += count[i] / sizeof(std::pair<Rank, MPI_Offset>);
            recvDispl[i] = recvDispl[i - 1] + recvCount[i - 1];
        }
        
        MPI_Alltoallv(M2, count.data(), displacements.data(), MPI_BYTE, M, recvCount, recvDispl.data(), MPI_BYTE, MPI_COMM_WORLD);

        for (int i = 0; i < sumRecv; i++) {
            M[i] = std::make_pair(B2[M[i].first - offset], M[i].second);
        }
        
        MPI_Alltoallv(M, recvCount, recvDispl.data(), MPI_BYTE, M2, count.data(), displacements.data(), MPI_BYTE, MPI_COMM_WORLD);

        delete[] recvCount;

        for (int i = 0; i < size; i++) {
            M[M2[i].second - offset].first = (M2[i].first >= totalSize) ? -1 : M2[i].first;
        }
        
        std::pair<Rank, std::pair<Rank, MPI_Offset>> *B4 = new std::pair<Rank, std::pair<Rank, MPI_Offset>>[size];

        for (long long int i = 0; i < size; i++) {
            B4[i] = std::make_pair(ranks[i], std::make_pair(M[i].first, B[i].second));
        }

        sortRanks(totalSize, offset, size, rank, nprocs, ranks, B4);

        for (long long int i = 0; i < size; i++) {
            B[i] = std::make_pair(B4[i].first, B4[i].second.second);
            B3[i] = B4[i].second.first;
        }
        delete[] B4;

        auto B5 = rebucket(totalSize, offset, size, rank, nprocs, B, B3);
        
        done = allSingletons(totalSize, offset, size, rank, nprocs, B5);
        delete[] ranks;
        ranks = B5;

        if (done) {
            delete[] ranks;
            delete[] M;
            delete[] M2;
            delete[] B2;
            delete[] B3;
            return;
        }

        std::fill(count.begin(), count.end(), 0);
        std::fill(count2.begin(), count2.end(), 0);
        std::fill(displacements.begin(), displacements.end(), 0);
        std::fill(recvDispl.begin(), recvDispl.end(), 0);
        for (int i = 0; i < size; i++) {
            M[i] = std::make_pair(B[i].second, ranks[i]);
            int d = getRank(nprocs, totalSize, B[i].second);
            count[d] += sizeof(std::pair<Rank, MPI_Offset>);
        }

        displacements[0] = 0;
        for (int i = 1; i < nprocs; i++) {
            displacements[i] = displacements[i - 1] + count[i - 1];
        }

        for (int i = 0; i < size; i++) {
            int d = getRank(nprocs, totalSize, B[i].second);
            int idx = displacements[d] / sizeof(std::pair<Rank, MPI_Offset>) + count2[d];
            M2[idx] = M[i];
            count2[d]++;
        }

        recvCount = new int[nprocs];

        MPI_Alltoall(count.data(), 1, MPI_INT, recvCount, 1, MPI_INT, MPI_COMM_WORLD);

        recvDispl[0] = 0;
        for (int i = 1; i <= nprocs; i++) {
            recvDispl[i] = recvDispl[i - 1] + recvCount[i - 1];
        }

        MPI_Alltoallv(M2, count.data(), displacements.data(), MPI_BYTE, M, recvCount, recvDispl.data(), MPI_BYTE, MPI_COMM_WORLD);

        for (int i = 0; i < size; i++) {
            B2[M[i].first - offset] = M[i].second;
        }

        delete[] recvCount;
        delete[] M;
        delete[] M2;

        h *= 2;
    }
}

std::vector<long long int> answerQueries(MPI_Offset totalSize, MPI_Offset myOffset, long long int bufferSize, long long int rank, long long int nprocs, char *buffer, std::pair<KMer, MPI_Offset> *SA, std::vector<std::string> &queries) {
    long long int queryStart = -1;
    long long int queryEnd = -1;

    long long int *recvBuff = new long long int[nprocs];
    char *recvChars = new char[getOffset(totalSize, 0, 1)];
    std::vector<long long int> countsSend(nprocs, 0); 
    std::vector<long long int> displacements(nprocs, 0);
    std::vector<long long int> results;
    std::vector<MPI_Request> requests;

    if (queries.size() <= nprocs) {
        if (rank < queries.size()) {
            queryStart = rank;
            queryEnd = rank + 1;
        }
    }
    else {
        queryStart = getOffset(queries.size(), 0, rank);
        queryEnd = getOffset(queries.size(), 0, rank + 1);
    }

    for (long long int queryIdx = queryStart; queryIdx < queryEnd; queryIdx++) {
        std::string query = queries[queryIdx];
        long long int firstOcc = -1, lastOcc = -1;
        
        for (long long int t = 0; t <= 1; t++) {
            long long int begin = 0, end = totalSize - 1;

            while (begin <= end) {
                long long int mid = (begin + end) / 2;

                MPI_Allgather(&mid, 1, MPI_LONG_LONG, recvBuff, 1, MPI_LONG_LONG, MPI_COMM_WORLD);
                
                std::vector<long long int> SASend(nprocs, 0);
                for (long long int i = 0; i < nprocs; i++) {
                    long long int pos = recvBuff[i];

                    if (pos >= myOffset && pos < myOffset + bufferSize) {
                        long long int sa = SA[pos - myOffset].second;
                        SASend[i] = sa;

                        MPI_Request req;
                        MPI_Isend(SASend.data() + i, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &req);
                        requests.push_back(req);
                    }
                }

                long long int sa;

                MPI_Recv(&sa, 1, MPI_LONG_LONG, getRank(nprocs, totalSize, mid), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (auto &req : requests) {
                    MPI_Wait(&req, MPI_STATUS_IGNORE);
                }

                requests.clear();
                
                long long int checkedChars = 0;
                long long int saRank = getRank(nprocs, totalSize, sa);
                long long int cmp = -2;

                while (checkedChars < query.size() && saRank < nprocs && cmp == -2) {
                    long long int recvSize = getOffset(totalSize, 0, saRank + 1) - getOffset(totalSize, 0, saRank);
                    MPI_Allgather(&saRank, 1, MPI_LONG_LONG, recvBuff, 1, MPI_LONG_LONG, MPI_COMM_WORLD);

                    for (long long int i = 0; i < nprocs; i++) {
                        countsSend[i] = 0;
                        if (recvBuff[i] == rank) {
                            MPI_Request req;
                            MPI_Isend(buffer, bufferSize, MPI_CHAR, i, 0, MPI_COMM_WORLD, &req);
                            requests.push_back(req);
                        }
                    }

                    MPI_Recv(recvChars, recvSize, MPI_CHAR, saRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    for (auto &req : requests) {
                        MPI_Wait(&req, MPI_STATUS_IGNORE);
                    }

                    requests.clear();

                    for (long long int i = sa - getOffset(totalSize, 0, saRank); i < recvSize; i++, checkedChars++) {
                        if (recvChars[i] < query[checkedChars]) {
                            cmp = 1;
                            break;
                        }
                        
                        if (recvChars[i] > query[checkedChars]) {
                            cmp = -1;
                            break;
                        }
                        
                        if (checkedChars == query.size() - 1) {
                            cmp = 0;
                            break;
                        }
                    }

                    saRank++;
                    sa = getOffset(totalSize, 0, saRank);

                }

                bool cont = false;
                saRank = -1;
                while (!cont) {
                    cont = true;
                    MPI_Allgather(&saRank, 1, MPI_LONG_LONG, recvBuff, 1, MPI_LONG_LONG, MPI_COMM_WORLD);
                    for (long long int i = 0; i < nprocs; i++) {
                        if (recvBuff[i] != -1) {
                            cont = false;
                        }

                        countsSend[i] = 0;
                        if (recvBuff[i] == rank) {
                            MPI_Request req;
                            MPI_Isend(buffer, bufferSize, MPI_CHAR, i, 0, MPI_COMM_WORLD, &req);
                            requests.push_back(req);
                        }
                    }

                    for (auto &req : requests) {
                        MPI_Wait(&req, MPI_STATUS_IGNORE);
                    }

                    requests.clear();
                }

                if (cmp == -1) {
                    end = mid - 1;
                }
                else if (cmp == 1 || cmp == -2) {
                    begin = mid + 1;
                }
                else if (t == 0) {
                    firstOcc = mid;
                    end = mid - 1;
                }
                else {
                    lastOcc = mid;
                    begin = mid + 1;
                }
            }
        }

        results.push_back((firstOcc == -1) ? -1 : lastOcc - firstOcc + 1);
    }

    while (true) {
        long long int send = -1;
        MPI_Allgather(&send, 1, MPI_LONG_LONG, recvBuff, 1, MPI_LONG_LONG, MPI_COMM_WORLD);

        std::vector<long long int> SASend(nprocs, 0);

        bool ret = true;
        for (long long int i = 0; i < nprocs; i++) {
            long long int pos = recvBuff[i];
            if (pos != -1) {
                ret = false;
            }

            if (pos >= myOffset && pos < myOffset + bufferSize) {
                long long int sa = SA[pos - myOffset].second;
                SASend[i] = sa;

                MPI_Request req;
                MPI_Isend(SASend.data() + i, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &req);
                requests.push_back(req);
            }
        }

        if (ret) {
            delete[] recvBuff;
            delete[] recvChars;
            return results;
        }

        for (auto &req : requests) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
        }

        requests.clear();

        bool cont = false;
        while (!cont) {
            cont = true;
            MPI_Allgather(&send, 1, MPI_LONG_LONG, recvBuff, 1, MPI_LONG_LONG, MPI_COMM_WORLD);
            for (long long int i = 0; i < nprocs; i++) {
                if (recvBuff[i] != -1) {
                    cont = false;
                }

                countsSend[i] = 0;
                if (recvBuff[i] == rank) {
                    MPI_Request req;
                    MPI_Isend(buffer, bufferSize, MPI_CHAR, i, 0, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
                }
            }

            for (auto &req : requests) {
                MPI_Wait(&req, MPI_STATUS_IGNORE);
            }

            requests.clear();
        }
    }
}

std::vector<long long int> getResults(long long int i, std::vector<std::string> &queries, DataSource &dataSource) {
    long long int size;
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    long long int test = i;

    size = dataSource.getNodeGenomeSize(test);
    long long int totalSize = dataSource.getTotalGenomeSize(test);
    long long int offset = dataSource.getNodeGenomeOffset(test);
    char *buffer = new char[size + 1];
    dataSource.getNodeGenomeValues(test, buffer);

    long long int k = 8 * sizeof(long long int) / 4;

    auto B = sortKmers(totalSize, offset, size, rank, nprocs, buffer, k);

    auto ranks = rebucket(totalSize, offset, size, rank, nprocs, B);

    getSA(totalSize, offset, size, rank, nprocs, B, ranks, k);

    auto result = answerQueries(totalSize, offset, size, rank, nprocs, buffer, B, queries);

    delete[] B;
    delete[] buffer;

    return result;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    long long int genomeNumber = atoi(argv[1]);
    long long int queryNumber = atoi(argv[2]);
    DataSource dataSource{argv[3]};
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::vector<std::string> queries;
    std::ifstream cin(argv[4]);
    std::string query;
    std::vector<std::string> results;

    while (cin >> query && queryNumber--) {
        queries.push_back(query);
    }

    for (long long int i = 0; i < genomeNumber; i++) {
        auto result = getResults(i, queries, dataSource);

        for (long long int j = 0; j < result.size(); j++) {
            if (i == 0) {
                results.push_back(std::to_string(result[j]));
            }
            else {
                results[j] += " " + std::to_string(result[j]);
            }

            if (i == genomeNumber - 1) {
                results[j] += "\n";
            }
        }
    }

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, argv[5], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    long long int tmp;

    if (rank != 0) {
        MPI_Recv(&tmp, 1, MPI_LONG_LONG, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (auto &res : results) {
        MPI_File_write_shared(fh, res.c_str(), res.size(), MPI_CHAR, MPI_STATUS_IGNORE);
    }

    if (rank + 1 != nprocs) {
        MPI_Send(&tmp, 1, MPI_LONG_LONG, rank + 1, 0, MPI_COMM_WORLD);
    }

    MPI_File_close(&fh);

    MPI_Finalize();
}