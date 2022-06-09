#include <algorithm>
#include <fstream>
#include <utility>
#include <cassert>
#include <vector>
#include <mpi.h>

#include "data_source.h"

using KMer = unsigned long long int;
using Rank = long long int;

MPI_Offset getOffset(MPI_Offset totalSize, int i, int rank) {
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    return rank * (totalSize / nprocs) + std::min(totalSize % nprocs, (MPI_Offset) rank);
}

int getRank(int nprocs, int totalSize, MPI_Offset offset) {
    int chunkSize = (totalSize / nprocs);
    if (offset <= (totalSize % nprocs) + (totalSize % nprocs) * chunkSize) {
        return offset / (chunkSize + 1);
    }
    else {
        return (offset - (totalSize % nprocs)) / chunkSize;
    }
}

template<typename T, typename F>
void writeAll(T *B5, F f, int size, int nprocs, int rank, const char *s = " ") {
    if (rank != 0) {
        MPI_Send(B5, size * sizeof(T), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }
    else {
        for (int i = 0; i < size; i++) {
            // printf(s, f(B5[i]));
            std::cout << f(B5[i]) << s;
        }
        for (int j = 1; j < nprocs; j++) {
            T *eee = new T[size];
            MPI_Recv(eee, size * sizeof(T), MPI_BYTE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < size; i++) {
                // printf(s, f(eee[i]));
                std::cout << f(eee[i]) << s;
            }

            delete[] eee;
        }
        printf("==========\n");
    }
}


template<typename T>
void sort(MPI_Offset totalSize, MPI_Offset myOffset, uint64_t bufferSize, int i, int rank, MPI_Offset start, MPI_Offset end, T *buffer) {
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int my_start = getOffset(totalSize, i, rank);
    int my_end = getOffset(totalSize, i, rank + 1) - 1;
    int procs_start = getRank(nprocs, totalSize, start);
    int procs_end = getRank(nprocs, totalSize, end);
    MPI_Comm myComm;

    MPI_Comm_split(MPI_COMM_WORLD, (rank < procs_start || rank > procs_end), rank, &myComm);
    printf("a %d | %d %d | %d\n", rank, my_start, my_end, (rank < procs_start || rank > procs_end));
    
    if (rank < procs_start || rank > procs_end) {
        MPI_Comm_free(&myComm);
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


    if (rank == 0) {
        printf("SORTING BUFFER\n");
        for (int i = my_start - myOffset; i <= my_end - myOffset; i++)
            printf("%llu ", buffer[i].second);
        printf("\n");
    }
    std::sort(buffer + my_start - myOffset, buffer + my_end + 1 - myOffset);
    if (rank == 0) {
        for (int i = my_start - myOffset; i <= my_end - myOffset; i++)
            printf("%llu ", buffer[i].first);
        printf("\n");
    }

    if (procs_start == procs_end) {
        return;
    }

    int num_procs = procs_end - procs_start + 1;
    int communicateWith = -1;
    T *newBuffer = new T[my_end - my_start + 1];
    MPI_Comm_rank(myComm, &rank);

    for (int phase = 0; phase < num_procs; phase++) {
        if (phase % 2 == 0) {
            communicateWith = (rank % 2 == 0) ? rank - 1 : rank + 1;
        }
        else {
            communicateWith = (rank % 2 == 0) ? rank + 1 : rank - 1;
        }

        if (communicateWith >= 0 && communicateWith < num_procs) {
            int recvSize = (totalSize + nprocs - communicateWith - 1) / nprocs;
            if (communicateWith == procs_start) {
                recvSize = getOffset(totalSize, i, communicateWith + 1) - start;
            }
            else if (communicateWith == procs_end) {
                recvSize = end - getOffset(totalSize, i, communicateWith) + 1;
            }

            T *recvBuf = new T[recvSize];
            printf("%d -> %d | %d %d\n", rank, communicateWith, (my_end - my_start + 1), recvSize);
            MPI_Sendrecv(buffer + my_start - myOffset, (my_end - my_start + 1) * sizeof(T), MPI_BYTE, communicateWith, 0,
                         recvBuf, recvSize * sizeof(T), MPI_BYTE, communicateWith, 0, myComm, MPI_STATUS_IGNORE);

            if (communicateWith > rank) {
                int ptr1 = my_start - myOffset, ptr2 = 0;
                for (int i = 0; i < (my_end - my_start + 1); i++) {
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
                int ptr1 = my_end - myOffset, ptr2 = recvSize - 1;
                for (int i = my_end - my_start; i >= 0; i--) {
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

            if (rank == 0) {
                printf("SORTING\n");
                for (int i = my_start - myOffset; i <= my_end - myOffset; i++)
                    printf("%llu ", buffer[i].first);
                printf("\n");
                for (int i = 0; i < recvSize; i++)
                    printf("%llu ", recvBuf[i].first);
                printf("\n");
                for (int i = 0; i < (my_end - my_start + 1); i++) 
                    printf("%llu ", newBuffer[i].first);
                printf("\n");
            }

            memcpy(buffer + my_start - myOffset, newBuffer, (my_end - my_start + 1) * sizeof(T));
            delete[] recvBuf;
        }
    }
    delete[] newBuffer;
    MPI_Comm_free(&myComm);
}

// template<>
// void sort<std::pair<Rank, std::pair<Rank, MPI_Offset>>>(MPI_Offset totalSize, MPI_Offset myOffset, uint64_t bufferSize, int i, int rank, MPI_Offset start, MPI_Offset end, std::pair<Rank, std::pair<Rank, MPI_Offset>> *buffer) {
//     int nprocs;
//     MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
//     int my_start = getOffset(totalSize, i, rank);
//     int my_end = getOffset(totalSize, i, rank + 1) - 1;
//     int procs_start = getRank(nprocs, totalSize, start);
//     int procs_end = getRank(nprocs, totalSize, end);
//     MPI_Comm myComm;

//     MPI_Comm_split(MPI_COMM_WORLD, (rank < procs_start || rank > procs_end), rank, &myComm);
//     printf("a %d | %d %d | %d\n", rank, my_start, my_end, (rank < procs_start || rank > procs_end));
    
//     if (rank < procs_start || rank > procs_end) {
//         MPI_Comm_free(&myComm);
//         return;
//     }

//     if (start > my_end || end < my_start) {
//         return;
//     }

//     if (my_start < start && start <= my_end) {
//         my_start = start;
//     }

//     if (my_end > end && end >= my_start) {
//         my_end = end;
//     }


//     if (rank == 1) {
//         printf("SORTING BUFFER\n");
//         for (int i = my_start - myOffset; i <= my_end - myOffset; i++)
//             printf("%llu ", buffer[i].second.second);
//         printf("\n");
//     }
//     std::sort(buffer + my_start - myOffset, buffer + my_end + 1 - myOffset);
//     if (rank == 0) {
//         for (int i = my_start - myOffset; i <= my_end - myOffset; i++)
//             printf("%llu ", buffer[i].second.second);
//         printf("\n");
//     }

//     if (procs_start == procs_end) {
//         return;
//     }

//     int num_procs = procs_end - procs_start + 1;
//     int communicateWith = -1;
//     std::pair<Rank, std::pair<Rank, MPI_Offset>> *newBuffer = new std::pair<Rank, std::pair<Rank, MPI_Offset>>[my_end - my_start + 1];
//     MPI_Comm_rank(myComm, &rank);

//     for (int phase = 0; phase < num_procs; phase++) {
//         if (phase % 2 == 0) {
//             communicateWith = (rank % 2 == 0) ? rank - 1 : rank + 1;
//         }
//         else {
//             communicateWith = (rank % 2 == 0) ? rank + 1 : rank - 1;
//         }

//         if (communicateWith >= 0 && communicateWith < num_procs) {
//             int recvSize = (totalSize + nprocs - communicateWith - 1) / nprocs;
//             if (communicateWith == procs_start) {
//                 recvSize = getOffset(totalSize, i, communicateWith + 1) - start;
//             }
//             else if (communicateWith == procs_end) {
//                 recvSize = end - getOffset(totalSize, i, communicateWith) + 1;
//             }

//             std::pair<Rank, std::pair<Rank, MPI_Offset>> *recvBuf = new std::pair<Rank, std::pair<Rank, MPI_Offset>>[recvSize];
//             printf("%d -> %d | %d %d\n", rank, communicateWith, (my_end - my_start + 1), recvSize);
//             MPI_Sendrecv(buffer + my_start - myOffset, (my_end - my_start + 1) * sizeof(std::pair<Rank, std::pair<Rank, MPI_Offset>>), MPI_BYTE, communicateWith, 0,
//                          recvBuf, recvSize * sizeof(std::pair<Rank, std::pair<Rank, MPI_Offset>>), MPI_BYTE, communicateWith, 0, myComm, MPI_STATUS_IGNORE);

//             if (communicateWith > rank) {
//                 int ptr1 = my_start - myOffset, ptr2 = 0;
//                 for (int i = 0; i < (my_end - my_start + 1); i++) {
//                     if (ptr1 > my_end - myOffset) {
//                         newBuffer[i] = recvBuf[ptr2];
//                         ptr2++;
//                     }
//                     else if (ptr2 == recvSize || buffer[ptr1] < recvBuf[ptr2]) {
//                         newBuffer[i] = buffer[ptr1];
//                         ptr1++;
//                     }
//                     else {
//                         newBuffer[i] = recvBuf[ptr2];
//                         ptr2++;
//                     }
//                 }
//             }
//             else {
//                 int ptr1 = my_end - myOffset, ptr2 = recvSize - 1;
//                 for (int i = my_end - my_start; i >= 0; i--) {
//                     if (ptr1 < my_start - myOffset) {
//                         newBuffer[i] = recvBuf[ptr2];
//                         ptr2--;
//                     }
//                     else if (ptr2 < 0 || buffer[ptr1] > recvBuf[ptr2]) {
//                         newBuffer[i] = buffer[ptr1];
//                         ptr1--;
//                     }
//                     else {
//                         newBuffer[i] = recvBuf[ptr2];
//                         ptr2--;
//                     }
//                 }
//             }

//             if (rank == 0) {
//                 printf("SORTING\n");
//                 for (int i = my_start - myOffset; i <= my_end - myOffset; i++)
//                     printf("%llu ", buffer[i].second.second);
//                 printf("\n");
//                 for (int i = 0; i < recvSize; i++)
//                     printf("%llu ", recvBuf[i].second.second);
//                 printf("\n");
//                 for (int i = 0; i < (my_end - my_start + 1); i++) 
//                     printf("%llu ", newBuffer[i].second.second);
//                 printf("\n");
//             }

//             memcpy(buffer + my_start - myOffset, newBuffer, (my_end - my_start + 1) * sizeof(std::pair<Rank, std::pair<Rank, MPI_Offset>>));
//             delete[] recvBuf;
//         }
//     }
//     delete[] newBuffer;
//     MPI_Comm_free(&myComm);
// }

template<typename F>
void serialize(int rank, int nprocs, F f) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        f();
        int i = 1;
        MPI_Send(&i, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == nprocs - 1) {
        int i;
        MPI_Recv(&i, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        f();
    }
    else {
        int i;
        MPI_Recv(&i, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        f();
        MPI_Send(&i, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }
}

unsigned int charToInt(char c) {
    switch (c) {
        case 'A':
            return 0;
        case 'C':
            return 1;
        case 'G':
            return 2;
        case 'T':
            return 3;
        default:
            return 0;
    }
}

unsigned int getNextChar(int &ptr, int &ptr2, uint64_t bufferSize, int nprocs, int k, char* buffer, char *endBuffer) {
    unsigned int nextChar;
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

std::pair<KMer, MPI_Offset> *sortKmers(MPI_Offset totalSize, MPI_Offset myOffset, uint64_t bufferSize, int rank, int nprocs, char *buffer, int k) {
    char *endBuffer = new char[nprocs * k];
    char *sendBuffer = new char[k];
    if (bufferSize >= k) {
        memcpy(sendBuffer, buffer, k * sizeof(char));
    }
    else {
        memcpy(sendBuffer, buffer, bufferSize * sizeof(char));
        for (int i = 0; i < k - bufferSize; i++) {
            sendBuffer[bufferSize + i] = 0;
        }
    }
    MPI_Allgather(sendBuffer, k, MPI_CHAR, endBuffer, k, MPI_CHAR, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("%d %d | %d\n", nprocs, k, bufferSize);
        for (int i = 0; i < k; i++) {
            if (sendBuffer[i] < 65 || sendBuffer[i] > 85)
                printf("%d ", sendBuffer[i]);
            else
                printf("%c ", sendBuffer[i]);
        }
        printf("\n");
        for (int i = 0; i < nprocs * k; i++) {
            if (endBuffer[i] < 65 || endBuffer[i] > 85)
                printf("%d ", endBuffer[i]);
            else
                printf("%c ", endBuffer[i]);
        }
        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    KMer kmer = 0;
    int ptr = 0, ptr2 = (rank + 1) * k;
    for (int i = 0; i < k; i++) {
        unsigned int nextChar = getNextChar(ptr, ptr2, bufferSize, nprocs, k, buffer, endBuffer);
        kmer <<= 2;
        kmer |= nextChar;
    }

    std::pair<KMer, MPI_Offset> *B = new std::pair<KMer, MPI_Offset>[bufferSize];
    B[0] = std::make_pair(kmer, myOffset);

    for (int i = 1; i < bufferSize; i++) {
        auto nextChar = getNextChar(ptr, ptr2, bufferSize, nprocs, k, buffer, endBuffer);
        kmer <<= 2;
        kmer |= nextChar;

        // kmer &= 3; // USUŃ TO

        B[i] = std::make_pair(kmer, myOffset + i);
    }

    delete[] endBuffer;
    delete[] sendBuffer;

    MPI_Barrier(MPI_COMM_WORLD);
    // if (rank == 0) {
    //     for (int j = 0; j < bufferSize; j++) {
    //         printf("\t");
    //         auto elem = B[j];
    //         KMer kmer = elem.first;
    //         for (int i = 0; i < k; i++) {
    //             // printf("%lld\n", kmer);
    //             unsigned int c = kmer >> (2 * k - 2);
    //             c &= 3;
    //             // printf("-> %llu\n", c);
    //             switch (c) {
    //                 case 0:
    //                     printf("A");
    //                     break;
    //                 case 1:
    //                     printf("C");
    //                     break;
    //                 case 2:
    //                     printf("G");
    //                     break;
    //                 case 3:
    //                     printf("T");
    //                     break;
    //             }
    //             kmer <<= 2;
    //         }
    //         printf("\n");
    //     }
    // }

    writeAll(B, [k](std::pair<KMer, MPI_Offset> e) {
            auto elem = e;
            std::string s;
            KMer kmer = elem.first;
            for (int i = 0; i < k; i++) {
                // printf("%lld\n", kmer);
                unsigned int c = kmer >> (2 * k - 2);
                c &= 3;
                // printf("-> %llu\n", c);
                switch (c) {
                    case 0:
                        s += "A";
                        break;
                    case 1:
                        s += "C";
                        break;
                    case 2:
                        s += "G";
                        break;
                    case 3:
                        s += "T";
                        break;
                }
                kmer <<= 2;
            }
            return s;
        }, bufferSize, nprocs, rank, "\n");
    // serialize(rank, nprocs, [B, rank, k, bufferSize](){
    //     printf("%d: \n", rank);
    //     for (int j = 0; j < bufferSize; j++) {
    //         printf("\t");
    //         auto elem = B[j];
    //         KMer kmer = elem.first;
    //         for (int i = 0; i < k; i++) {
    //             printf("%llu\n", kmer);
    //             unsigned int c = kmer >> (k - 2);
    //             switch (c) {
    //                 case 0:
    //                     printf("A");
    //                     break;
    //                 case 1:
    //                     printf("C");
    //                     break;
    //                 case 2:
    //                     printf("G");
    //                     break;
    //                 case 3:
    //                     printf("T");
    //                     break;
    //             }
    //             kmer <<= 2;
    //         }
    //         printf("\n");
    //     }
    // });

    MPI_Barrier(MPI_COMM_WORLD);

    sort(totalSize, myOffset, bufferSize, 0, rank, 0, totalSize - 1, B);
    MPI_Barrier(MPI_COMM_WORLD);
    // if (rank == 0) {
    //     for (int j = 0; j < bufferSize; j++) {
    //         printf("\t");
    //         auto elem = B[j];
    //         KMer kmer = elem.first;
    //         for (int i = 0; i < k; i++) {
    //             // printf("%lld\n", kmer);
    //             unsigned int c = kmer >> (2 * k - 2);
    //             c &= 3;
    //             // printf("-> %llu\n", c);
    //             switch (c) {
    //                 case 0:
    //                     printf("A");
    //                     break;
    //                 case 1:
    //                     printf("C");
    //                     break;
    //                 case 2:
    //                     printf("G");
    //                     break;
    //                 case 3:
    //                     printf("T");
    //                     break;
    //             }
    //             kmer <<= 2;
    //         }
    //         printf("\n");
    //     }
    // }
    writeAll(B, [k](std::pair<KMer, MPI_Offset> e) {
            auto elem = e;
            std::string s;
            KMer kmer = elem.first;
            for (int i = 0; i < k; i++) {
                // printf("%lld\n", kmer);
                unsigned int c = kmer >> (2 * k - 2);
                c &= 3;
                // printf("-> %llu\n", c);
                switch (c) {
                    case 0:
                        s += "A";
                        break;
                    case 1:
                        s += "C";
                        break;
                    case 2:
                        s += "G";
                        break;
                    case 3:
                        s += "T";
                        break;
                }
                kmer <<= 2;
            }
            return s;
        }, bufferSize, nprocs, rank, "\n");
    return B;
}

template<typename T>
Rank *rebucket(MPI_Offset totalSize, MPI_Offset myOffset, uint64_t bufferSize, int rank, int nprocs, std::pair<T, MPI_Offset> *B, Rank *B2 = nullptr) {
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

    if (rank == 1)
        printf("PREV %llu %llu\n", prev.first, B[0].first);
    for (int i = 0; i < bufferSize; i++) {
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

    MPI_Allgather(&maxRank, 1, MPI_UNSIGNED_LONG_LONG, counts, 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

    Rank prevMaxRank = 0;
    for (int i = 0; i < rank; i++) {
        prevMaxRank = std::max(prevMaxRank, counts[i]);
    }

    if (result[0] == 0) {
        result[0] = prevMaxRank;
    }

    for (int i = 1; i < bufferSize; i++) {
        result[i] = std::max(result[i - 1], result[i]);
    }

    delete[] prevs;
    delete[] counts;

    return result;
}

template<typename T>
Rank *reorder(MPI_Offset totalSize, MPI_Offset myOffset, uint64_t bufferSize, int rank, int nprocs, std::pair<T, MPI_Offset> *B, Rank *ranks) {
    std::vector<std::pair<MPI_Offset, Rank>> destinations;

    for (int i = 0; i < bufferSize; i++) {
        destinations.push_back(std::make_pair(B[i].second, ranks[i]));
    }

    std::vector<int> counts(nprocs, 0);
    std::vector<int> displacemets(nprocs, 0);
    std::vector<std::pair<MPI_Offset, Rank>> send;
    std::pair<MPI_Offset, Rank> *recv = new std::pair<MPI_Offset, Rank>[bufferSize];
    if (rank == 0) {
        printf("ASDSAD %d\n", counts[0]);
    }

    std::sort(destinations.begin(), destinations.end());
    for (int i = 0; i < destinations.size(); i++) {
        auto dest = destinations[i];
        int destRank = getRank(nprocs, totalSize, dest.first);
        if (rank == 0) {
            printf("DESTRANK %d | POS %llu | VAL %llu\n", destRank, dest.first, dest.second);
            printf("COUNTS %d\n", counts[destRank]);
        }
        if (counts[destRank] == 0) {
            displacemets[destRank] = i * sizeof(std::pair<MPI_Offset, Rank>);
        }

        counts[destRank] += sizeof(std::pair<MPI_Offset, Rank>);
        send.push_back(std::make_pair(dest.first, dest.second));
    }

    int *recvCounts = new int[nprocs];
    int *recvDispl = new int[nprocs];

    MPI_Alltoall(counts.data(), 1, MPI_INT, recvCounts, 1, MPI_INT, MPI_COMM_WORLD);

    recvDispl[0] = 0;
    for (int i = 1; i < nprocs; i++) {
        recvDispl[i] = recvDispl[i - 1] + recvCounts[i - 1];
    }

    if (rank == 0) {
        printf("SIZEOF: %lu\n", sizeof(std::pair<MPI_Offset, Rank>));
        printf("RECVCOUNTS: ");
        for (int i = 0; i < nprocs; i++) {
            printf("%d ", recvCounts[i]);
        }
        printf("\n");
        
        printf("RECVDISPL: ");
        for (int i = 0; i < nprocs; i++) {
            printf("%d ", recvDispl[i]);
        }
        printf("\n");
    }

    MPI_Alltoallv(send.data(), counts.data(), displacemets.data(), MPI_BYTE, recv, recvCounts, recvDispl, MPI_BYTE, MPI_COMM_WORLD);

    Rank *result = new Rank[bufferSize];
    for (int i = 0; i < bufferSize; i++) {
        result[recv[i].first - myOffset] = recv[i].second;
    }

    delete[] recvCounts;
    delete[] recvDispl;
    delete[] recv;

    return result;
}

Rank *shift(MPI_Offset totalSize, MPI_Offset myOffset, uint64_t bufferSize, int rank, int nprocs, Rank *ranks, int h) {
    Rank *result = new Rank[bufferSize];

    MPI_Request sendReq1, sendReq2;
    int destOff1 = myOffset - h;
    int dest1 = (destOff1 < 0) ? -1 : getRank(nprocs, totalSize, destOff1);

    int destOff2 = myOffset + bufferSize - 1 - h;
    int dest2 = (destOff2 < 0) ? -1 : getRank(nprocs, totalSize, destOff2);

    int split = -1;
    if (dest2 != -1) {
        split = bufferSize - (destOff2 - getOffset(totalSize, 0, dest2) + 1);
        if (rank == 1) {
            printf("SHIFT %d | %d (%d) %d %d (%d)\n", h, dest1, destOff1, split, dest2, destOff2);
        }
        if (dest2 == dest1) {
            dest1 = -1;
            MPI_Isend(ranks, bufferSize, MPI_UNSIGNED_LONG_LONG, dest2, 0, MPI_COMM_WORLD, &sendReq2);
        }
        else {
            if (dest1 != -1) {
                MPI_Isend(ranks, split, MPI_UNSIGNED_LONG_LONG, dest1, 0, MPI_COMM_WORLD, &sendReq1);
            }
            
            MPI_Isend(ranks + split, bufferSize - split, MPI_UNSIGNED_LONG_LONG, dest2, 0, MPI_COMM_WORLD, &sendReq2);
        }
    }

    int recvOff1 = myOffset + h;
    int src1 = (recvOff1 >= totalSize) ? -1 : getRank(nprocs, totalSize, recvOff1);

    int recvOff2 = myOffset + bufferSize - 1 + h;
    int src2 = (recvOff2 >= totalSize) ? -1 : getRank(nprocs, totalSize, recvOff2);

    split = -1;
    if (src1 != -1) {
        split = getOffset(totalSize, 0, src1 + 1) - recvOff1;
        if (rank == 2) {
            printf("SPLIT %d\n", split);
        }

        MPI_Recv(result, split, MPI_UNSIGNED_LONG_LONG, src1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (src2 != -1 && bufferSize - split != 0) {
            printf("AASD %d %d\n", rank, split);
            MPI_Recv(result + split, bufferSize - split, MPI_UNSIGNED_LONG_LONG, src2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("ASDASDASD %d\n", rank);
        }
        else {
            for (int i = split; i < bufferSize; i++) {
                result[i] = -1;
            }
        }
    } 

    if (dest2 != -1) {
        MPI_Wait(&sendReq2, MPI_STATUS_IGNORE);

        if (dest1 != -1) {
            MPI_Wait(&sendReq1, MPI_STATUS_IGNORE);
        }
    }

    printf("END %d\n", rank);
    return result;
}

bool allSingletons(MPI_Offset totalSize, MPI_Offset myOffset, uint64_t bufferSize, int rank, int nprocs, Rank *ranks) {
    bool allSingletonsLocal = true;

    for (int i = 0; i < bufferSize; i++) {
        if (ranks[i] != myOffset + i) {
            allSingletonsLocal = false;
        }
    }

    bool allSingletons;
    MPI_Allreduce(&allSingletonsLocal, &allSingletons, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);

    return allSingletons;
}

void getSA(MPI_Offset totalSize, MPI_Offset offset, uint64_t size, int rank, int nprocs, std::pair<KMer, MPI_Offset> *B, Rank *ranks, int k) {
    int h = 0;
    bool done = false;

    while (true) {
        h += k;
        if (rank == 0)
            printf("NEXT LOOP %d\n", h);
        auto B2 = reorder(totalSize, offset, size, rank, nprocs, B, ranks);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            printf("REORDER\n");
            // for (int i = 0; i < size; i++) {
            //     printf("%llu ", B2[i]);
            // }
            // printf("\n");
        }

        writeAll(B2, [](Rank e){ return e; }, size, nprocs, rank);

        auto B3 = shift(totalSize, offset, size, rank, nprocs, B2, h);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            printf("SHIFT %d\n", h);
            // for (int i = 0; i < size; i++) {
            //     printf("%llu ", B2[i]);
            // }
            // printf("\n");
            // for (int i = 0; i < size; i++) {
            //     printf("%llu ", B3[i]);
            // }
            // printf("\n");
        }

        writeAll(B2, [](Rank e){ return e; }, size, nprocs, rank);
        writeAll(B3, [](Rank e){ return e; }, size, nprocs, rank);

        std::pair<Rank, std::pair<Rank, MPI_Offset>> *B4 = new std::pair<Rank, std::pair<Rank, MPI_Offset>>[size];

        for (int i = 0; i < size; i++) {
            B4[i] = std::make_pair(B2[i], std::make_pair(B3[i], offset + i));
        }
        delete[] B2;

        sort(totalSize, offset, size, 0, rank, 0, totalSize - 1, B4);

        if (rank == 0) {
            printf("SORTED\n");
            // for (int i = 0; i < size; i++) {
            //     printf("%llu ", B4[i].first);
            // }
            // printf("\n");
            // for (int i = 0; i < size; i++) {
            //     printf("%llu ", B4[i].second.first);
            // }
            // printf("\n");
            // for (int i = 0; i < size; i++) {
            //     printf("%llu ", B4[i].second.second);
            // }
            // printf("\n");
        }
        
        writeAll(B4, [](std::pair<Rank, std::pair<Rank, MPI_Offset>> e){ return e.first; }, size, nprocs, rank);
        writeAll(B4, [](std::pair<Rank, std::pair<Rank, MPI_Offset>> e){ return e.second.first; }, size, nprocs, rank);
        writeAll(B4, [](std::pair<Rank, std::pair<Rank, MPI_Offset>> e){ return e.second.second; }, size, nprocs, rank);

        for (int i = 0; i < size; i++) {
            B[i] = std::make_pair(B4[i].first, B4[i].second.second);
            B3[i] = B4[i].second.first;
        }
        delete[] B4;

        auto B5 = rebucket(totalSize, offset, size, rank, nprocs, B, B3);
        delete[] B3;

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            printf("REBUCKET 2\n");
            for (int i = 0; i < size; i++) {
                printf("%llu ", B5[i]);
            }
            printf("\n");
        }

        writeAll(B5, [](Rank e){ return e; }, size, nprocs, rank);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            printf("SA\n");
            for (int i = 0; i < size; i++) {
                printf("%llu ", B[i].second);
            }
            printf("\n");
        }
        
        
        done = allSingletons(totalSize, offset, size, rank, nprocs, B5);
        MPI_Barrier(MPI_COMM_WORLD);
        delete[] ranks;
        ranks = B5;

        if (done) {
            delete[] ranks;
            return;
        }
    }
}

std::vector<int> answerQueries(MPI_Offset totalSize, MPI_Offset myOffset, uint64_t bufferSize, int rank, int nprocs, char *buffer, std::pair<KMer, MPI_Offset> *SA, std::vector<std::string> &queries) {
    int queryStart = -1;
    int queryEnd = -1;
    bool done = false;

    int *recvBuff = new int[nprocs];
    char *recvChars = new char[getOffset(totalSize, 0, 1)];
    std::vector<int> countsSend(nprocs, 0); 
    std::vector<int> displacements(nprocs, 0);
    std::vector<int> results;
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

    for (int queryIdx = queryStart; queryIdx < queryEnd; queryIdx++) {
        std::string query = queries[queryIdx];
        int firstOcc = -1, lastOcc = -1;
        
        for (int t = 0; t <= 1; t++) {
            int begin = 0, end = totalSize - 1;

            while (begin <= end) {
                int mid = (begin + end) / 2;
                if (rank == 1) {
                    printf("DEBUG | %d %d %d\n", begin, mid, end);
                }

                MPI_Allgather(&mid, 1, MPI_INT, recvBuff, 1, MPI_INT, MPI_COMM_WORLD);
                
                std::vector<int> SASend(nprocs, 0);
                for (int i = 0; i < nprocs; i++) {
                    int pos = recvBuff[i];
                    if (rank == 1) {
                        printf("AA %d\n", pos);
                    }
                    if (pos >= myOffset && pos < myOffset + bufferSize) {
                        int sa = SA[pos - myOffset].second;

                        // countsSend[i]++;
                        // displacements[i] = i;
                        // SASend[i] = sa;
                        MPI_Request req;
                        MPI_Isend(&sa, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &req);
                        requests.push_back(req);
                    }
                }

                MPI_Request scatterReq;
                int sa;
                if (rank == 1) {
                    printf("DEBUG 3 | %d %d\n", getRank(nprocs, totalSize, mid), countsSend[0]);
                    for (int i = 0; i < nprocs; i++) {
                        printf("%d %d %d\n", SASend[i], countsSend[i], displacements[i]);
                    }
                }

                MPI_Recv(&sa, 1, MPI_INT, getRank(nprocs, totalSize, mid), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (rank == 1) {
                    printf("DEBUG 3.5 | %d\n", sa);
                }

                for (auto &req : requests) {
                    MPI_Wait(&req, MPI_STATUS_IGNORE);
                }

                requests.clear();
                
                // if (getRank(nprocs, totalSize, mid) != rank) {
                //     MPI_Iscatterv(SASend.data(), countsSend.data(), displacements.data(), MPI_INT, &sa, 1, MPI_INT, rank, MPI_COMM_WORLD, &scatterReq);
                //     // if (rank == 0) {
                //     //     printf("DEBUG 2 | %d\n", getRank(nprocs, totalSize, mid));
                //     // }
                //     MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, &sa, 1, MPI_INT, getRank(nprocs, totalSize, mid), MPI_COMM_WORLD);
                //     if (rank == 0) {
                //         printf("DEBUG 2 | %d\n", getRank(nprocs, totalSize, mid));
                //     }
                //     MPI_Wait(&scatterReq, MPI_STATUS_IGNORE);
                // }
                // else {
                //     MPI_Scatterv(SASend.data(), countsSend.data(), displacements.data(), MPI_INT, &sa, 1, MPI_INT, rank, MPI_COMM_WORLD);
                // }

                
                if (rank == 1) {
                    printf("DEBUG 2 %d\n", sa);
                }
                int checkedChars = 0;
                int saRank = getRank(nprocs, totalSize, sa);
                int cmp = -2;

                while (checkedChars < query.size() && saRank < nprocs && cmp == -2) {
                    int recvSize = getOffset(totalSize, 0, saRank + 1) - getOffset(totalSize, 0, saRank);
                    // printf("BARRIER 1 | %d\n", rank);
                    // MPI_Barrier(MPI_COMM_WORLD);
                    MPI_Allgather(&saRank, 1, MPI_INT, recvBuff, 1, MPI_INT, MPI_COMM_WORLD);
                    if (rank == 1) {
                        printf("DEBUG 212 %d\n", saRank);
                    }

                    for (int i = 0; i < nprocs; i++) {
                        countsSend[i] = 0;
                        if (recvBuff[i] == rank) {
                            MPI_Request req;
                            MPI_Isend(buffer, bufferSize, MPI_CHAR, i, 0, MPI_COMM_WORLD, &req);
                            requests.push_back(req);
                        }

                        if (rank == 1) {
                            printf("%d %d %d\n", recvBuff[i], countsSend[i], displacements[i]);
                        }
                    }

                    if (rank == 1) {
                        printf("DEBUG 4 | %d\n", recvSize);
                    }

                    MPI_Recv(recvChars, recvSize, MPI_CHAR, saRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    if (rank == 1) {
                        recvChars[recvSize] = 0;
                        printf("DEBUG 4.5 | %d\n", recvSize);
                        printf("GOT %s\n", recvChars);
                    }

                    for (auto &req : requests) {
                        MPI_Wait(&req, MPI_STATUS_IGNORE);
                    }

                    requests.clear();

                    // assert(MPI_Iscatterv(buffer, countsSend.data(), displacements.data(), MPI_CHAR, recvChars, bufferSize, MPI_CHAR, rank, MPI_COMM_WORLD, &scatterReq) == 0);

                    // MPI_Wait(&scatterReq, MPI_STATUS_IGNORE);

                    // if (saRank != rank) {
                    //     if (rank == 0) {
                    //         // MPI_Wait(&scatterReq, MPI_STATUS_IGNORE);
                    //         printf("DEBUG 4 | %d %d \n", recvSize, saRank);
                    //     }
                    //     assert(MPI_Scatterv(nullptr, nullptr, nullptr, MPI_CHAR, recvChars, recvSize, MPI_CHAR, saRank, MPI_COMM_WORLD) == 0);
                    // }


                    if (rank == 1) {
                        recvChars[recvSize] = 0;
                        printf("DEBUG 5 | %d - %d | %s | %s\n", saRank, recvSize, recvChars, recvChars + sa - getOffset(totalSize, 0, saRank));
                    }

                    for (int i = sa - getOffset(totalSize, 0, saRank); i < recvSize; i++, checkedChars++) {
                        if (recvChars[i] < query[checkedChars]) {
                            cmp = 1;
                            printf("DONE %d %d\n", rank, cmp);
                            break;
                        }
                        
                        if (recvChars[i] > query[checkedChars]) {
                            cmp = -1;
                            printf("DONE %d %d\n", rank, cmp);
                            break;
                        }
                        
                        if (checkedChars == query.size() - 1) {
                            cmp = 0;
                            printf("DONE %d %d\n", rank, cmp);
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
                    MPI_Allgather(&saRank, 1, MPI_INT, recvBuff, 1, MPI_INT, MPI_COMM_WORLD);
                    for (int i = 0; i < nprocs; i++) {
                        if (recvBuff[i] != -1) {
                            cont = false;
                        }

                        countsSend[i] = 0;
                        if (recvBuff[i] == rank) {
                            MPI_Request req;
                            MPI_Isend(buffer, bufferSize, MPI_CHAR, i, 0, MPI_COMM_WORLD, &req);
                            requests.push_back(req);
                            // countsSend[i] = bufferSize;
                            // displacements[i] = 0;
                        }

                        if (rank == 1) {
                            printf("CONT %d\n", recvBuff[i]);
                        }
                    }

                    for (auto &req : requests) {
                        MPI_Wait(&req, MPI_STATUS_IGNORE);
                    }

                    requests.clear();

                    // MPI_Scatterv(buffer, countsSend.data(), displacements.data(), MPI_CHAR, recvChars, 0, MPI_CHAR, rank, MPI_COMM_WORLD);
                }

                printf("======= (%d)\n", rank);

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

            printf("DDDONE %d | %d %d\n", rank, firstOcc, lastOcc);
        }

        results.push_back((firstOcc == -1) ? 0 : lastOcc - firstOcc + 1);
    }

    while (true) {
        int send = -1;
        MPI_Allgather(&send, 1, MPI_INT, recvBuff, 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> SASend(nprocs, 0);

        bool ret = true;
        for (int i = 0; i < nprocs; i++) {
            int pos = recvBuff[i];
            if (pos != -1) {
                ret = false;
            }

            if (pos >= myOffset && pos < myOffset + bufferSize) {
                int sa = SA[pos - myOffset].second;

                MPI_Request req;
                MPI_Isend(&sa, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &req);
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

        // MPI_Scatterv(SASend.data(), countsSend.data(), displacements.data(), MPI_INT, nullptr, 0, MPI_INT, rank, MPI_COMM_WORLD);

        bool cont = false;
        while (!cont) {
            cont = true;
            // printf("BARRIER 2 | %d\n", rank);
            // MPI_Barrier(MPI_COMM_WORLD);
            MPI_Allgather(&send, 1, MPI_INT, recvBuff, 1, MPI_INT, MPI_COMM_WORLD);
            for (int i = 0; i < nprocs; i++) {
                if (recvBuff[i] != -1) {
                    cont = false;
                }

                countsSend[i] = 0;
                if (recvBuff[i] == rank) {
                    MPI_Request req;
                    MPI_Isend(buffer, bufferSize, MPI_CHAR, i, 0, MPI_COMM_WORLD, &req);
                    requests.push_back(req);
                }

                if (rank == 0) {
                    printf("%d %d %d\n", recvBuff[i], countsSend[i], displacements[i]);
                }
            }

            // MPI_Scatterv(buffer, countsSend.data(), displacements.data(), MPI_CHAR, recvChars, 0, MPI_CHAR, rank, MPI_COMM_WORLD);
            for (auto &req : requests) {
                MPI_Wait(&req, MPI_STATUS_IGNORE);
            }

            requests.clear();
        }
    }
}

std::vector<int> getResults(int i, std::vector<std::string> &queries, DataSource &dataSource) {
    int size;
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int test = i;

    size = dataSource.getNodeGenomeSize(test);
    int totalSize = dataSource.getTotalGenomeSize(test);
    int offset = dataSource.getNodeGenomeOffset(test);
    char *buffer = new char[size];
    dataSource.getNodeGenomeValues(test, buffer);

    // sort(totalSize, offset, size, 0, rank, 0, totalSize - 1, buffer);
    // MPI_Barrier(MPI_COMM_WORLD);
    // serialize(rank, nprocs, [rank, size, buffer, &dataSource](){
    //     printf("%d: ", rank);
    //     for (int i = 0; i < size; i++) {
    //         printf("%c ", buffer[i]);
    //     }
    //     printf("\n");
    // });
    // MPI_Barrier(MPI_COMM_WORLD);
// 1000010110110100000000000000000000000000000000000000000000000000
// 1001110001011000010110110100000000000000000000000000000000000000
// 9634325502852333568
// 11265854798303330304
    int k = 8 * sizeof(long long int) / 2;

    auto B = sortKmers(totalSize, offset, size, rank, nprocs, buffer, k);
    if (rank == 0)
        printf("SORTED KMERS\n");
    writeAll(B, [](std::pair<KMer, MPI_Offset> k){return k.first;}, size, nprocs, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    serialize(rank, nprocs, [rank, size, B, &dataSource](){
        // printf("%d: ", rank);
        for (int i = 0; i < size; i++) {
            printf("%llu\n", B[i].first);
        }
        // printf("\n");
    });
    MPI_Barrier(MPI_COMM_WORLD);

    auto ranks = rebucket(totalSize, offset, size, rank, nprocs, B);

    if (rank == 0) {
        printf("REBUCKET\n");
        for (int i = 0; i < size; i++) {
            printf("%llu ", ranks[i]);
        }
        printf("\n");
    }

    getSA(totalSize, offset, size, rank, nprocs, B, ranks, k);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("SA 2\n");
        // for (int i = 0; i < size; i++) {
        //     printf("%llu ", B[i].second);
        // }
        // printf("\n");
    }
    // serialize(rank, nprocs, [rank, size, B, nprocs, &dataSource](){
    //     // printf("%d: ", rank);
        
    //     // printf("\n");
    // });
    if (rank != 0) {
        MPI_Send(B, size * sizeof(std::pair<Rank, MPI_Offset>), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }
    else {
        for (int i = 0; i < size; i++) {
            printf("%llu ", B[i].second);
        }
        for (int j = 1; j < nprocs; j++) {
            std::pair<Rank, MPI_Offset> *eee = new std::pair<Rank, MPI_Offset>[size];
            MPI_Recv(eee, size * sizeof(std::pair<Rank, MPI_Offset>), MPI_BYTE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < size; i++) {
                printf("%llu ", eee[i].second);
            }
        }
        printf("==========\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // if (rank == 0) {
    //     printf("==========\n");
    //     // for (int i = 0; i < size; i++) {
    //     //     printf("%llu ", B[i].second);
    //     // }
    //     // printf("\n");
    // }
    MPI_Barrier(MPI_COMM_WORLD);

    
    auto result = answerQueries(totalSize, offset, size, rank, nprocs, buffer, B, queries);

    delete[] B;
    delete[] buffer;

    if (rank == 0) {
        printf("RESULTS\n");
    }

    return result;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int genomeNumber = atoi(argv[1]);
    int queryNumber = atoi(argv[2]);
    DataSource dataSource{argv[3]};
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::vector<std::string> queries;
    std::ifstream cin(argv[4]);
    std::string query;
    std::vector<std::string> results;

    while (cin >> query) {
        queries.push_back(query);
    }

    for (int i = 0; i < genomeNumber; i++) {
        auto result = getResults(i, queries, dataSource);

        for (int j = 0; j < result.size(); j++) {
            if (i == 0) {
                results.push_back(std::to_string(result[j]) + " ");
            }
            else {
                results[j] += std::to_string(result[j]) + " ";
            }

            if (i == genomeNumber - 1) {
                results[j] += "\n";
            }
        }
    }

    MPI_File fh;
    assert(MPI_File_open(MPI_COMM_WORLD, argv[5], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh) == 0);
    printf("%llu\n", (unsigned long long)(&fh));
    int tmp;

    if (rank != 0) {
        MPI_Recv(&tmp, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (auto &res : results) {
        printf("%s ", res.c_str());
        MPI_File_write_shared(fh, res.c_str(), res.size(), MPI_CHAR, MPI_STATUS_IGNORE);

        printf("\n");
    }

    if (rank + 1 != nprocs) {
        MPI_Send(&tmp, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }

    MPI_File_close(&fh);
    // serialize(rank, nprocs, [rank, result, size, B, &dataSource](){
    //     // printf("%d: ", rank);
    //     for (auto i : result) {
    //         printf("%d ", i);
    //     }
    //     printf("\n");
    //     // printf("\n");
    // });

    // writeAll(result.data(), [](int e){return e;}, result.size(), nprocs, rank, "\n");

    // delete[] B;

    MPI_Finalize();
}