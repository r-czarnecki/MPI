#include <iostream>
#include <cstdio>
#include <sys/time.h>
#include <random>
#include <ctime>

int R(int a, int b) {
    return a + rand() % (b - a + 1);
}

int main(int argc, char *argv[]) {
    timeval e;
    gettimeofday(&e, 0);
    srand(e.tv_sec * 1000 + e.tv_usec % 1000);

    char letters[] = {'A', 'C', 'G', 'T'};

    int t = atoi(argv[1]);

    for (int i = 0; i < t; i++) {
        int c = R(0, 3);
        printf("%c", letters[c]);
    }
}