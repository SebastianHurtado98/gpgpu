#include <iostream>
#include <vector>

using namespace std;

int main() {
    int N;
    cin >> N;

    std::vector<std::vector<int>> A(N, std::vector<int>(N, 0));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 0));
    std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));

    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
        }
    }

    int k;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

}