import warnings
import time
import numpy
import math
from numba.core.errors import NumbaPerformanceWarning
from numba import cuda

warnings.simplefilter('ignore', category = NumbaPerformanceWarning)
print_full_time = True

# --- funkcje

@cuda.jit
def operation_A(M, multipliers, index, q, n):
    pos = cuda.grid(1)
    if pos + q + 1 < n:
        k = index[q][pos + q + 1]
        i = q
        multipliers[k][i] = M[k][i] / M[i][i]

@cuda.jit
def operation_B(M, multipliers, substractions, index, q, n): 
    pos = cuda.grid(1)
    if pos + q * (q + 1) < n * (n + 1):
        k, j = index[q][pos + q * (q + 1)]
        i = q
        substractions[k][j][i] = M[i][j] * multipliers[k][i]

@cuda.jit
def operation_C(M, substractions, index, q, n): 
    pos = cuda.grid(1)
    if pos + q * (q + 1) < n * (n + 1):
        k, j = index[q][pos + q * (q + 1)]
        i = q
        M[k][j] = M[k][j] - substractions[k][j][i] 

def gauss(n, M):
    # współbieżne sprowadzanie macierzy do postaci trójkątnej na rdzeniach CUDA
    start_time = time.time()

    M = cuda.to_device(numpy.array(M, numpy.float64))
    multipliers = cuda.to_device(numpy.zeros([n + 1, n + 1], numpy.float64))
    substractions = cuda.to_device(numpy.zeros([n + 1, n + 1, n + 1], numpy.float64))

    threadsperblock = 512  

    # index = [(k, q) for k in range(q + 1, n)]
    index_for_A = numpy.zeros([n - 1, n], numpy.int32)
    for q in range(0, n - 1):
        p = 0
        for k in range(q + 1, n):
            index_for_A[q][q + 1 + p] = k
            p += 1

    index_for_A = cuda.to_device(index_for_A)
    
    # index = [(k, j, q) for k in range(q + 1, n) for j in range(q, n + 1)]
    index_for_BC = numpy.zeros([n - 1, n * (n + 1), 2], numpy.int32)
    for q in range(0, n - 1):
        p = 0
        for k in range(q + 1, n):
            for j in range(q, n + 1):
                index_for_BC[q][q * (q + 1) + p] = (k, j)
                p += 1

    index_for_BC = cuda.to_device(index_for_BC)

    end_time = time.time()
    if print_full_time: print("Inicializing GPU memory done in", round(end_time - start_time, 5), "seconds.") 
    else: print(f"{(end_time - start_time):.5f}", end = ' ') 
    
    start_time = time.time()
    
    for q in range(0, n - 1):
        blockspergrid = math.ceil((n - q - 1) / threadsperblock)
        operation_A[blockspergrid, threadsperblock](M, multipliers, index_for_A, q, n) 

        blockspergrid = math.ceil(((n * (n + 1)) - (q + 1) * (q - 1)) / threadsperblock)
        operation_B[blockspergrid, threadsperblock](M, multipliers, substractions, index_for_BC, q, n) 

        operation_C[blockspergrid, threadsperblock](M, substractions, index_for_BC, q, n) 

    end_time = time.time()
    if print_full_time: print("Calculations done in", round(end_time - start_time, 5), "seconds.") 
    else: print(f"{(end_time - start_time):.5f}", end = ' ') 
    
    M = M.copy_to_host()
    return M

def solve(n, M):
    # rozwiązuje macierz w postaci trójkątnej
    for i in range(0, n):
        for j in range(i - 1, -1, -1):
            w = M[j][i] / M[i][i]
            for k in range(i, n + 1):
                M[j][k] -= w * M[i][k]

    for i in range(0, n):
        w = M[i][i]
        for j in range(0, n + 1):
            M[i][j] /= w

    return M

def load_input():
    # odczytuje wejście z pliku in.txt
    with open("in.txt", "r") as file:
        n = int(file.readline())
        M = [[0 for _ in range(0, n + 1)] for _ in range(0, n)]
        for i in range(0, n):
            w = file.readline()
            w = w.split(' ')
            for j in range(0, n): M[i][j] = float(w[j])

        w = file.readline()
        w = w.split(' ')
        for k in range(0, n): M[k][n] = float(w[k])

    return n, M

def save_input(n, M):
    # zapisuje wejście do pliku out.txt
    with open("out.txt", "w") as file:
        file.write(str(n) + '\n')
        for i in range(0, n):
            for j in range(0, n): file.write(str(M[i][j]) + ' ')
            file.write('\n')

        for i in range(0, n): 
            file.write(str(M[i][n]) + ' ')
        file.write('\n')

# --- obliczenia

start_time_all = time.time()

n, M = load_input()

if print_full_time: print("Matrix size n:", n)
else: print_full_time: print(n, end = ' ')

M = gauss(n, M)

M = solve(n, M)

save_input(n, M)

end_time_all = time.time()

if print_full_time: print("All done in", round(end_time_all - start_time_all, 5), "seconds.") 
else: print(f"{(end_time_all - start_time_all):.5f}") 
