#include <iostream>
#include <typeinfo>
#include <string>
#include <fstream>
#include <chrono>
#include <utility>
#include <string>
#include <vector>
#include <array>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <sstream>

#include <cuda_runtime.h>
#include "cusparse.h"


#define err std::cerr
#define puts std::cout
#define br std::endl
#define ti(x) std::stoi(argv[x])


cusparseHandle_t handle;


typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

template<typename T> struct test_data{
    int num_rows;
    int num_cols;
    int nnz;
    T* values;
    int* cols;
    int* rptr;
    T a;
    T b;
    T* x;
    T* y;
    T* solution;
};

template<typename F, typename... Args>
double funcTime(F func, Args&&... args){
    TimeVar t1=timeNow();
    func(std::forward<Args>(args)...);
    double elapsed = duration(timeNow()-t1);
    return elapsed;
}


/*
 * nnz rows cols  a  b
 * data[nnz]
 * col_indexes[nnz]
 * rowptrs[rows + 1]
 * x[cols]
 * y[rows]
 * solution[rows]
 */
template<typename T> void generate_data(const char* datafile_path, std::vector<struct test_data<T>> &matrixes, int reps){
    std::ifstream file(datafile_path, std::ios::in);

    for(int i = 0; i < reps; i++){
        std::string currentline;
        std::getline(file, currentline);
        std::stringstream tss(currentline);

        int nnz, rows, cols;
        T a;
        T b;

        //nnz << tss;
        //rows << tss;
        //cols << tss;
        //a << tss;
        //b << tss;
        tss >> nnz >> cols >> rows >> a >> b;

#ifdef debug
        puts << nnz << " " << rows << " " << cols << " " << a << " " << b << br;
#endif


        T* data = NULL;
        int* col_indexes = NULL;
        int* rowptrs = NULL;
        T* x = NULL;
        T* y = NULL;
        T* solution = NULL;

        std::array<int, 6> memerrors;
#define ALIGNMENT 1024
        memerrors[0] = posix_memalign((void**) &data,        ALIGNMENT, sizeof(T)   * nnz);
        memerrors[1] = posix_memalign((void**) &col_indexes, ALIGNMENT, sizeof(int) * nnz);
        memerrors[2] = posix_memalign((void**) &rowptrs,     ALIGNMENT, sizeof(int) * rows + 1);
        memerrors[3] = posix_memalign((void**) &x,           ALIGNMENT, sizeof(T)   * cols);
        memerrors[4] = posix_memalign((void**) &y,           ALIGNMENT, sizeof(T)   * rows);
        memerrors[5] = posix_memalign((void**) &solution,    ALIGNMENT, sizeof(T)   * rows);

        if(std::any_of(memerrors.begin(), memerrors.end(), [](int i){return i != 0;})){
            std::cerr << "Error allocaling aligned memory" << br;
            exit(-1);
        }

        std::getline(file, currentline);
        tss.clear();
        tss.str(currentline);
        for(int j = 0; j < nnz; j++){
            tss >> data[j];
#ifdef debug
            puts << data[j] << " ";
#endif
        }

#ifdef debug
        puts << br;
#endif

        std::getline(file, currentline);
        tss.clear();
        tss.str(currentline);
        for(int j = 0; j < nnz; j++){
            tss >> col_indexes[j];
#ifdef debug
            puts << col_indexes[j] << " ";
#endif
        }

#ifdef debug
        puts << br;
#endif

        std::getline(file, currentline);
        tss.clear();
        tss.str(currentline);
        for(int j = 0; j < rows + 1; j++){
            tss >> rowptrs[j];
#ifdef debug
            puts << rowptrs[j] << " ";
#endif
        }

#ifdef debug
        puts << br;
#endif

        std::getline(file, currentline);
        tss.clear();
        tss.str(currentline);
        for(int j = 0; j < cols; j++){
            tss >> x[j];
#ifdef debug
            puts << x[j] << " ";
#endif
        }

#ifdef debug
        puts << br;
#endif

        std::getline(file, currentline);
        tss.clear();
        tss.str(currentline);
        for(int j = 0; j < rows; j++){
            tss >> y[j];
#ifdef debug
            puts << y[j] << " ";
#endif
        }

#ifdef debug
        puts << br;
#endif

        std::getline(file, currentline);
        tss.clear();
        tss.str(currentline);
        for(int j = 0; j < rows; j++){
            tss >> solution[j];
#ifdef debug
            puts << solution[j] << " ";
#endif
        }

        struct test_data<T> td = {
                rows,
                cols,
                nnz,
                data,
                col_indexes,
                rowptrs,
                a,
                b,
                x,
                y,
                solution
        };

        matrixes.push_back(td);
    }
}

template <typename T> bool check(T* cpu_output, T* cl_output, int l){
    bool correct = true;
    if (typeid(T) == typeid(int)){
        correct = std::equal(cpu_output, cpu_output + l, cl_output);
    }else{
        for(int j = 0; j < l; j++){
            if(fabs(cpu_output[j] - cl_output[j]) > fabs(cpu_output[j] * 0.001)){
                correct = false;
                puts << "Wrong. Output " << cl_output[j] << " should be "
                     << cpu_output[j] << "Difference " << fabs(cpu_output[j] - cl_output[j]) <<
                     "Threshold " << cpu_output[j] * 0.001 << br;
                break;
            }
        }
    }
    return correct;
}

#define T float
void run_test(std::vector<struct test_data<T>> &matrices, std::string devinfo, const int internal_reps,  const char* datafile_path){
    cudaError_t cudaStat;
    cusparseStatus_t status;


    /* initialize cusparse library */
    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("CUSPARSE Library initialization failed");
        exit(-1);
    }
    for(int i = 0; i < matrices.size(); i++) {
        for (int j = 0; j < internal_reps; j++) {
            struct test_data<T> test = matrices[i];
            T* gpu_values;
            T* gpu_x;
            T* gpu_y;
            int* gpu_col_ind;
            int* gpu_row_ofset;

            cusparseMatDescr_t descrA;
            cusparseCreateMatDescr(&descrA);

            cudaStat = cudaMalloc ((void**)&gpu_values, test.nnz*sizeof(T));
            if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed");
                exit(-1);
            }

            cudaStat = cudaMalloc ((void**)&gpu_x, test.num_cols*sizeof(T));
            if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed");
                exit(-1);
            }

            cudaStat = cudaMalloc ((void**)&gpu_y, test.num_rows*sizeof(T));
            if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed");
                exit(-1);
            }

            cudaStat = cudaMalloc ((void**)&gpu_col_ind, test.nnz*sizeof(int));
            if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed");
                exit(-1);
            }

            cudaStat = cudaMalloc ((void**)&gpu_row_ofset, (test.num_rows + 1)*sizeof(int));
            if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed");
                exit(-1);
            }


            cudaStat = cudaMemcpy(gpu_values, test.values, test.nnz*sizeof(T), cudaMemcpyHostToDevice);
            if (cudaStat != cudaSuccess) {
                printf ("mem copy 1 failed");
                exit(-1);
            }

            cudaStat = cudaMemcpy(gpu_x, test.x, test.num_cols*sizeof(T), cudaMemcpyHostToDevice);
            if (cudaStat != cudaSuccess) {
                printf ("mem copy 1 failed");
                exit(-1);
            }

            cudaStat = cudaMemcpy(gpu_y, test.y, test.num_rows*sizeof(T), cudaMemcpyHostToDevice);
            if (cudaStat != cudaSuccess) {
                printf ("mem copy 1 failed");
                exit(-1);
            }

            cudaStat = cudaMemcpy(gpu_col_ind, test.cols, test.nnz*sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStat != cudaSuccess) {
                printf ("mem copy 1 failed");
                exit(-1);
            }

            cudaStat = cudaMemcpy(gpu_row_ofset, test.rptr, (test.num_rows + 1)*sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStat != cudaSuccess) {
                printf ("mem copy 1 failed");
                exit(-1);
            }

            double elapsed = -1;
            if(typeid(T) == typeid(float)){
                elapsed = funcTime(cusparseScsrmv, handle, CUSPARSE_OPERATION_NON_TRANSPOSE, test.num_rows, test.num_cols, test.nnz, &test.a, descrA,
                                   gpu_values, gpu_row_ofset, gpu_col_ind, gpu_x, &test.b, gpu_y);
            }


            T result[test.num_rows];

            cudaMemcpy(result, gpu_y, test.num_rows*sizeof(T), cudaMemcpyDeviceToHost);

bool correct = check(test.solution, result, test.num_rows);

            puts << datafile_path << "\t" << test.num_rows << "\t" << test.num_cols << "\t" << test.nnz << "\t" << elapsed << "\t" << correct
                 << br;

            cudaFree((void*)&gpu_values);
            cudaFree((void*)&gpu_x);
            cudaFree((void*)&gpu_y);
            cudaFree((void*)&gpu_col_ind);
            cudaFree((void*)&gpu_row_ofset);
        }
    }

}

#define T double
void run_test(std::vector<struct test_data<T>> &matrices, std::string devinfo, const int internal_reps,  const char* datafile_path){
    cudaError_t cudaStat;
    cusparseStatus_t status;


    /* initialize cusparse library */
    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("CUSPARSE Library initialization failed");
        exit(-1);
    }
    for(int i = 0; i < matrices.size(); i++) {
        for (int j = 0; j < internal_reps; j++) {
            struct test_data<T> test = matrices[i];
            T* gpu_values;
            T* gpu_x;
            T* gpu_y;
            int* gpu_col_ind;
            int* gpu_row_ofset;

            cusparseMatDescr_t descrA;
            cusparseCreateMatDescr(&descrA);

            cudaStat = cudaMalloc ((void**)&gpu_values, test.nnz*sizeof(T));
            if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed");
                exit(-1);
            }

            cudaStat = cudaMalloc ((void**)&gpu_x, test.num_cols*sizeof(T));
            if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed");
                exit(-1);
            }

            cudaStat = cudaMalloc ((void**)&gpu_y, test.num_rows*sizeof(T));
            if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed");
                exit(-1);
            }

            cudaStat = cudaMalloc ((void**)&gpu_col_ind, test.nnz*sizeof(int));
            if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed");
                exit(-1);
            }

            cudaStat = cudaMalloc ((void**)&gpu_row_ofset, (test.num_rows + 1)*sizeof(int));
            if (cudaStat != cudaSuccess) {
                printf ("device memory allocation failed");
                exit(-1);
            }


            cudaStat = cudaMemcpy(gpu_values, test.values, test.nnz*sizeof(T), cudaMemcpyHostToDevice);
            if (cudaStat != cudaSuccess) {
                printf ("mem copy 1 failed");
                exit(-1);
            }

            cudaStat = cudaMemcpy(gpu_x, test.x, test.num_cols*sizeof(T), cudaMemcpyHostToDevice);
            if (cudaStat != cudaSuccess) {
                printf ("mem copy 1 failed");
                exit(-1);
            }

            cudaStat = cudaMemcpy(gpu_y, test.y, test.num_rows*sizeof(T), cudaMemcpyHostToDevice);
            if (cudaStat != cudaSuccess) {
                printf ("mem copy 1 failed");
                exit(-1);
            }

            cudaStat = cudaMemcpy(gpu_col_ind, test.cols, test.nnz*sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStat != cudaSuccess) {
                printf ("mem copy 1 failed");
                exit(-1);
            }

            cudaStat = cudaMemcpy(gpu_row_ofset, test.rptr, (test.num_rows + 1)*sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStat != cudaSuccess) {
                printf ("mem copy 1 failed");
                exit(-1);
            }

            double elapsed = -1;
            if(typeid(T) == typeid(double)){
                elapsed = funcTime(cusparseDcsrmv, handle, CUSPARSE_OPERATION_NON_TRANSPOSE, test.num_rows, test.num_cols, test.nnz, &test.a, descrA,
                                   gpu_values, gpu_row_ofset, gpu_col_ind, gpu_x, &test.b, gpu_y);
            }

            T result[test.num_rows];

            cudaMemcpy(result, gpu_y, test.num_rows*sizeof(T), cudaMemcpyDeviceToHost);

            bool correct = check(test.solution, result, test.num_rows);

            puts  << datafile_path << "\t" << test.num_rows << "\t" << test.num_cols << "\t" << test.nnz << "\t" << elapsed << "\t" << correct
                 << br;

            cudaFree((void*)&gpu_values);
            cudaFree((void*)&gpu_x);
            cudaFree((void*)&gpu_y);
            cudaFree((void*)&gpu_col_ind);
            cudaFree((void*)&gpu_row_ofset);
        }
    }

}

/*
 * datafile
 * reps
 * datatype
 * internal_reps
 */
int main(int argc, char* argv[]){
    const char* datafile_path = argv[1];
    const int reps = ti(2);
    std::string datatype;
    datatype = std::string(argv[3]);
    const int internal_reps = ti(4);



    if(datatype == "float") {
        std::vector<struct test_data<float>> matrixes;
        generate_data(datafile_path, matrixes, reps);
        run_test(matrixes, "1070", internal_reps, datafile_path);
    }
    if(datatype == "double") {
        std::vector<struct test_data<double>> matrixes;
        generate_data(datafile_path, matrixes, reps);
        run_test(matrixes, "1070", internal_reps, datafile_path);
    }
}