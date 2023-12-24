#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>

#pragma warning(disable : 4996)  // deprecate error (clCreateCommandQueue)
#pragma warning(disable : 4849)  // collapse
#pragma warning(disable : 6993)  // omp

#define BLOCK_SIZE 16
#define M_SIZE 2000
#define N_SIZE 1600
#define K_SIZE 2400

// A = M x N
// B = N x K
// C = M x K

std::string eq_vector(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return "false";
    for (auto i = 0; i < a.size(); i++)
        if (abs(a[i] - b[i]) > 5e-3) return "false";
    return "true";
}

std::vector<float> gen_rand_vector(int row, int col) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<> urd(-10, 10);
    int size = row * col;
    std::vector<float> result(size);
    for (int i = 0; i < size; i++)
        result[i] = static_cast<float>(urd(gen));
    return result;
}

double task_3(const char* source, const char* kernel_name,
    float* a, float* b, float* c, int M, int N, int K) {
    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    cl_platform_id platform = nullptr;
    if (numPlatforms > 0) {
        cl_platform_id* platforms = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(numPlatforms, platforms, nullptr);
        platform = platforms[0];
        delete[] platforms;
    }
    cl_context_properties properties[3] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_context context = clCreateContextFromType(
        (platform == nullptr) ? nullptr : properties,
        CL_DEVICE_TYPE_GPU, nullptr, nullptr, nullptr);
    size_t numDevices = 0;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &numDevices);
    cl_device_id device = nullptr;
    if (numDevices > 0) {
        cl_device_id* devices = new cl_device_id[numDevices];
        clGetContextInfo(context, CL_CONTEXT_DEVICES, numDevices, devices,
            nullptr);
        device = devices[0];
        delete[] devices;
    }
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);
    size_t srclen[] = { strlen(source) };
    cl_program program = clCreateProgramWithSource(context, 1, &source, srclen,
        nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, kernel_name, nullptr);
    // ---------------------------------------------------
    // -------------------- Work zone --------------------
    volatile double start = omp_get_wtime();
    cl_mem A = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
        sizeof(float) * M * N, a, nullptr);
    cl_mem B = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
        sizeof(float) * N * K, b, nullptr);
    cl_mem C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * M * K, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &C);
    clSetKernelArg(kernel, 3, sizeof(int), &M);
    clSetKernelArg(kernel, 4, sizeof(int), &N);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    size_t local_work_size[] = { BLOCK_SIZE, BLOCK_SIZE };
    size_t global_work_size[] = { static_cast<size_t>(M), static_cast<size_t>(K) };
    if (kernel_name == "mul_gpu_2") {
        global_work_size[0] = static_cast<size_t>(std::max(M, K));
        global_work_size[1] = static_cast<size_t>(std::max(M, K));
    }

    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work_size,
        local_work_size, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, C, CL_TRUE, 0,
        sizeof(float) * M * K, c, 0, nullptr, nullptr);

    clReleaseMemObject(A);
    clReleaseMemObject(B);
    clReleaseMemObject(C);
    volatile double stop = omp_get_wtime();
    // -------------------- Work zone --------------------
    // ---------------------------------------------------
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return stop - start;
}

double mul_mtrx_gpu_0(
    float* a, float* b, float* c, int M, int N, int K) {
    const char* source =
        "__kernel void mul_gpu_0("
        "        __global float* a,"
        "        __global float* b,"
        "        __global float* c,"
        "        int M, int N, int K) {"
        "    return;"
        "}";
    return task_3(source, "mul_gpu_0", a, b, c, M, N, K);
}

double mul_mtrx_seq(
    float* a, float* b, float* c, int M, int N, int K) {
    volatile double start = omp_get_wtime();
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            for (int k = 0; k < N; k++)
                c[i * K + j] += a[i * N + k] * b[k * K + j];
    volatile double stop = omp_get_wtime();
    return stop - start;
}

double mul_mtrx_omp(
    float* a, float* b, float* c, int M, int N, int K) {
    volatile double start = omp_get_wtime();
    float sum = 0;
#pragma omp parallel for reduction(+:sum) collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            sum = 0;
            for (int k = 0; k < N; k++)
                sum += a[i * N + k] * b[k * K + j];
            c[i * K + j] = sum;
        }
    }
    volatile double stop = omp_get_wtime();
    return stop - start;
}

double mul_mtrx_gpu_1(
    float* a, float* b, float* c, int M, int N, int K) {
    const char* source =
        "__kernel void mul_gpu_1("
        "        __global float* a,"
        "        __global float* b,"
        "        __global float* c,"
        "        int M, int N, int K) {"
        "    const int BS = 16;"
        "    int row = get_group_id(0) * BS + get_local_id(1);"
        "    int col = get_group_id(1) * BS + get_local_id(0);"
        "    float sum = 0;"
        "    for (int k = 0; k < N; k++)"
        "        sum += a[row * N + k] * b[k * K + col];"
        "    c[row * K + col] = sum;"
        "}";
    return task_3(source, "mul_gpu_1", a, b, c, M, N, K);
}

double mul_mtrx_gpu_2(
    float* a, float* b, float* c, int M, int N, int K) {
    const char* source =
        "__kernel void mul_gpu_2("
        "        __global float* a,"
        "        __global float* b,"
        "        __global float* c,"
        "        int M, int N, int K) {"
        "    const int BS = 16;"
        "    int col = get_group_id(0) * BS + get_local_id(0);"
        "    int row = get_group_id(1) * BS + get_local_id(1);"
        "    if (col < K && row < M) {"
        "        float sum = 0;"
        "        for (int k = 0; k < N; k++)"
        "            sum += a[row * N + k] * b[k * K + col];"
        "        c[row * K + col] = sum;"
        "    }"
        "}";
    return task_3(source, "mul_gpu_2", a, b, c, M, N, K);
}

double mul_mtrx_gpu_3(
    float* a, float* b, float* c, int M, int N, int K) {
    const char* source =
        "__kernel void mul_gpu_3("
        "        __global float* a,"
        "        __global float* b,"
        "        __global float* c,"
        "        int M, int N, int K) {"
        "    const int BS = 16;"
        "    __local float locA[BS][BS];"
        "    __local float locB[BS][BS];"
        "    int bRow = get_group_id(0) * BS;"
        "    int bCol = get_group_id(1) * BS;"
        "    int row = get_local_id(1);"
        "    int col = get_local_id(0);"
        "    float sum = 0;"
        "    for (int m = 0; m < N; m += BS) {"
        "        locA[row][col] = a[(bRow + row) * N + m + col];"
        "        locB[row][col] = b[(m + row) * K + bCol + col];"
        "        barrier(CLK_LOCAL_MEM_FENCE);"
        "        for (int e = 0; e < BS; e++)"
        "            sum += locA[row][e] * locB[e][col];"
        "        barrier(CLK_LOCAL_MEM_FENCE);"
        "    }"
        "    c[K * bRow + bCol + K * row + col] = sum;"
        "}";
    return task_3(source, "mul_gpu_3", a, b, c, M, N, K);
}

int main() {
    int M = M_SIZE, N = N_SIZE, K = K_SIZE;

    std::cout << "Matrix size:\n" << "  A: " << M << " x " << N << "\n  B: "
        << N << " x " << K << "\n  C: " << M << " x " << K << "\nGenerate data... ";

    std::vector<float> a = gen_rand_vector(M, N);
    std::vector<float> b = gen_rand_vector(N, K);
    std::vector<float> c_seq(M * K);
    std::vector<float> c_omp(M * K);
    std::vector<float> c_gpu_1(M * K);
    std::vector<float> c_gpu_2(M * K);
    std::vector<float> c_gpu_3(M * K);
    mul_mtrx_gpu_0(a.data(), b.data(), c_seq.data(), M, N, K);

    std::cout << "done!\n\nResults:";
    std::cout << "\n  SEQ: " << mul_mtrx_seq(a.data(), b.data(), c_seq.data(), M, N, K);
    std::cout << "\n  OMP: " << mul_mtrx_omp(a.data(), b.data(), c_omp.data(), M, N, K);
    std::cout << "\n  GPU_1: " << mul_mtrx_gpu_1(a.data(), b.data(), c_gpu_1.data(), M, N, K);
    std::cout << "\n  GPU_2: " << mul_mtrx_gpu_2(a.data(), b.data(), c_gpu_2.data(), M, N, K);
    std::cout << "\n  GPU_3: " << mul_mtrx_gpu_3(a.data(), b.data(), c_gpu_3.data(), M, N, K);

    return 0;
}
