#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#pragma warning(disable : 4996)

#define SIZE_VECTOR 0x080000
#define START_BLOCK_SIZE 8
#define END_BLOCK_SIZE 256
#define STEP_BLOCK_SIZE 2

bool eq_vector(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (auto i = 0; i < a.size(); i++)
        if (abs(a[i] - b[i]) > 5e-3)
            return false;
    return true;
}

template <typename T>
double task_2(int n, T a, T* x, int incx, T* y, int incy,
        const char* source, const char* func_name, cl_device_type type,
        size_t* local_work_size = nullptr) {
    cl_int error = 0;
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
        type, nullptr, nullptr, &error);
    if (error) {  // Error if type = CL_DEVICE_TYPE_CPU
        clReleaseContext(context);
        return static_cast<double>(error);
    }
    size_t numDevices = 0;
    error = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr,
        &numDevices);
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
    error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (error) {  // Error if T = double
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return static_cast<double>(error);
    }
    cl_kernel kernel = clCreateKernel(program, func_name, nullptr);
    // ---------------------------------------------------
    // -------------------- Work zone --------------------
    cl_mem X = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
        sizeof(T) * n, x, nullptr);
    cl_mem Y = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
        sizeof(T) * n, y, nullptr);

    clSetKernelArg(kernel, 0, sizeof(int), &n);
    clSetKernelArg(kernel, 1, sizeof(T), &a);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &X); //¿–√”Ã≈Õ“€ ƒÀﬂ «¿œ”— ¿ ﬂƒ–¿
    clSetKernelArg(kernel, 3, sizeof(int), &incx);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &Y);
    clSetKernelArg(kernel, 5, sizeof(int), &incy);

    size_t N = static_cast<size_t>(n);

    volatile double start = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &N, //‘”Õ ÷»ﬂ ƒÀﬂ œŒ—“¿ÕŒ¬ » «¿œ”— ¿ ﬂƒ–¿ ¬ Œ◊≈–≈ƒ‹  ŒÃ¿Õƒ
        local_work_size, 0, nullptr, nullptr);
    volatile double stop = omp_get_wtime();

    clEnqueueReadBuffer(queue, Y, CL_TRUE, 0, sizeof(T) * n,
        y, 0, nullptr, nullptr);

    clReleaseMemObject(X);
    clReleaseMemObject(Y);
    // -------------------- Work zone --------------------
    // ---------------------------------------------------
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return stop - start;
}

double saxpy_seq(int n, float a,
        float* x, int incx,
        float* y, int incy) {
    volatile double start = omp_get_wtime();
    for (int i = 0; i < n && i * incx < n && i * incy < n; i++)
        y[i * incy] += a * x[i * incx];
    volatile double stop = omp_get_wtime();
    return stop - start;
}

double daxpy_seq(int n, double a,
        double* x, int incx,
        double* y, int incy) {
    volatile double start = omp_get_wtime();
    for (int i = 0; i < n && i * incx < n && i * incy < n; i++)
        y[i * incy] += a * x[i * incx];
    volatile double stop = omp_get_wtime();
    return stop - start;
}

double saxpy_omp(int n, float a,
        float* x, int incx,
        float* y, int incy) {
    volatile double start = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        if (i * incx < n && i * incy < n)
            y[i * incy] += a * x[i * incx];
    }
    volatile double stop = omp_get_wtime();
    return stop - start;
}

double daxpy_omp(int n, double a,
        double* x, int incx,
        double* y, int incy) {
    volatile double start = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        if (i * incx < n && i * incy < n)
            y[i * incy] += a * x[i * incx];
    }
    volatile double stop = omp_get_wtime();
    return stop - start;
}

double saxpy_gpu(int n, float a, //ÙÛÂÍˆËˇ ‰Îˇ ÙÎÓ‡Ú ‚˚ÔÓÎÌ Ì‡ „ÔÛ
        float* x, int incx, float* y, int incy,
        size_t* local_work_size = nullptr) {
    const char* source =
        "__kernel void saxpy_gpu("
        "        int n,"
        "        float a,"
        "        __global float* x,"
        "        int incx,"
        "        __global float* y,"
        "        int incy"
        "    ) {"
        "    int i = get_global_id(0);"\
        "    if (i < n && i * incx < n && i * incy < n)"
        "        y[i * incy] += a * x[i * incx];"
        "}";
    return task_2<float>(n, a, x, incx, y, incy, source, "saxpy_gpu",
        CL_DEVICE_TYPE_GPU);
}

double daxpy_gpu(int n, double a, //ÙÛÌÍˆËˇ ‰Îˇ ‰‡·Î ‚˚ÔÎÌˇ˛˘‡ˇ ‚˚˜ËÒÎÂÌËˇ Ì‡ „ÔÛ
        double* x, int incx, double* y, int incy,
        size_t* local_work_size = nullptr) {
    const char* source =
        "__kernel void daxpy_gpu("
        "        int n,"
        "        double a,"
        "        __global double* x,"
        "        int incx,"
        "        __global double* y,"
        "        int incy"
        "    ) {"
        "    int i = get_global_id(0);"\
        "    if (i < n && i * incx < n && i * incy < n)"
        "        y[i * incy] += a * x[i * incx];"
        "}";
    return task_2<double>(n, a, x, incx, y, incy, source, "daxpy_gpu",
        CL_DEVICE_TYPE_GPU);
}

double saxpy_cpu(int n, float a,
        float* x, int incx, float* y, int incy,
        size_t* local_work_size = nullptr) {
    const char* source =
        "__kernel void saxpy_cpu("
        "        int n,"
        "        float a,"
        "        __global float* x,"
        "        int incx,"
        "        __global float* y,"
        "        int incy"
        "    ) {"
        "    int i = get_global_id(0);"\
        "    if (i < n && i * incx < n && i * incy < n)"
        "        y[i * incy] += a * x[i * incx];"
        "}";
    return task_2<float>(n, a, x, incx, y, incy, source, "saxpy_cpu",
        CL_DEVICE_TYPE_CPU);
}

double daxpy_cpu(int n, double a,
        double* x, int incx, double* y, int incy,
        size_t* local_work_size = nullptr) {
    const char* source =
        "__kernel void daxpy_cpu("
        "        int n,"
        "        double a,"
        "        __global double* x,"
        "        int incx,"
        "        __global double* y,"
        "        int incy"
        "    ) {"
        "    int i = get_global_id(0);"\
        "    if (i < n && i * incx < n && i * incy < n)"
        "        y[i * incy] += a * x[i * incx];"
        "}";
    return task_2<double>(n, a, x, incx, y, incy, source, "daxpy_cpu", CL_DEVICE_TYPE_CPU);
}

int main() {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<> urd_real(1, 255);
    std::uniform_int_distribution<> urd_int(1, 10);

    int size_vector = SIZE_VECTOR;
    std::cout << "Vector size: " << size_vector << "\nGenerate data... ";

    float a = static_cast<float>(urd_real(gen));
    int incx = urd_int(gen);
    int incy = urd_int(gen);

    std::vector<float> x(size_vector);
    std::vector<float> y_default(size_vector);
    std::vector<float> y_seq(size_vector);
    std::vector<float> y_gpu(size_vector);
    std::vector<float> y_omp(size_vector);

#pragma omp parallel for
    for (int i = 0; i < size_vector; i++) {
        x[i] = static_cast<float>(urd_real(gen));
        y_default[i] = y_seq[i] = y_gpu[i] = y_omp[i] =
            static_cast<float>(urd_real(gen));
    }

   // std::cout << "done!\n\nTimes:\n";

    //volatile double start_seq = omp_get_wtime();
    double seq_time_math = saxpy_seq(size_vector, a,
        x.data(), incx, y_seq.data(), incy);
    volatile double stop_seq = omp_get_wtime();
    //std::cout << "  SEQ: " << stop_seq - start_seq << '\n';

    //volatile double start_gpu = omp_get_wtime();
    double gpu_time_math = saxpy_gpu(size_vector, a,
        x.data(), incx, y_gpu.data(), incy);
    //volatile double stop_gpu = omp_get_wtime();
    //std::cout << "  GPU: " << stop_gpu - start_gpu << '\n';

   // volatile double start_omp = omp_get_wtime();
    double omp_time_math = saxpy_omp(size_vector, a,
        x.data(), incx, y_omp.data(), incy);
    //volatile double stop_omp = omp_get_wtime();
    //std::cout << "  OMP: " << stop_omp - start_omp << '\n';

    std::cout << "\nResults:"
        << "\n  SEQ: " << seq_time_math
        << "\n  GPU: " << gpu_time_math
        << "\n  OMP: " << omp_time_math
        << "\n\nCheck eq:";
    std::cout << "\n  SEQ != DEFAULT: "
        << (!eq_vector(y_default, y_seq) ? "true" : "false");
    std::cout << "\n  SEQ = GPU: "
        << (eq_vector(y_seq, y_gpu) ? "true" : "false");
    std::cout << "\n  SEQ = OMP: "
        << (eq_vector(y_seq, y_omp) ? "true" : "false");
    std::cout << "\n\nGPU results with block size:\n";
    double best_time = 100.0;
    size_t best_block_size = START_BLOCK_SIZE;
    for (size_t block_size = START_BLOCK_SIZE; block_size <= END_BLOCK_SIZE;
            block_size *= STEP_BLOCK_SIZE) {
        y_gpu = y_default;
        double time = saxpy_gpu(size_vector, a,
            x.data(), incx, y_gpu.data(), incy, &block_size);
        if (time < best_time) {
            best_time = time;
            best_block_size = block_size;
        }
        std::cout << "  " << block_size << ": " << time << '\n';
    }
    std::cout << "Now the best timing is " << best_time << " with block size "
        << best_block_size << '\n';

    

    return 0;
}
