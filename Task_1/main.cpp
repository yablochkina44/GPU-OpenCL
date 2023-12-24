#include <CL/cl.h>
#include <iostream>
#pragma warning(disable : 4996)

#define BLOCK_SIZE 4
#define GLOBAL_SIZE 20
#define DATA_SIZE 20

void task_1(int* data, size_t data_size) {
    const char* source =
        "__kernel void hello("
        "        __global int* A,"
        "        unsigned int size"
        "    ) {"
        "        int n = get_group_id(0);"
        "        int m = get_local_id(0);"
        "        int k = get_global_id(0);"
        "        printf(\"I am from %d block, %d thread "
        "(global index: %d)\\n\", n, m, k);"
        "        if (k < size)"
        "            A[k] = A[k] + k;"
        "    }";
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
    cl_kernel kernel = clCreateKernel(program, "hello", nullptr);
    // ---------------------------------------------------
    // -------------------- Work zone --------------------
    cl_mem A = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
        sizeof(int) * GLOBAL_SIZE, data, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &A);
    clSetKernelArg(kernel, 1, sizeof(unsigned), &data_size); //¿–√”Ã≈Õ“€ ƒÀﬂ «¿œ”— ¿ ﬂƒ–¿

    size_t local_work_size[] = { BLOCK_SIZE };
    size_t global_work_size[] = { GLOBAL_SIZE };

    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_work_size, //‘”Õ ÷»ﬂ ƒÀﬂ œŒ—“¿ÕŒ¬ » «¿œ”— ¿ ﬂƒ–¿ ¬ Œ◊≈–≈ƒ‹  ŒÃ¿Õƒ
        local_work_size, 0, nullptr, nullptr);

    clEnqueueReadBuffer(queue, A, CL_TRUE, 0, sizeof(int) * GLOBAL_SIZE,
        data, 0, nullptr, nullptr);

    clReleaseMemObject(A);
    // -------------------- Work zone --------------------
    // ---------------------------------------------------
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main() {
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);

    cl_platform_id* platform = new cl_platform_id[platformCount];
    clGetPlatformIDs(platformCount, platform, nullptr);

    for (cl_uint i = 0; i < platformCount; ++i) {
        char platformName[128];
        clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 128, platformName,
            nullptr);
        std::cout << platformName << std::endl;
    }

    size_t data_size = DATA_SIZE;
    int* data = new int[data_size];

    std::cout << "Using:\n  Global size: " << GLOBAL_SIZE
        << "\n  Block size: " << BLOCK_SIZE
        << "\n  Vector size: " << data_size << '\n';

    std::cout << "  Vector data:\n    ";
    for (auto i = 0; i < data_size; i++) {
        data[i] = 10;
        std::cout << data[i] << ' ';
    }
    std::cout << "\n\n";

    task_1(data, data_size);

    std::cout << "\nUpdated vector data:\n  ";
    for (auto i = 0; i < data_size; i++)
        std::cout << data[i] << ' ';
    std::cout << '\n';

    delete[] data;

    return 0;
}
