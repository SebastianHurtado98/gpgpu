#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
int main() {
    cl_platform_id platform;
    cl_device_id *devices;
    cl_uint num_devices, addr_data;
    cl_int i, err;
    char name_data[48], ext_data[4096];
    cl_context context;
    cl_uint ref_count;

    err = clGetPlatformIDs(1, &platform, NULL);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
                         1, NULL, &num_devices);

    if(err < 0) {
        perror("Couldn't find any devices");
        exit(1);
    }

    devices = (cl_device_id*) malloc(sizeof(cl_device_id) * num_devices);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,
                   num_devices, devices, NULL);

    context = clCreateContext(NULL, num_devices, devices,
                              NULL, NULL, &err);

    if(err < 0) {
        perror("Couldn't create a context");
        exit(1);
    }

    err = clGetContextInfo(context,
                           CL_CONTEXT_REFERENCE_COUNT,
                           sizeof(ref_count), &ref_count, NULL);
    if(err < 0) {
        perror("Couldn't read the reference count.");
        exit(1);
    }
    printf("Initial reference count: %u\n", ref_count);

    clRetainContext(context);
    clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT,
                     sizeof(ref_count), &ref_count, NULL);
    printf("Reference count: %u\n", ref_count);
    clReleaseContext(context);
    clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT,
                     sizeof(ref_count), &ref_count, NULL);
    printf("Reference count: %u\n", ref_count);
    clReleaseContext(context);

}