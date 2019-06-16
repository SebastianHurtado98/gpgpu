#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#define NUM_FILES 2
#define PROGRAM_FILE_1 "good.cl"
#define PROGRAM_FILE_2 "bad.cl"

int main() {
    cl_platform_id platform;
    cl_device_id *devices;
    cl_uint num_devices, addr_data;
    cl_int i, err;
    char name_data[48], ext_data[4096];
    cl_context context;
    cl_uint ref_count;
    cl_program program;
    FILE *program_handle;
    char *program_buffer[NUM_FILES];
    char *program_log;
    const char *file_name[] = {PROGRAM_FILE_1, PROGRAM_FILE_2};
    const char options[] = "-cl-finite-math-only -cl-no-signed-zeros";
    size_t program_size[NUM_FILES];
    size_t log_size;

    err = clGetPlatformIDs(1, &platform, NULL);
    if(err < 0) {
        perror("Couldn't find any platforms");
        exit(1);
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                         1, NULL, &num_devices);
    if(err < 0) {
        perror("Couldn't find any devices");
        exit(1);
    }
    devices = (cl_device_id*)
            malloc(sizeof(cl_device_id) * num_devices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                   num_devices, devices, NULL);



    context = clCreateContext(NULL, 1, devices,
                              NULL, NULL, &err);
    err = clGetContextInfo(context,
                           CL_CONTEXT_REFERENCE_COUNT,
                           sizeof(ref_count), &ref_count, NULL);
    if(err < 0) {
        perror("Couldn't read the reference count.");
        exit(1);
    }

    for(i=0; i<NUM_FILES; i++) {
        program_handle = fopen(file_name[i], "r");
        if(program_handle == NULL) {
            perror("Couldn't find the program file");
            exit(1);
        }
        fseek(program_handle, 0, SEEK_END);
        program_size[i] = ftell(program_handle);
        rewind(program_handle);
        program_buffer[i] = (char*)malloc(program_size[i]+1);
        program_buffer[i][program_size[i]] = '\0';
        fread(program_buffer[i], sizeof(char),
        program_size[i], program_handle);
        fclose(program_handle);
    }
    program = clCreateProgramWithSource(context, NUM_FILES,
                                        (const char**)program_buffer, program_size, &err);
    if(err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    err = clBuildProgram(program, 1, devices,
                         options, NULL, NULL);
    if(err < 0) {
        clGetProgramBuildInfo(program, *devices,
                              CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*)malloc(log_size+1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, *devices,
                              CL_PROGRAM_BUILD_LOG,
                              log_size+1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }
    for(i=0; i<NUM_FILES; i++) {
        free(program_buffer[i]);
    }


}