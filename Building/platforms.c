#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
int main() {
    cl_platform_id *platforms;
    cl_uint num_platforms;
    cl_int i, err, platform_index = -1;
    char* ext_data;
    size_t ext_size;
    const char icd_ext[] = "cl_khr_icd";

    //Find number of platforms
    err = clGetPlatformIDs(1, NULL, &num_platforms);
    if(err < 0) {
        perror("Couldn't find any platforms.");
        exit(1);
    }

    printf("You have %i platforms\n", num_platforms);


    //Allocate space for platforms
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * num_platforms);

    // (Number of platforms to add, array, platforms available)
    clGetPlatformIDs(num_platforms, platforms, NULL);

    for(i=0; i<num_platforms; i++) {

        //(id platform, Info being queried, bits being queried from data, data, bits available)
        err = clGetPlatformInfo(platforms[i],
                                CL_PLATFORM_EXTENSIONS, 0, NULL, &ext_size);
        if(err < 0) {
            perror("Couldn't read extension data.");
            exit(1);
        }

        //Allocate space for data
        ext_data = (char*)malloc(ext_size);

        //Get info
        clGetPlatformInfo(platforms[i],
                          CL_PLATFORM_EXTENSIONS,
                          ext_size, ext_data, NULL);


        printf("Platform %d supports extensions: %s\n",
               i, ext_data);

        //this platforms supports cl_khr_icd extension?
        if(strstr(ext_data, icd_ext) != NULL) {
            free(ext_data);
            platform_index = i;
            break;
        }
        free(ext_data);
    }
    if(platform_index > -1)
        printf("Platform %d supports the %s extension.\n",
               platform_index, icd_ext);
    else
        printf("No platforms support the %s extension.\n", icd_ext);
    free(platforms);
    return 0;
}