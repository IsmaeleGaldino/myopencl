#include "ocl_boiler.h"
#include <stdbool.h>

cl_event matrix_multiply(cl_program src, cl_command_queue q, cl_device_id d, cl_mem matrixA, cl_mem matrixB, cl_mem matrixR, unsigned int rowsA, unsigned int colsA , unsigned int colsB, const size_t lws[]);
cl_event matrix_generator(cl_program src, cl_command_queue q, cl_device_id d, cl_mem matrix, unsigned int rows, unsigned int cols);

int main (int argc,char * argv[]){
    if(argc != 6){
        printf("invalid parameters, syntax: matrix rowsA colsA colsB\n");
        exit(1);
    }
    cl_uint rowsA, colsA, colsB;
    const size_t lws [] = {strtoul(argv[4],NULL,10),strtoul(argv[5],NULL,10)};
    rowsA = atoi(argv[1]);
    colsA = atoi(argv[2]);
    colsB = atoi(argv[3]);


    if(rowsA == 0 || colsA == 0 || colsB == 0){
        printf("invalid input for sizes of matrixes\n");
        exit(2);
    }

    srand(time(NULL));

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p,d);
    cl_command_queue q = create_queue(ctx,d);
    cl_program src = create_program("matrix.ocl",ctx,d);

    cl_int err;    
    cl_mem matrixA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, rowsA*colsA*sizeof(cl_int), NULL, &err);
    ocl_check(err,"failed to create buffer for matrixA");

    cl_mem matrixB = clCreateBuffer(ctx, CL_MEM_READ_WRITE, colsA*colsB*sizeof(cl_int), NULL, &err);
    ocl_check(err,"failed to create buffer for matrixB");

    cl_mem matrixR = clCreateBuffer(ctx,CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,rowsA*colsB*sizeof(cl_int),NULL,&err);
    ocl_check(err,"failed to create buffer for matrixR");

    cl_event gen_matrix_a = matrix_generator(src,q,d,matrixA,rowsA,colsA);
    err = clWaitForEvents(1,&gen_matrix_a);
    ocl_check(err,"failed on waiting for matrix A generation");

    cl_event gen_matrix_b = matrix_generator(src,q,d,matrixB,colsA,colsB);
    err = clWaitForEvents(1,&gen_matrix_b);
    ocl_check(err,"failed on waiting for matrix B generation");

    cl_event matrix_mul;
    matrix_mul = matrix_multiply(src,q,d,matrixA,matrixB,matrixR,rowsA,colsA,colsB,lws);
    err = clWaitForEvents(1,&matrix_mul);
    ocl_check(err,"failed on waiting for matrix multiplication");
    printf("matrix multipliction: time = %gms , throughput = %gGE/s \n",runtime_ms(matrix_mul),(float)(rowsA*colsA*colsB)/runtime_ns(matrix_mul));

    cl_int * matrixRH = malloc(sizeof(cl_int)*rowsA*colsB);
    if(matrixRH == NULL){
        printf("error on malloc\n");
        exit(3);
    }
    cl_event readMatrix;
    matrixRH = clEnqueueMapBuffer(q,matrixR,CL_TRUE,CL_MAP_READ,0,sizeof(cl_int)*rowsA*colsB,0,NULL,&readMatrix,&err);
    ocl_check(err,"failed on reading matrix result");
    return 0;
}

cl_event matrix_multiply(cl_program src, cl_command_queue q, cl_device_id d, cl_mem matrixA, cl_mem matrixB, cl_mem matrixR, cl_uint rowsA, cl_uint colsA , cl_uint colsB,const size_t lws []){
    static bool init= true;
    static cl_kernel k;
    cl_int err;
    if(init){
        k = clCreateKernel(src,"matrix_multiplication",&err);
        ocl_check(err,"failed to load kernel: matrix multiplication");
        init = false;
    }

    size_t preferred_size;
    err = clGetKernelWorkGroupInfo(k,d,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,sizeof(preferred_size),&preferred_size,NULL);
    ocl_check(err,"failed to get KWG for multiply");
    
    printf("Preferred WGS for multiply: %u\n",preferred_size);

    err = clSetKernelArg(k,0,sizeof(matrixA),&matrixA);
	ocl_check(err,"failed to set first argument");

    err = clSetKernelArg(k,1,sizeof(matrixB),&matrixB);
	ocl_check(err,"failed to set second argument");

    err = clSetKernelArg(k,2,sizeof(matrixR),&matrixR);
	ocl_check(err,"failed to set third argument");

    err = clSetKernelArg(k,3,sizeof(rowsA),&rowsA);
	ocl_check(err,"failed to set fourth argument");

    err = clSetKernelArg(k,4,sizeof(colsA),&colsA);
	ocl_check(err,"failed to set fifth argument");

    err = clSetKernelArg(k,5,sizeof(colsB),&colsB);
	ocl_check(err,"failed to set sixth argument");

    cl_event event;
    const size_t gws [] = {round_mul_up(rowsA,preferred_size),round_mul_up(colsB,preferred_size)};
    err = clEnqueueNDRangeKernel(q,k,2,NULL,gws,lws,0,NULL,&event);
    ocl_check(err,"failed to enqueue matrix multiplication kernel");

    return event;
}

cl_event matrix_generator(cl_program src, cl_command_queue q, cl_device_id d, cl_mem matrix, cl_uint rows, cl_uint cols){
    static bool init = true;
    static cl_kernel k;
    cl_int err;
    if(init){
        k = clCreateKernel(src,"matrix_generator",&err);
        ocl_check(err,"failed to load kernel: matrix generator",&err);
        init = false;
    }

    err = clSetKernelArg(k,0,sizeof(matrix),&matrix);
    ocl_check(err,"failed to set first arg of matrix generator");

    err = clSetKernelArg(k,1,sizeof(cols),&cols);
    ocl_check(err,"failed to set second arg of matrix generator");

    unsigned int random = (rand()%100)+1;
    err = clSetKernelArg(k,2,sizeof(random),&random);

    cl_event event;
    const size_t gws[] = {rows,cols};
    err = clEnqueueNDRangeKernel(q,k,2,NULL,gws,NULL,0,NULL,&event);
    ocl_check(err,"failed to launch kernel matrix generator");

    return event;
}

cl_event matrix_traspose(cl_program src, cl_command_queue q, cl_mem matrix_in, cl_mem matrix_out,cl_uint rows, cl_uint cols){
    static bool init=true;
    static cl_kernel k;
    cl_int err;
    if(init){
        k = clCreateKernel(src,"matrix_traspose",&err);
        ocl_check(err, "create kernel matrix traspose");
        init = false;
    }

    size_t arg = 0;

    err = clSetKernelArg(k,arg++,sizeof(matrix_in),&matrix_in);
    ocl_check(err,"argument %d of kernel transpose",arg-1);

    err = clSetKernelArg(k,arg++,sizeof(matrix_out),&matrix_out);
    ocl_check(err,"argument %d of kernel transpose",arg-1);

    err = clSetKernelArg(k,arg++,sizeof(cols),&cols);
    ocl_check(err,"argument %d of kernel transpose",arg-1);

    cl_event event;
    const size_t gws[] = {round_mul_up(rows,32),round_mul_up(cols,32)};
    err = clEnqueueNDRangeKernel(q,k,2,NULL,gws,NULL,0,NULL,&event);
    ocl_check(err,"launching transpose kernel");

    return event;
}
