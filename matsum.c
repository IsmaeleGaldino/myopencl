#include "ocl_boiler.h"

cl_event matinit(cl_command_queue q, cl_kernel k, cl_mem mat, cl_uint nrows, cl_uint ncols, size_t lws_in){

    size_t arg = 0;
    cl_int err;

    err = clSetKernelArg(k,arg++,sizeof(mat),&mat);
    ocl_check(err,"matinit clSetKernelArg mat");

    err = clSetKernelArg(k,arg++,sizeof(nrows),&nrows);
    ocl_check(err,"matinit clSetKernelArg nrows");

    err = clSetKernelArg(k,arg++,sizeof(ncols),&ncols);
    ocl_check(err,"matinit clSetKernelArg ncols");

    const size_t gws[] = {round_mul_up(ncols,lws_in),round_mul_up(nrows,lws_in)};
    const size_t lws[] = {lws_in,lws_in};

    cl_event event;
    err = clEnqueueNDRangeKernel(q,k,2,NULL,gws,lws,0,NULL,&event);
    ocl_check(err,"matinit clEnqueueNDRangeKernel");

    return event;
}

cl_event matsum(cl_command_queue q, cl_kernel k, cl_mem mat_a, cl_mem mat_b, cl_mem mat_r, cl_uint nrows, cl_uint ncols, size_t lws_in){

    size_t arg = 0;
    cl_int err;

    err = clSetKernelArg(k,arg++,sizeof(mat_a),&mat_a);
    ocl_check(err,"matsum clSetKernelArg mat_a");
    
    err = clSetKernelArg(k,arg++,sizeof(mat_b),&mat_b);
    ocl_check(err,"matsum clSetKernelArg mat_b");

    err = clSetKernelArg(k,arg++,sizeof(mat_r),&mat_r);
    ocl_check(err,"matsum clSetKernelArg mat_r");

    err = clSetKernelArg(k,arg++,sizeof(nrows),&nrows);
    ocl_check(err,"matsum clSetKernelArg nrows");
    
    err = clSetKernelArg(k,arg++,sizeof(ncols),&ncols);
    ocl_check(err,"matsum clSetKernelArg ncols");

    const size_t gws[] = {round_mul_up(ncols,lws_in),round_mul_up(nrows,lws_in)};
    const size_t lws[] = {lws_in,lws_in};

    cl_event event;
    err = clEnqueueNDRangeKernel(q,k,2,NULL,gws,lws,0,NULL,&event);
    ocl_check(err,"matsum clEnqueueNDRangeKernel");

    return event;
}

void verify(cl_int * mat, cl_uint nrows, cl_uint ncols){
    for(cl_uint r = 0 ; r < nrows ; r++){
        for(cl_uint c = 0 ; c < ncols ; c++){

            if( mat[r*ncols + c] != (2 * c) - (2 * r) ) {
                fprintf(stderr,"(%u,%u) != 0 \n",r,c);
                exit(2);
            }
        }
    }
}

int main(int argc, char * argv []){

    if(argc != 4){
        fprintf(stderr,"invalid arguments: ./matsum nrows ncols lws\n");
        exit(1);
    }

    cl_uint nrows = strtoul(argv[1],NULL,10);
    cl_uint ncols = strtoul(argv[2],NULL,10);
    size_t lws_in = strtoul(argv[3],NULL,10);

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p,d);
    cl_program src = create_program("matsum.ocl",ctx,d);
    cl_command_queue q = create_queue(ctx,d);

    const size_t memsize = ncols * nrows * sizeof(cl_int);

    cl_int err;

    cl_mem mat_a = clCreateBuffer(ctx,CL_MEM_READ_WRITE,memsize,NULL,&err);
    ocl_check(err,"mat_a clCreateBuffer");

    cl_mem mat_b = clCreateBuffer(ctx,CL_MEM_READ_WRITE,memsize,NULL,&err);
    ocl_check(err,"mat_b clCreateBuffer");

    cl_mem mat_r = clCreateBuffer(ctx,CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,memsize,NULL,&err);
    ocl_check(err,"mat_r clCreateBuffer");

    cl_kernel matinit_k = clCreateKernel(src,"matinit",&err);
    ocl_check(err,"matinit clCreateKernel");

    cl_kernel matsum_k = clCreateKernel(src,"matsum",&err);
    ocl_check(err,"matsum clCreateKernel");

    cl_event matinit_evt_a = matinit(q,matinit_k,mat_a,nrows,ncols,lws_in);
    cl_event matinit_evt_b = matinit(q,matinit_k,mat_b,nrows,ncols,lws_in);

    clFinish(q);

    cl_event matsum_evt = matsum(q,matsum_k,mat_a,mat_b,mat_r,nrows,ncols,lws_in);
    
    cl_int * result = clEnqueueMapBuffer(q,mat_r,CL_TRUE,CL_MAP_READ,0,memsize,1,&matsum_evt,NULL,&err);
    ocl_check(err,"clEnqueueMapBuffer");

    verify(result,nrows,ncols);

    cl_ulong time_ns = runtime_ns(matsum_evt);
    double time_ms = time_ns * 10e-6;
    double througput =(double) ( nrows * ncols )/ time_ns;
    printf("time: %lf , throughput: %lf \n",time_ms,througput);

    clEnqueueUnmapMemObject(q,mat_r,result,0,NULL,NULL);
    clReleaseMemObject(mat_a);
    clReleaseMemObject(mat_b);
    clReleaseMemObject(mat_r);
    clReleaseCommandQueue(q);
    clReleaseProgram(src);
    clReleaseContext(ctx);
    clReleaseDevice(d);

}