#include "ocl_boiler.h"
#include <CL/cl.h>
#include "geo_mapping.h"

#define LWS 32
#define JUMPS 50

cl_event grid_init(
    cl_command_queue q, 
    cl_kernel k, 
    cl_mem grid_index, 
    cl_mem grid_dist,
    size_t nels, 
    size_t bound
){
    cl_int err;
    cl_uint arg=0;

    err = clSetKernelArg(k,arg,sizeof(grid_index),&grid_index);
    ocl_check(err,"clSetKernelArg grid_init grid_index");
    arg++;

    err = clSetKernelArg(k,arg,sizeof(grid_dist),&grid_dist);
    ocl_check(err,"clSetKernelArg grid_init grid_dist");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(nels), &nels);
    ocl_check(err, "clSetKernelArg grid_init nels");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(bound), &bound);
    ocl_check(err, "clSetKernelArg grid_init bound");
    arg++;

    cl_event event;
    const size_t gws[] = {round_mul_up(bound,32)};
    err = clEnqueueNDRangeKernel(q,k,1,NULL,gws,NULL,0,NULL,&event);
    ocl_check(err,"clEnqueueNDRangeKernel grid_init");
    
    return event;
}

cl_event build_vectormap(
    cl_command_queue q,
    cl_kernel k,
    size_t cols,
    size_t rows,
    cl_mem input,
    cl_mem output
){
    cl_int err;
    cl_uint arg = 0;

    err = clSetKernelArg(k,arg,sizeof(cols),&cols);
    ocl_check(err,"clSetKernelArg cols");
    arg++;

    err = clSetKernelArg(k,arg,sizeof(rows),&rows);
    ocl_check(err,"clSetKernelArg rows");
    arg++;

    err = clSetKernelArg(k,arg,sizeof(input),&input);
    ocl_check(err,"clSetKernelArg input");
    arg++;

    err = clSetKernelArg(k,arg,sizeof(output),&output);
    ocl_check(err,"clSetKernelArg output");
    arg++;

    size_t gws [] = {round_mul_up(cols * rows,LWS)};
    size_t lws [] = {LWS};

    cl_event event;

    err = clEnqueueNDRangeKernel(q,k,1,NULL,gws,lws,0,NULL,&event);
    ocl_check(err,"clEnqueueNDRangeKernel build_vectormap");

    return event;
}

cl_event grid_jumping(
    cl_command_queue q,
    cl_kernel k,
    double corner_west,
    double corner_north,
    double res,
    unsigned long cols,
    unsigned long rows,
    int jumps,
    int radius,
    cl_mem northing,
    cl_mem easting,
    cl_mem northing_vm,
    cl_mem easting_vm,
    cl_mem grid_index,
    cl_mem grid_dist,
    int span_ew,
    int span_ns
){
    cl_int err;
    cl_uint arg = 0;

    err = clSetKernelArg(k,arg,sizeof(corner_west),&corner_west);
    ocl_check(err,"clSetKernelArg corner_west");
    arg++;

    err = clSetKernelArg(k,arg,sizeof(corner_north),&corner_north);
    ocl_check(err,"clSetKernelArg corner_north");
    arg++;

    err = clSetKernelArg(k,arg,sizeof(res),&res);
    ocl_check(err,"clSetKernelArg res");
    arg++;

    err = clSetKernelArg(k,arg,sizeof(cols),&cols);
    ocl_check(err,"clSetKernelArg cols");
    arg++;

    err = clSetKernelArg(k,arg,sizeof(rows),&rows);
    ocl_check(err,"clSetKernelArg rows");
    arg++;

    err = clSetKernelArg(k,arg,sizeof(jumps),&jumps);
    ocl_check(err,"clSetKernelArg jumps");
    arg++;
      
    err = clSetKernelArg(k,arg,sizeof(radius),&radius);
    ocl_check(err,"clSetKernelArg radius");
    arg++;

    err = clSetKernelArg(k,arg,sizeof(northing),&northing);
    ocl_check(err,"clSetKernelArg northing");
    arg++;
    

    err = clSetKernelArg(k,arg,sizeof(easting),&easting);
    ocl_check(err,"clSetKernelArg easting");
    arg++;
    

    err = clSetKernelArg(k,arg,sizeof(northing_vm),&northing_vm);
    ocl_check(err,"clSetKernelArg northing_vm");
    arg++;
    

    err = clSetKernelArg(k,arg,sizeof(easting_vm),&easting_vm);
    ocl_check(err,"clSetKernelArg easting_vm");
    arg++;
    

    err = clSetKernelArg(k,arg,sizeof(grid_index),&grid_index);
    ocl_check(err,"clSetKernelArg grid_index");
    arg++;
    

    err = clSetKernelArg(k,arg,sizeof(grid_dist),&grid_dist);
    ocl_check(err,"clSetKernelArg grid_dist");
    arg++;
    
    size_t gws[] = {(size_t)span_ew, (size_t)span_ns};
    cl_event event;

    err = clEnqueueNDRangeKernel(q,k,2,NULL,gws,NULL,0,NULL,&event);
    ocl_check(err,"clEnqueueNDRangeKernel grid_jumping");

    return event;
}


unsigned long * geo_mapping(
    double corner_west, 
    double corner_north, 
    double res, 
    int span_ew, 
    int span_ns,
    int radius,
    const double * easting,
    const double * northing,
    const unsigned long rows, 
    const unsigned long cols
){
    unsigned long * r_array = (unsigned long *) malloc(sizeof(unsigned long) * span_ew * span_ns);
    if (r_array != NULL){
        fprintf(stderr,"error on allocation of return array\n");
        exit(1);
    }

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p,d);
    cl_command_queue q = create_queue(ctx,d);
    cl_program src = create_program("geo_mapping.ocl",ctx,d);

    cl_int err;

    cl_kernel grid_init_k = clCreateKernel(src,"grid_init",&err);
    ocl_check(err,"clCreateKernel grid_init");

    cl_kernel build_vectormap_k = clCreateKernel(src,"build_vectormap",&err);
    ocl_check(err,"clCreateKernel build_vectormap");

    cl_kernel grid_jumping_k = clCreateKernel(src,"grid_jumping",&err);
    ocl_check(err,"clCreateKernel grid_jumping");

    cl_mem easting_d = clCreateBuffer(ctx,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,rows*cols*sizeof(cl_double),(void*)easting,&err);
    ocl_check(err,"clCreateBuffer easting_d");

    cl_mem easting_vm = clCreateBuffer(ctx, CL_MEM_READ_WRITE,rows*cols*sizeof(cl_double2),NULL,&err);
    ocl_check(err,"clCreateBuffer easting_vm");

    cl_mem northing_d = clCreateBuffer(ctx,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,rows*cols*sizeof(cl_double),(void*)northing,&err);
    ocl_check(err,"clCreateBuffer northing_d");

    cl_mem northing_vm = clCreateBuffer(ctx, CL_MEM_READ_WRITE,rows*cols*sizeof(cl_double2),NULL,&err);
    ocl_check(err,"clCreateBuffer northing_vm");

    size_t index_memsize = span_ew*span_ns*sizeof(size_t);
    cl_mem grid_index = clCreateBuffer(ctx,CL_MEM_READ_WRITE ,index_memsize,NULL,&err);
    ocl_check(err,"clCreateBuffer grid_index");

    size_t distance_memsize = span_ew*span_ns*sizeof(cl_double);
    cl_mem grid_dist = clCreateBuffer(ctx,CL_MEM_READ_WRITE ,distance_memsize,NULL,&err);
    ocl_check(err,"clCreateBuffer grid_dist"); 
    
    cl_event grid_init_e = grid_init(q,grid_init_k,grid_index,grid_dist,rows*cols,span_ns*span_ew);
    clWaitForEvents(1,&grid_init_e);

    cl_event build_vectormaps_e[2];
    build_vectormaps_e[0] = build_vectormap(q,build_vectormap_k,cols,rows,easting_d,easting_vm);
    build_vectormaps_e[1] = build_vectormap(q,build_vectormap_k,cols,rows,northing_d,northing_vm);
    clWaitForEvents(2,build_vectormaps_e);
    
    cl_event grid_jumping_e = grid_jumping(q,
        grid_jumping_k,
        corner_west,
        corner_north,
        res,
        cols,
        rows,
        JUMPS,
        radius,
        northing_d,
        easting_d,
        northing_vm,
        easting_vm,
        grid_index,
        grid_dist,
        span_ew,
        span_ns
    );
    clWaitForEvents(1,&grid_jumping_e);

    cl_event read_buffer_grid_index;
    err = clEnqueueReadBuffer(q,grid_index,CL_TRUE,0,index_memsize,r_array,0,NULL,&read_buffer_grid_index);
    ocl_check(err,"clEnqueueReadBuffer r_array");

    cl_ulong time_read_buffer_grid_index = runtime_ns(read_buffer_grid_index);

    size_t bytes_build_vm = ((rows - 1) * (cols - 1) * 5 * sizeof(cl_double)) + (((rows * 2) + (2 * (cols - 2))) * 4 * sizeof(cl_double)) - (4 * sizeof(cl_double)) + (rows * cols * sizeof(cl_double2));
    cl_ulong time_build_easting_vm = runtime_ns(build_vectormaps_e[0]);
    double bandwith_build_easting_vm = (double) bytes_build_vm / time_build_easting_vm;
    double throughput_build_easting_vm = (double) (rows * cols) / time_build_easting_vm;

    cl_ulong time_build_northing_vm = runtime_ns(build_vectormaps_e[1]);
    double bandwith_build_northing_vm = (double) bytes_build_vm / time_build_northing_vm;
    double throughput_build_northing_vm = (double) (rows * cols) / time_build_northing_vm;

    size_t bytes_grid_jumping = ((span_ew * span_ns) * JUMPS * 2 * (sizeof(cl_double2) + sizeof(cl_double))) + ((span_ew - ((radius * 2) + 1)) * (span_ns - ((radius * 2) + 1))) * (((radius * 2) + 1) * ((radius * 2) + 1) * 2 * sizeof(cl_double));
    cl_ulong time_grid_jumping = runtime_ns(grid_jumping_e);
    double bandwith_grid_jumping = (double) bytes_grid_jumping / time_grid_jumping;
    double throughput_grid_jumping = (double) (span_ew * span_ns ) / time_grid_jumping;

    printf(" build_easting_vm time:%g bandwith:%g throughput:%g\n",time_build_easting_vm * 1.e-6,bandwith_build_easting_vm,throughput_build_easting_vm);
    printf(" build_northing_vm time:%g bandwith:%g throughput:%g\n",time_build_northing_vm * 1.e-6,bandwith_build_northing_vm,throughput_build_northing_vm);
    printf("profiling: grid_jumping time:%g bandwith:%g throughput:%g\n",time_grid_jumping * 1.e-6,bandwith_grid_jumping,throughput_grid_jumping);

    printf("read_buffer grid_index time:%g bandwith:%g throughput:%g\n",time_read_buffer_grid_index * 1.e-6,(double)(sizeof(size_t) * span_ew * span_ns) / time_read_buffer_grid_index,(double) (span_ew * span_ns) / time_read_buffer_grid_index);

    clReleaseMemObject(easting_vm);
    clReleaseMemObject(easting_d);
    clReleaseMemObject(northing_vm);
    clReleaseMemObject(northing_d);
    clReleaseMemObject(grid_dist);
    clReleaseMemObject(grid_index);
    clReleaseProgram(src);
	clReleaseCommandQueue(q);
	clReleaseContext(ctx);

    return r_array;
}