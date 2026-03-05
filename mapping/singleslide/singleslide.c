#include "ocl_boiler.h"
#include "singleslide.h"

#define LWS 32

cl_event grid_init(
    cl_command_queue q, 
    cl_kernel k, 
    cl_mem grid_index, 
    cl_mem grid_dist,
    size_t nels, 
    int span_ns, 
    int span_ew
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

    cl_event event;
    const size_t gws[] = {round_mul_up(span_ns*span_ew,32)};
    err = clEnqueueNDRangeKernel(q,k,1,NULL,gws,NULL,0,NULL,&event);
    ocl_check(err,"clEnqueueNDRangeKernel grid_init");
    
    return event;
}

cl_event neighbor_single_slide(
    cl_command_queue q,
    cl_kernel k,
    double corner_west,
    double corner_north,
    double res,
    cl_int span_ew,
    cl_int span_ns,
    cl_int radius,
    cl_mem easting,
    cl_mem northing,
    cl_ulong nels,
    cl_mem grid_index,
    cl_mem grid_dist,
    size_t lws_in
){
    cl_int err;
    cl_uint arg=0;

    err = clSetKernelArg(k, arg, sizeof(corner_west), &corner_west);
    ocl_check(err, "clsetkernelarg neighbor_single_slide corner_west");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(corner_north), &corner_north);
    ocl_check(err, "clSetKernelArg neighbor_single_slide corner_north");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(res), &res);
    ocl_check(err, "clSetKernelArg neighbor_single_slide res");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(span_ew), &span_ew);
    ocl_check(err, "clSetKernelArg neighbor_single_slide span_ew");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(span_ns), &span_ns);
    ocl_check(err, "clSetKernelArg neighbor_single_slide span_ns");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(radius), &radius);
    ocl_check(err, "clSetKernelArg neighbor_single_slide radius");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(easting), &easting);
    ocl_check(err, "clSetKernelArg neighbor_single_slide easting");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(northing), &northing);
    ocl_check(err, "clSetKernelArg neighbor_single_slide northing");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(nels), &nels);
    ocl_check(err, "clSetKernelArg neighbor_single_slide nels");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(grid_index), &grid_index);
    ocl_check(err, "clSetKernelArg neighbor_single_slide grid_index");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(grid_dist), &grid_dist);
    ocl_check(err, "clSetKernelArg neighbor_single_slide grid_dist");
    arg++;

    size_t neighbour_elements = ((radius * 2) + 1) * ((radius * 2) + 1);

    err = clSetKernelArg(k, arg, sizeof(cl_int)*neighbour_elements*lws_in, NULL);
    ocl_check(err, "clSetKernelArg neighbor_single_slide cache_index");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(cl_double)*neighbour_elements*lws_in, NULL);
    ocl_check(err, "clSetKernelArg neighbor_single_slide cache_dist");
    arg++;

    const size_t lws [] = {lws_in};
    const size_t gws [] = {lws[0]};

    cl_event event;
    err = clEnqueueNDRangeKernel(q,k,1,NULL,gws,lws,0,NULL,&event);
    ocl_check(err, "clEnqueueNDRangeKernel neighbor_single_slide");

    return event;
}

unsigned long * single_slide_mapping(
    const double corner_west, 
    const double corner_north, 
    const double res, 
    const int span_ew, 
    const int span_ns,
    const int radius,
    const double * easting,
    const double * northing,
    const unsigned long nels
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
    cl_program src = create_program("mapping.ocl",ctx,d);

    cl_int err;

    cl_kernel init_k = clCreateKernel(src,"grid_init",&err);
    ocl_check(err,"clCreateKernel grid_init");

    cl_kernel neigh_k = clCreateKernel(src,"neighbor_single_slide",&err);
    ocl_check(err,"clCreateKernel neighbor_single_slide");

    cl_mem easting_d = clCreateBuffer(ctx,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,nels*sizeof(double),(void*)easting,&err);
    ocl_check(err,"clCreateBuffer easting_d");

    cl_mem northing_d = clCreateBuffer(ctx,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,nels*sizeof(double),(void*)northing,&err);
    ocl_check(err,"clCreateBuffer northing_d");

    size_t index_memsize = span_ew*span_ns*sizeof(size_t);
    cl_mem grid_index = clCreateBuffer(ctx,CL_MEM_READ_WRITE ,index_memsize,NULL,&err);
    ocl_check(err,"clCreateBuffer grid_index");

    size_t distance_memsize = span_ew*span_ns*sizeof(cl_double);
    cl_mem grid_dist = clCreateBuffer(ctx,CL_MEM_READ_WRITE ,distance_memsize,NULL,&err);
    ocl_check(err,"clCreateBuffer grid_dist");

    cl_event grid_init_e = grid_init(q,init_k,grid_index,grid_dist,nels,span_ns,span_ew);
    clWaitForEvents(1,&grid_init_e);

    cl_event neighbor_e = neighbor_single_slide(q,neigh_k,
        corner_west,corner_north,res,
        span_ew,span_ns,radius,
        easting_d,northing_d,nels,
        grid_index,grid_dist,LWS);
    clWaitForEvents(1,&neighbor_e);

    cl_event read_buffer_grid_index;

    err = clEnqueueReadBuffer(q,grid_index,CL_TRUE,0,index_memsize,r_array,0,NULL,&read_buffer_grid_index);
    ocl_check(err,"clEnqueueReadBuffer index_array");
    
    cl_ulong time_read_buffer_grid_index = runtime_ns(read_buffer_grid_index);

    cl_ulong time_grid_init_ns = runtime_ns(grid_init_e);
    double time_grid_init_ms = time_grid_init_ns * 1.e-6;
    double bandwith_grid_init = (double) (span_ew * span_ns * 2 * sizeof(cl_double)) / time_grid_init_ns;
    double throughput_grid_init = (double)(span_ew * span_ns) / time_grid_init_ns;

    cl_ulong time_neighbor_ns = runtime_ns(neighbor_e);
    double time_neighbor_ms = time_neighbor_ns * 1.e-6;
    double bandwith_neighbor = (double) ( nels * (sizeof(cl_double) * 2) +  (span_ew * span_ns) * (sizeof(size_t) + sizeof(double))) / time_neighbor_ns;
    double throughput_neighbor = (double) (nels) / time_neighbor_ns;
    

    printf("grid_init time:%g bandwith:%g throughput:%g\n",time_grid_init_ms,bandwith_grid_init,throughput_grid_init);
    printf("neighbor_single_slide time:%g bandwith:%g throughput:%g\n",time_neighbor_ms,bandwith_neighbor,throughput_neighbor);
    printf("read_buffer grid_index time:%g bandwith:%g throughput:%g\n",time_read_buffer_grid_index * 1.e-6,(double)(sizeof(size_t) * span_ew * span_ns) / time_read_buffer_grid_index,(double) (span_ew * span_ns) / time_read_buffer_grid_index);

    clReleaseMemObject(easting_d);
    clReleaseMemObject(northing_d);
    clReleaseMemObject(grid_dist);
    clReleaseMemObject(grid_index);
    clReleaseProgram(src);
	clReleaseCommandQueue(q);
	clReleaseContext(ctx);

    return r_array;
}