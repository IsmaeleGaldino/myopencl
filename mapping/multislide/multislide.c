#include "ocl_boiler.h"
#include "multislide.h"

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

    err = clSetKernelArg(k,arg,sizeof(nels), &nels);
    ocl_check(err, "clSetKernelArg grid_init rows*cols");
    arg++;

    cl_event event;
    const size_t gws[] = {round_mul_up(span_ns*span_ew,32)};
    err = clEnqueueNDRangeKernel(q,k,1,NULL,gws,NULL,0,NULL,&event);
    ocl_check(err,"clEnqueueNDRangeKernel grid_init");
    
    return event;
}

cl_event neighbor_multi_slide(
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
    cl_ulong cols,
    cl_int rows_offset,
    cl_int rows_dist,
    cl_int slide_offset,
    cl_int slide_n,
    cl_mem grid_index,
    cl_mem grid_dist,
    size_t lws_cols,
    size_t lws_rows,
    size_t n_groups
){
    cl_int err;
    cl_uint arg=0;

    err = clSetKernelArg(k, arg, sizeof(corner_west), &corner_west);
    ocl_check(err, "clsetkernelarg neighbor_multi_slide corner_west");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(corner_north), &corner_north);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide corner_north");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(res), &res);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide res");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(span_ew), &span_ew);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide span_ew");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(span_ns), &span_ns);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide span_ns");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(radius), &radius);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide radius");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(easting), &easting);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide easting");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(northing), &northing);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide northing");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(cols), &cols);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide cols");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(rows_offset), &rows_offset);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide group_offset");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(rows_dist), &rows_dist);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide group_dist");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(slide_offset), &slide_offset);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide slide_offset");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(slide_n), &slide_n);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide slide_n");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(grid_index), &grid_index);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide grid_index");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(grid_dist), &grid_dist);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide grid_dist");
    arg++;

    size_t neighbour_elements = ((radius*2) + 1) * ((radius*2) + 1);


    err = clSetKernelArg(k, arg, sizeof(cl_int)*neighbour_elements*lws_cols*lws_rows, NULL);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide cache_index");
    arg++;

    err = clSetKernelArg(k, arg, sizeof(cl_double)*neighbour_elements*lws_cols*lws_rows, NULL);
    ocl_check(err, "clSetKernelArg neighbor_multi_slide cache_dist");
    arg++;

    const size_t lws [] = {lws_cols,lws_rows};
    const size_t gws [] = {lws[0] * slide_n,lws_rows*n_groups};

    cl_event event;
    err = clEnqueueNDRangeKernel(q,k,2,NULL,gws,lws,0,NULL,&event);
    ocl_check(err, "clEnqueueNDRangeKernel neighbor_multi_slide");

    return event;
}

unsigned long * multi_slide_mapping(
    const double corner_west, 
    const double corner_north, 
    const double res, 
    const int span_ew, 
    const int span_ns,
    const int radius,
    const int dist,
    const int slides,
    const int lws_rows,
    const int lws_cols,
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
    cl_program src = create_program("mapping.ocl",ctx,d);

    cl_int err;

    cl_kernel init_k = clCreateKernel(src,"grid_init",&err);
    ocl_check(err,"clCreateKernel grid_init");

    cl_kernel neigh_k = clCreateKernel(src,"neighbor_multi_slide",&err);
    ocl_check(err,"clCreateKernel neighbor_multi_slide");

    cl_mem easting_d = clCreateBuffer(ctx,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,rows*cols*sizeof(double),(void*)easting,&err);
    ocl_check(err,"clCreateBuffer easting_d");

    cl_mem northing_d = clCreateBuffer(ctx,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,rows*cols*sizeof(double),(void*)northing,&err);
    ocl_check(err,"clCreateBuffer northing_d");

    size_t index_memsize = span_ew*span_ns*sizeof(size_t);
    cl_mem grid_index = clCreateBuffer(ctx,CL_MEM_READ_WRITE ,index_memsize,NULL,&err);
    ocl_check(err,"clCreateBuffer grid_index");

    size_t distance_memsize = span_ew*span_ns*sizeof(cl_double);
    cl_mem grid_dist = clCreateBuffer(ctx,CL_MEM_READ_WRITE ,distance_memsize,NULL,&err);
    ocl_check(err,"clCreateBuffer grid_dist");

    cl_event grid_init_e = grid_init(q,init_k,grid_index,grid_dist,rows*cols,span_ns,span_ew);
    clWaitForEvents(1,&grid_init_e);

    size_t remain_rows = rows % dist;
    
    size_t n_groups = rows/dist;
    size_t kernel_launches = dist/lws_rows;
    cl_event multi_slide_events[kernel_launches * 2];
    for(size_t group_offset = 0 ; group_offset < kernel_launches ; group_offset++){
        multi_slide_events[group_offset * 2] = neighbor_multi_slide(q,neigh_k,corner_west,corner_north,res,span_ew,span_ns,radius,easting_d,northing_d,cols,group_offset,dist,0,slides,grid_index,grid_dist,lws_cols,lws_rows,n_groups);
        clWaitForEvents(1,&multi_slide_events[(group_offset * 2)]);
        multi_slide_events[(group_offset * 2) + 1] = neighbor_multi_slide(q,neigh_k,corner_west,corner_north,res,span_ew,span_ns,radius,easting_d,northing_d,cols,group_offset,dist,1,slides,grid_index,grid_dist,lws_cols,lws_rows,n_groups);
        clWaitForEvents(1,&multi_slide_events[(group_offset * 2) + 1]);
    }

    cl_event read_buffer_grid_index;
    err = clEnqueueReadBuffer(q,grid_index,CL_TRUE,0,index_memsize,r_array,0,NULL,&read_buffer_grid_index);
    ocl_check(err,"clEnqueueReadBuffer index_array");
    
    cl_ulong time_read_buffer_grid_index = runtime_ns(read_buffer_grid_index);

    cl_ulong time_grid_init_ns = runtime_ns(grid_init_e);
    double time_grid_init_ms = time_grid_init_ns * 1.e-6;
    double bandwith_grid_init = (double) ((span_ew * span_ns) * (sizeof(cl_double) + sizeof(size_t)))  / time_grid_init_ns;
    double throughput_grid_init = (double)(span_ew * span_ns) / time_grid_init_ns;

    size_t bytes_neighboring = (rows*cols) * sizeof(cl_double) + distance_memsize + (2 * index_memsize);

    cl_ulong time_multislide_ns = total_runtime_ns(multi_slide_events[0],multi_slide_events[(kernel_launches*2)-1]);
    double time_multislide_ms = time_multislide_ns * 1.e-6;
    double bandwith_multislide = (double) (bytes_neighboring) / time_multislide_ns;
    double throughput_multislide = (double)((rows*cols)) / time_multislide_ns;

    printf("remaining rows: %lu \n",remain_rows);

    printf("grid_init time:%g , bandwith:%g , throughput:%g\n",time_grid_init_ms,bandwith_grid_init,throughput_grid_init);
    printf("profiling: neigh_multi_slide time:%g bandwith:%g throughput:%g\n",time_multislide_ms,bandwith_multislide,throughput_multislide);

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