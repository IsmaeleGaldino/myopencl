#include "ocl_boiler.h"
#include <stdbool.h>

void verify(const int *array, int nels)
{
	for (int i = 0; i < nels; ++i) {
		const int expected = i < nels - 1 ? i : i-1;
		const int computed = array[i];
		if (expected != computed) {
			fprintf(stderr, "%d: %d ≠ %d\n",
				i, computed, expected);
			exit(4);
		}
	}
}

cl_event vec_init(cl_program src, cl_command_queue q, cl_device_id d, cl_mem vec, cl_uint nels){
    static cl_kernel k;
    static bool init = true;
    cl_int err;  

    if(init){
        k=clCreateKernel(src,"vec_init",&err);
        ocl_check(err,"create kernel vec_init");
        init = false;
    }
    cl_uint arg=0;
    err = clSetKernelArg(k,arg++,sizeof(vec),&vec);
    ocl_check(err,"setKernelArg 1 of vecinit");

    err = clSetKernelArg(k,arg++,sizeof(nels),&nels);
    ocl_check(err,"setKernelArg 2 of vecinit");

    size_t gwsm;
    err = clGetKernelWorkGroupInfo(k,d,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,sizeof(gwsm),&gwsm,NULL);
    ocl_check(err,"get kernel work group size multiple");

    cl_event event;
    const size_t gws[]={round_mul_up(nels,gwsm)};
    err = clEnqueueNDRangeKernel(q,k,1,NULL,gws,NULL,0,NULL,&event);
    ocl_check(err,"launch of vec_init");

    return event;
}


cl_event vec_smooth_lm(cl_program src, cl_command_queue q, cl_mem vec_in, cl_mem vec_out, size_t nels_rounded,size_t lws_in){
    static cl_kernel k;
    static bool init = true;
    cl_int err;

    if(init){
        k=  clCreateKernel(src,"vec_smooth_lm",&err);
        ocl_check(err,"crate vec_smooth_lm kernel");
        init = false;
    }

    const size_t lws[] = {lws_in};
    const size_t gws[] = {nels_rounded/8};

    cl_uint arg=0;

    err = clSetKernelArg(k,arg++,sizeof(vec_in),&vec_in);
    ocl_check(err,"setKernelArg %u of vec_smooth_lm",arg-1);

    err = clSetKernelArg(k,arg++,sizeof(vec_out),&vec_out);
    ocl_check(err,"setKernelArg %u of vec_smooth_lm",arg-1);
    
    err = clSetKernelArg(k,arg++,sizeof(cl_int2)*(lws[0]+2),NULL);
    ocl_check(err,"setKernelArg %u of vec_smooth_lm",arg-1);

    cl_event event;
    err = clEnqueueNDRangeKernel(q,k,1,NULL,gws,lws,0,NULL,&event);
    ocl_check(err,"launch of vec_smooth_lm kernel");

    return event;
}


int main(int argc,char * argv[]){
    if(argc != 3){
        printf("./smooth nels lws\n");
        return 1;
    }


    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p,d);
    cl_command_queue q = create_queue(ctx,d);
    cl_program src = create_program("vecslm.ocl",ctx,d);

    cl_int err;
    size_t nels_rounded = round_mul_up(strtoul(argv[1],NULL,10),64)*8;
    size_t lws_in = strtoul(argv[2],NULL,10);
    printf("input rounded to %u\n",nels_rounded);
    size_t memsize = nels_rounded*sizeof(cl_int);
    cl_mem vec_in = clCreateBuffer(ctx,CL_MEM_READ_WRITE,memsize,NULL,&err);
    ocl_check(err,"create buffer vec_in");

    cl_mem vec_out = clCreateBuffer(ctx,CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,memsize,NULL,&err);
    ocl_check(err,"create buffer vec_out");

    cl_event vecinit_evt;
    vecinit_evt = vec_init(src,q,d,vec_in,nels_rounded);
    err = clWaitForEvents(1,&vecinit_evt);
    ocl_check(err,"wait for vecinit");
    
    cl_event vecsmoothlm_evt;
    vecsmoothlm_evt = vec_smooth_lm(src,q,vec_in,vec_out,nels_rounded,lws_in);
    err = clWaitForEvents(1,&vecsmoothlm_evt);
    ocl_check(err,"wait for vecsmoothlm");

    
    cl_event map_evt;
    cl_int * host_ptr = clEnqueueMapBuffer(q,vec_out,CL_TRUE,CL_MAP_READ,0,memsize,0,NULL,&map_evt,&err);
    ocl_check(err,"read vec_out from map");    

    verify(host_ptr,nels_rounded);

    err = clEnqueueUnmapMemObject(q,vec_out,host_ptr,0,NULL,NULL);
    ocl_check(err,"unmap of vec_out");

    clFinish(q);

    double time_ms = runtime_ms(vecsmoothlm_evt);
    double througput =(double)nels_rounded/runtime_ns(vecsmoothlm_evt);
    printf("time : %gms , througput: %gGE/s \n",time_ms,througput);


    clReleaseMemObject(vec_in);
    clReleaseMemObject(vec_out);
    clReleaseProgram(src);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    return 0;
}