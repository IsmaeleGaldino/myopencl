#include <stdio.h>
#include <stdlib.h>

#include "ocl_boiler.h"

size_t vecinit_preferred_wg_multiple;
size_t vecsmooth_preferred_wg_multiple;

cl_event vecinit(cl_command_queue q, cl_kernel vecinit_k, cl_int nels, int lws_in,
	cl_mem d_array)
{
	const size_t lws[] = { lws_in ? lws_in : vecinit_preferred_wg_multiple };
	const size_t gws[] = { round_mul_up(nels, lws[0]) };

	printf("nels %d round to %zu GWS %zu\n", nels, lws[0], gws[0]);

	cl_int err = clSetKernelArg(vecinit_k, 0, sizeof(d_array), &d_array);
	ocl_check(err, "setKernelArg vecinit_k 0");

	err = clSetKernelArg(vecinit_k, 1, sizeof(nels), &nels);
	ocl_check(err, "setKernelArg vecinit_k 1");

	cl_event vecinit_evt;
	err = clEnqueueNDRangeKernel(q, vecinit_k,
		1, // numero dimensioni
		NULL, // offset
		gws, // global work size
		(lws_in ? lws : NULL), // local work size
		0, // numero di elementi nella waiting list
		NULL, // waiting list
		&vecinit_evt); // evento di questo comando
	ocl_check(err, "Enqueue vecinit_k");

	return vecinit_evt;
}

cl_event vecsmooth(cl_command_queue q, cl_kernel vecsmooth_k, cl_int nels, int lws_in,
	cl_mem d_out, cl_mem d_in,
	size_t num_events, cl_event *wait_list)
{

	cl_int nvec = (nels - 1)/4 + 1; // int4

	const size_t lws[] = { lws_in ? lws_in : vecsmooth_preferred_wg_multiple };
	const size_t gws[] = { round_mul_up(nvec, lws[0]) };

	printf("nels %d round to %zu GWS %zu\n", nels, lws[0], gws[0]);

	cl_int arg = 0;
	cl_int err = clSetKernelArg(vecsmooth_k, arg, sizeof(d_out), &d_out);
	ocl_check(err, "setKernelArg vecsmooth_k %d", arg);
	++arg;

	err = clSetKernelArg(vecsmooth_k, arg, sizeof(d_in), &d_in);
	ocl_check(err, "setKernelArg vecsmooth_k %d", arg);
	++arg;

	err = clSetKernelArg(vecsmooth_k, arg, sizeof(nvec), &nvec);
	ocl_check(err, "setKernelArg vecsmooth_k %d", arg);
	++arg;

	cl_event vecsmooth_evt;
	err = clEnqueueNDRangeKernel(q, vecsmooth_k,
		1, // numero dimensioni
		NULL, // offset
		gws, // global work size
		(lws_in ? lws : NULL), // local work size
		num_events, // numero di elementi nella waiting list
		wait_list, // waiting list
		&vecsmooth_evt); // evento di questo comando
	ocl_check(err, "Enqueue vecsmooth_k");

	return vecsmooth_evt;
}

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

int main(int argc, char *argv[])
{
	int nels = 0;
	int lws = 0;

	if (argc < 2) {
		fprintf(stderr, "%s nels [lws]\n", argv[0]);
		exit(1);
	}

	nels = atoi(argv[1]);

	if (nels < 1) {
		fprintf(stderr, "nels %d < 1\n", nels);
		exit(2);

	}

	if (argc == 3) {
		lws = atoi(argv[2]);
		if (lws < 1) {
			fprintf(stderr, "lws %d < 1\n", lws);
			exit(2);

		}
	}

	/* TODO OpenCL */

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue q = create_queue(ctx, d);
	cl_program prog = create_program("vecsmooth.ocl", ctx, d);

	/* Allocate memory */

	size_t memsize = nels*sizeof(int);

	cl_int err;
	cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_WRITE, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer failed");
	cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, memsize, NULL, &err);
	ocl_check(err, "clCreateBuffer failed");

	/* Create kernel handle */

	cl_kernel vecinit_k = clCreateKernel(prog, "vecinit_k", &err);
	ocl_check(err, "clCreateKernel vecinit_k");
	cl_kernel vecsmooth_k = clCreateKernel(prog, "vecsmooth4v_k", &err);
	ocl_check(err, "clCreateKernel vecsmooth_k");

	/* Preferred workgroup size multiple */
	if (lws == 0) {
		err = clGetKernelWorkGroupInfo(vecinit_k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
			sizeof(vecinit_preferred_wg_multiple), &vecinit_preferred_wg_multiple, NULL);
		ocl_check(err, "get preferred work-group size multiple");
		err = clGetKernelWorkGroupInfo(vecinit_k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
			sizeof(vecsmooth_preferred_wg_multiple), &vecsmooth_preferred_wg_multiple, NULL);
		ocl_check(err, "get preferred work-group size multiple");
	}

	/* Run kernel */
	cl_event vecinit_evt = vecinit(q, vecinit_k, nels, lws,
		d_in);
	cl_event vecsmooth_evt = vecsmooth(q, vecsmooth_k, nels, lws,
		d_out, d_in,
		1, &vecinit_evt);

	cl_event map_evt;
	cl_int *h_array = clEnqueueMapBuffer(q, d_out, CL_TRUE,
		CL_MAP_READ,
		0, memsize,
		1, &vecsmooth_evt, &map_evt, &err);
	ocl_check(err, "map buffer");

	verify(h_array, nels);

	cl_event unmap_evt;
	err = clEnqueueUnmapMemObject(q, d_out, h_array,
		0, NULL, &unmap_evt);
	ocl_check(err, "unmap buffer");

	clFinish(q);

	cl_ulong vecinit_ns = runtime_ns(vecinit_evt);
	cl_ulong vecsmooth_ns = runtime_ns(vecsmooth_evt);
	cl_ulong map_ns = runtime_ns(map_evt);
	cl_ulong unmap_ns = runtime_ns(unmap_evt);

	double vecinit_ms = vecinit_ns*1.e-6;
	double vecsmooth_ms = vecsmooth_ns*1.e-6;
	double map_ms = map_ns*1.e-6;
	double unmap_ms = unmap_ns*1.e-6;

	double vecinit_bw = (double)memsize/vecinit_ns;
	double vecsmooth_bw = (double)memsize*2.5/vecsmooth_ns;
	double map_bw = (double)memsize/map_ns;
	double unmap_bw = (double)memsize/unmap_ns;

	double vecinit_throughput = (double)nels/vecinit_ns;
	double vecsmooth_throughput = (double)nels/vecsmooth_ns;
	double map_throughput = (double)nels/map_ns;
	double unmap_throughput = (double)nels/unmap_ns;

	printf("vecinit: %gms %gGB/s %gGE/s\n", vecinit_ms, vecinit_bw, vecinit_throughput);
	printf("vecsmooth: %gms %gGB/s %gGE/s\n", vecsmooth_ms, vecsmooth_bw, vecsmooth_throughput);
	printf("map: %gms %gGB/s %gGE/s\n", map_ms, map_bw, map_throughput);
	printf("unmap: %gms %gGB/s %gGE/s\n", unmap_ms, unmap_bw, unmap_throughput);

	clReleaseMemObject(d_out);
	clReleaseMemObject(d_in);
	clReleaseProgram(prog);
	clReleaseCommandQueue(q);
	clReleaseContext(ctx);
}
