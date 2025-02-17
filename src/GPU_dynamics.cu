#pragma execution_character_set("utf-8")
#include "HEOM_utilize.h"
#include "HEOM_kernel_func.h"
#include "HEOM_constant.h"
#include "HEOM_dynamics.h"
#include "HEOM_TD_hamiltonian.h"
#include "HEOM_param.h"
#include "HEOM_polar.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>


using namespace std;

int main(int argc, char** argv)
{	
	string filename(argv[1]);
	cout << "Now is GPU version" << '\n';
	std::cout << "running " << filename << '\n';
	param k(filename);
	construct_ADO_set(k);
	
	cublasHandle_t cublasH;
	cublasCreate(&cublasH);

	int sys_size = k.sys_size;
	int total_size = k.sys_size * k.sys_size;
	
    vector<int> sites = {1, 2};
	vector<vector<float>> population(sites.size(), vector<float>((k.t_end - k.t_start) / k.print_step + 1, 0.0));

    dynamics_solver(k, sites, population);

	for (int i = 0; i < sites.size(); i++) {
        cout << "site" << " " << i + 1 << " " << "population: " << '\n' << '\n';
		for (int j = 0; j < population[i].size(); j++) {
			cout << population[i][j] << '\n';
		}
        cout << '\n';
	}
	
	cublasError(cublasDestroy(cublasH), __FILE__, __LINE__);
	cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
	
	k.param_free();
	return 0;
}



