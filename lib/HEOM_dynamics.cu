#include "HEOM_dynamics.h"

int delta(const int a, const int b) {
	return a == b ? 1 : 0;
}

void construct_Hal(param& key, vector<data_type>& Hal) {
	int sys_size = key.sys_size;
	int total_size = sys_size * sys_size;
	float sigma = 0;
	float tmp = 0;
	vector<data_type> deltaH(key.single_size * key.single_size, make_cuComplex(0.0, 0.0));

	for (int i = 0; i < key.single_size; i++) {
		for (int j = 0; j < key.single_size; j++) {
			sigma = key.Disorder[i * key.single_size + j].x;
			if (fabs(sigma) < 1e-10) tmp = 0;
			else tmp = generateRandomGaussian(0, sigma);
			deltaH[i * key.single_size + j] = make_cuComplex(tmp, 0.0);
		}
	}

	Hal = key.Hal;
	for (int i = 0; i < key.single_size; i++) {
		for (int j = 0; j < key.single_size; j++) {
			Hal[i * sys_size + j] = cuCaddf(Hal[i * sys_size + j], deltaH[i * key.single_size + j]);
		}
	}

	for (int i = key.single_size; i < key.sys_size; i++) {
		for (int j = key.single_size; j < key.sys_size; j++) {
			if (i == j) {
				int a = key.d_map[i].first;
				int b = key.d_map[i].second;
				Hal[i * sys_size + j] = cuCaddf(Hal[a * sys_size + a], Hal[b * sys_size + b]);
			}
			else {
				int a = key.d_map[i].first;
				int b = key.d_map[i].second;
				int c = key.d_map[j].first;
				int d = key.d_map[j].second;
				double x1 = delta(a, c) * (1 - delta(b, d)) * Hal[b * sys_size + d].x;
				double x2 = delta(a, d) * (1 - delta(b, c)) * Hal[b * sys_size + c].x;
				double x3 = delta(b, c) * (1 - delta(a, d)) * Hal[a * sys_size + d].x;
				double x4 = delta(b, d) * (1 - delta(a, c)) * Hal[a * sys_size + c].x;
				Hal[i * sys_size + j] = make_cuComplex(x1 + x2 + x3 + x4, 0.0);
			}
		}
	}


}

void Build_ADO_Ivalue(const param& key, std::string& current, const int cur_L, std::vector<float>& I_val) {
	if (current.length() == key.K * key.K_m) {
		if (cur_L <= key.L) {
			double I = calculate_importance_val(key, current);
			I_val.push_back(I);
		}
		return;
	}

	for (char c = 0; c <= key.L; c++) {
		if (cur_L + c <= key.L) {
			current.push_back(c);
			Build_ADO_Ivalue(key, current, cur_L + c, I_val);
			current.pop_back();
		}
	}
}

void Build_ADO(param& key, std::string& current, const int cur_L, const std::vector<float>& I_val){
	if (current.length() == key.K * key.K_m) {
		if (cur_L <= key.L) {
			float I = calculate_importance_val(key, current);
			if (I > I_val[key.cutoff_level]) {
				key.ado.push_back(current);
				/*
				for (char c : current) {
					std::cout << (int)c;
				}
				std::cout << " I: " << I << '\n';
				*/
			}
		}
		return;
	}

	for (char c = 0; c <= key.L; c++) {
		if (cur_L + c <= key.L) {
			current.push_back(c);
			Build_ADO(key, current, cur_L + c, I_val);
			current.pop_back();
		}
	}
}

void Build_ADO_map(param& key) {
	for (int i = 0; i < key.ado.size(); i++) {
		key.ado_map[key.ado[i]] = i;
	}
}

float calculate_importance_val(const param& key, std::string arr) {
	float res = 1;
	int arr_level = 0;
	for (char ele : arr) arr_level += (int)ele;
	if (arr_level == 0) return res;

	for (int i = 0; i < arr.length(); i++) {
		float c1 = 0;
		float c2 = 0;
		int idx1 = (int)arr[i] / key.K;
		int idx2 = (int)arr[i] % key.K_m;
		c1 = key.alpha[0][idx2].x / key.gamma[0][idx2].x;
		for (int j = 0; j <= idx2; j++) {
			c2 += key.gamma[0][j].x;
		}
		if (c2 != 0) res *= c1 / c2;
		else res *= c1;
	}
	return res;
}

void construct_ADO_set(param& key) {
	key.K_m = key.K_m + key.K_extra;
	string s1, s2;
	std::vector<float> I;
	I.push_back(-1);
	Build_ADO_Ivalue(key, s1, 0, I);
	sort(I.begin(), I.end());
	auto last = unique(I.begin(), I.end());
	I.erase(last, I.end());
	if (key.cutoff_level >= I.size() || key.cutoff_level < 0) {
		cout << "cutoff level should in the range 0 to " << I.size() - 1 << "\n";
		exit(EXIT_FAILURE);
	}
	Build_ADO(key, s2, 0, I);
	if (key.ado.size() <= 0) {
		cout << "The ADO set size is 0 under importance cutoff level " << key.cutoff_level << '\n';
		exit(EXIT_FAILURE);
	}
	Build_ADO_map(key);
	key.reset_rho();
}

std::unordered_map<std::string, int> Build_ADO_map(const std::vector<std::string> ado) {
	std::unordered_map<std::string, int> ado_map;
	for (int i = 0; i < ado.size(); i++) ado_map[ado[i]] = i;
	return ado_map;
}

void total_ADO_dynamics(const param& key, const data_type* d_rho_copy, data_type* d_drho, std::vector<cudaStream_t> streams) {
	int sys_size = key.sys_size;
	int total_size = sys_size * sys_size;
	cublasHandle_t cublasH = NULL;
	cublasCreate(&cublasH);

	for (int i = 0; i < key.ado.size(); i++) {
		int stream_used = i % streams.size();
		cublasSetStream(cublasH, streams[stream_used]);


		//L_s
		cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
			sys_size, sys_size, sys_size,
			&ICNT, d_rho_copy + total_size * i, sys_size, key.d_Hal, sys_size,
			&ZERO, d_drho + total_size * i, sys_size), __FILE__, __LINE__);

		cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
			sys_size, sys_size, sys_size,
			&MINUS_ICNT, key.d_Hal, sys_size, d_rho_copy + total_size * i, sys_size,
			&ONE, d_drho + total_size * i, sys_size), __FILE__, __LINE__);

		//Damp
		for (int j = 0; j < key.K; j++) {
			if ((int)key.ado[i][j] == 0) continue;
			cublasError(cublasCaxpy(cublasH, total_size, &cuCmulf(MINUS_ONE, { key.gamma[j].x * (int)key.ado[i][j], key.gamma[j].y * (int)key.ado[i][j] }), d_rho_copy + total_size * i, 1, d_drho + total_size * i, 1), __FILE__, __LINE__);
			//cudaStreamSynchronize(streams[i]);
		}

		//Upper

		for (int j = 0; j < key.K; j++) {
			if ((int)key.ado[i][j] >= key.L) continue;
			std::string tmp(key.ado[i]);
			tmp[j] += 1;
			if (key.ado_map.find(tmp) != key.ado_map.end()) {
				//cudaError(cudaDeviceSynchronize());
				//add_commutator(cublasH, ICNT, d_S + total_size * j, d_rho + total_size * ado_map.at(tmp), d_drho + total_size * i, sys_size);
				cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
					sys_size, sys_size, sys_size,
					&MINUS_ICNT, d_rho_copy + total_size * key.ado_map.at(tmp), sys_size, key.d_S + total_size * j, sys_size,
					&ONE, d_drho + total_size * i, sys_size), __FILE__, __LINE__);
				//cudaStreamSynchronize(streams[i]);
				cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
					sys_size, sys_size, sys_size,
					&ICNT, key.d_S + total_size * j, sys_size, d_rho_copy + total_size * key.ado_map.at(tmp), sys_size,
					&ONE, d_drho + total_size * i, sys_size), __FILE__, __LINE__);
				//cudaStreamSynchronize(streams[i]);
			}
		}
		//Lower

		for (int j = 0; j < key.K; j++) {
			if ((int)key.ado[i][j] == 0 || key.ado[i][j] - 1 < 0) continue;
			std::string tmp(key.ado[i]);
			tmp[j] -= 1;
			if (key.ado_map.find(tmp) != key.ado_map.end()) {
				cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
					sys_size, sys_size, sys_size,
					&cuCmulf(ICNT, { key.alpha[j].x * (int)key.ado[i][j], key.alpha[j].y * (int)key.ado[i][j] }),
					key.d_S + total_size * j, sys_size, d_rho_copy + total_size * key.ado_map.at(tmp), sys_size,
					&ONE, d_drho + total_size * i, sys_size), __FILE__, __LINE__);
				//cudaStreamSynchronize(streams[i]);
				cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
					sys_size, sys_size, sys_size,
					&cuCmulf(MINUS_ICNT, { key.alpha_t[j].x * (int)key.ado[i][j], key.alpha_t[j].y * (int)key.ado[i][j] }),
					d_rho_copy + total_size * key.ado_map.at(tmp), sys_size, key.d_S + total_size * j, sys_size,
					&ONE, d_drho + total_size * i, sys_size), __FILE__, __LINE__);
				//cudaError(cudaDeviceSynchronize());

			}
		}

	}
	//cudaError(cudaDeviceSynchronize());

	for (int i = 0; i < streams.size(); i++) cudaStreamSynchronize(streams[i]);
	cublasError(cublasDestroy(cublasH), __FILE__, __LINE__);
	
}

void total_ADO_dynamics_Ht(cublasHandle_t& cublasH, param& key, const data_type* d_rho_copy, data_type* d_drho, const data_type* d_Ht) {
	int sys_size = key.sys_size;
	int total_size = sys_size * sys_size;

	data_type* d_tmp = nullptr;
	cudaError(cudaMalloc(&d_tmp, total_size * sizeof(data_type)), __FILE__, __LINE__);


	// Loop over all ado
	for (int i = 0; i < key.ado.size(); i++) {
		
		// L_s
		cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
			sys_size, sys_size, sys_size,
			&ICNT, d_rho_copy + total_size * i, sys_size, d_Ht, sys_size,
			&ZERO, d_drho + total_size * i, sys_size), __FILE__, __LINE__);
		//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
		cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
			sys_size, sys_size, sys_size,
			&MINUS_ICNT, d_Ht, sys_size, d_rho_copy + total_size * i, sys_size,
			&ONE, d_drho + total_size * i, sys_size), __FILE__, __LINE__);
		//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
		
		
		// Damp
		if (i != 0) {
			data_type x = make_cuComplex(0.0, 0.0);
			for (int j = 0; j < key.K; j++) {
				for (int k = 0; k < key.K_m; k++) {
					int offset = k * key.K + j;
					if ((int)key.ado[i][offset] != 0) {
						x = cuCaddf(x, cuCmulf(MINUS_ONE, { key.gamma[j][k].x * (int)key.ado[i][offset], key.gamma[j][k].y * (int)key.ado[i][offset] }));
						//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
					}
				}
			}
			cublasError(cublasCaxpy_v2(cublasH, total_size, &x, d_rho_copy + total_size * i, 1, d_drho + total_size * i, 1), __FILE__, __LINE__);
			//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
		}
		
		// Upper
		
		for (int j = 0; j < key.K; j++) {
			for (int k = 0; k < key.K_m; k++) {
				std::string tmp(key.ado[i]);
				int offset = k * key.K + j;
				tmp[offset] += 1;
				if (key.ado_map.find(tmp) != key.ado_map.end()) {
					int target_idx = key.ado_map.at(tmp);
					data_type x = make_cuComplex(sqrt(((int)key.ado[i][offset] + 1) * cuComplex_abs(key.alpha[j][k])), 0.0);
					cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
						sys_size, sys_size, sys_size,
						&cuCmulf(MINUS_ICNT, x), d_rho_copy + total_size * target_idx, sys_size, key.d_S + total_size * j, sys_size,
						&ONE, d_drho + total_size * i, sys_size), __FILE__, __LINE__);
					//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
					cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
						sys_size, sys_size, sys_size,
						&cuCmulf(ICNT, x), key.d_S + total_size * j, sys_size, d_rho_copy + total_size * target_idx, sys_size,
						&ONE, d_drho + total_size * i, sys_size), __FILE__, __LINE__);
					//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
				}
			}
		}
		// Lower
		if (i != 0) {
			for (int j = 0; j < key.K; j++) {
				for (int k = 0; k < key.K_m; k++) {
					std::string tmp(key.ado[i]);
					int offset = k * key.K + j;
					tmp[offset] -= 1;
					data_type x = make_cuComplex(0.0, 0.0);
					if (key.ado_map.find(tmp) != key.ado_map.end()) {
						int target_idx = key.ado_map.at(tmp);
						data_type y = make_cuComplex(sqrt((int)key.ado[i][offset] / cuComplex_abs(key.alpha[j][k])), 0.0);
						x = cuCmulf(y, cuCmulf(ICNT, key.alpha[j][k]));
						cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
							sys_size, sys_size, sys_size,
							&x, key.d_S + total_size * j, sys_size, d_rho_copy + total_size * target_idx, sys_size,
							&ONE, d_drho + total_size * i, sys_size), __FILE__, __LINE__);
						//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
						x = cuCmulf(y, cuCmulf(MINUS_ICNT, key.alpha_t[j][k]));
						cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
							sys_size, sys_size, sys_size,
							&x, d_rho_copy + total_size * target_idx, sys_size, key.d_S + total_size * j, sys_size,
							&ONE, d_drho + total_size * i, sys_size), __FILE__, __LINE__);
						//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
					}
				}
			}
		}
	}

	// Free allocated memory
	cudaError(cudaFree(d_tmp), __FILE__, __LINE__);
}

void total_ADO_dynamics_Ht_batch(cublasHandle_t& cublasH, param& key, const data_type* d_rho_copy, data_type* d_drho, const data_type* d_Ht, const data_type* const* d_rho_batch,  data_type** d_drho_batch, data_type* d_work_space, data_type** d_work_batch, data_type* d_rho_work_space, data_type** d_rho_work_batch, vector<cudaStream_t>& streams) {
	int sys_size = key.sys_size;
	int total_size = sys_size * sys_size;

	data_type* d_tmp = nullptr;
	cudaError(cudaMalloc(&d_tmp, total_size * sizeof(data_type)), __FILE__, __LINE__);
	//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
	
	// L_s	
	cudaError(cudaMemcpy(d_work_space, d_Ht, total_size * sizeof(data_type), cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
	cublasError(cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
				sys_size, sys_size, sys_size,
				&ICNT, d_rho_batch , sys_size, d_work_batch, sys_size,
				&ZERO, d_drho_batch, sys_size, key.ado.size()), __FILE__, __LINE__);
	//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
	cublasError(cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
				sys_size, sys_size, sys_size,
				&MINUS_ICNT, d_work_batch, sys_size, d_rho_batch, sys_size,
				&ONE, d_drho_batch, sys_size, key.ado.size()), __FILE__, __LINE__);
	//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
	
	// Upper
	cudaError(cudaMemset(d_rho_work_space, 0.0, total_size * sizeof(data_type)), __FILE__, __LINE__);
	for (int j = 0; j < key.K; j++) {
		cublasError(cublasCcopy(cublasH, total_size, key.d_S + total_size * j, 1, d_work_space, 1), __FILE__, __LINE__);
		for (int k = 0; k < key.K_m; k++) {
			int offset = k * key.K + j;
			vector<data_type*> ptr_tmp;
			for (int i = 0; i < key.ado.size(); i++) {
				std::string tmp(key.ado[i]);
				tmp[offset] += 1;
				if (key.ado_map.find(tmp) != key.ado_map.end()) {
					int target_idx = key.ado_map.at(tmp);
					ptr_tmp.push_back(key.d_rho_ptr[target_idx]);
				}
				else ptr_tmp.push_back(d_rho_work_space);
			}
			cudaError(cudaMemcpy(d_rho_work_batch, ptr_tmp.data(), key.ado.size() * sizeof(data_type*), cudaMemcpyHostToDevice), __FILE__, __LINE__);

			cublasError(cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
				sys_size, sys_size, sys_size,
				&MINUS_ICNT, d_rho_work_batch, sys_size, d_work_batch, sys_size,
				&ONE, d_drho_batch, sys_size, key.ado.size()), __FILE__, __LINE__);
			//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
			cublasError(cublasCgemmBatched(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
				sys_size, sys_size, sys_size,
				&ICNT, d_work_batch, sys_size, d_rho_work_batch, sys_size,
				&ONE, d_drho_batch, sys_size, key.ado.size()), __FILE__, __LINE__);
			//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
		}
	}
	
	// Damp
	for (int i = 0; i < key.ado.size(); i++) {
		if (i != 0) {
			data_type x = make_cuComplex(0.0, 0.0);
			for (int j = 0; j < key.K; j++) {
				for (int k = 0; k < key.K_m; k++) {
					int offset = k * key.K + j;
					if ((int)key.ado[i][offset] != 0) {
						x = cuCaddf(x, cuCmulf(MINUS_ONE, { key.gamma[j][k].x * (int)key.ado[i][offset], key.gamma[j][k].y * (int)key.ado[i][offset] }));
						//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
					}
				}
			}
			cublasError(cublasCaxpy_v2(cublasH, total_size, &x, d_rho_copy + total_size * i, 1, d_drho + total_size * i, 1), __FILE__, __LINE__);
			//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
		}

		// Lower
		if (i != 0) {
			for (int j = 0; j < key.K; j++) {
				for (int k = 0; k < key.K_m; k++) {
					std::string tmp(key.ado[i]);
					int offset = k * key.K + j;
					tmp[offset] -= 1;
					data_type x = make_cuComplex(0.0, 0.0);
					if (key.ado_map.find(tmp) != key.ado_map.end()) {
						int target_idx = key.ado_map.at(tmp);
						x = cuCmulf(ICNT, key.alpha[j][k]);
						cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
							sys_size, sys_size, sys_size,
							&x, key.d_S + total_size * j, sys_size, d_rho_copy + total_size * target_idx, sys_size,
							&ONE, d_drho + total_size * i, sys_size), __FILE__, __LINE__);
						//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
						x = cuCmulf(MINUS_ICNT, key.alpha_t[j][k]);
						cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
							sys_size, sys_size, sys_size,
							&x, d_rho_copy + total_size * target_idx, sys_size, key.d_S + total_size * j, sys_size,
							&ONE, d_drho + total_size * i, sys_size), __FILE__, __LINE__);
						//cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
					}
				}
			}
		}
	}
	// Free allocated memory
	cudaError(cudaFree(d_tmp), __FILE__, __LINE__);
}

void dynamics_solver(param& key, std::vector<int>& sites, vector<vector<double>>& population) {
	int sys_size = key.sys_size;
	int total_size = sys_size * sys_size;
	if (sites.size() > sys_size) {
		std::cout << "number of output sites should not over the system size " << sys_size << "." << '\n';
		exit(EXIT_FAILURE);
	}
	for (int site : sites) {
		if (site > sys_size || site < 0) {
			std::cout << "site " << site << " is out of range" << '\n';
			exit(EXIT_FAILURE);
		}
	}

	std::vector<std::ofstream> files;
	std::vector<std::string> file_name(sites.size());
	for (int i = 0; i < sites.size(); i++) file_name[i] = "site " + std::to_string(sites[i]) + " population.txt";
	for (const auto& name : file_name) files.emplace_back(name);

	int stream_num = 16;
	std::vector<cudaStream_t> streams(stream_num);
	for (int i = 0; i < stream_num; i++) cudaError(cudaStreamCreate(&streams[i]), __FILE__, __LINE__);
	

	std::vector<float> b = { 1.0 / 6, 2.0 / 6, 2.0 / 6, 1.0 / 6 };
	cublasHandle_t cublasH;
	cublasCreate(&cublasH);
	//cudaDeviceSynchronize();
	data_type* d_rho_copy = nullptr;
	cudaError(cudaMalloc(&d_rho_copy, total_size * key.ado.size() * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_k1 = nullptr;
	cudaError(cudaMalloc(&d_k1, total_size * key.ado.size() * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_k2 = nullptr;
	cudaError(cudaMalloc(&d_k2, total_size * key.ado.size() * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_k3 = nullptr;
	cudaError(cudaMalloc(&d_k3, total_size * key.ado.size() * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_k4 = nullptr;
	cudaError(cudaMalloc(&d_k4, total_size * key.ado.size() * sizeof(data_type)), __FILE__, __LINE__);

	int ct = 0;
	for (float t = 0; t <= key.t_end; t += key.step_size) {
		cudaMemcpy(d_rho_copy, key.d_rho, total_size * key.ado.size() * sizeof(data_type), cudaMemcpyDeviceToDevice);
		cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
		for (int j = 0; j < sites.size(); j++) {
			data_type r;
			To_Host(key.d_rho[0], r, (sites[j] - 1) * (sys_size + 1));
			population[j][ct] = r.x;
		}
		ct++;

		//printf("%f\n", r.x);
		//cudaMemcpy(d_rho_copy, d_rho, total_size * ado.size() * sizeof(data_type), cudaMemcpyDeviceToDevice);
		//f = drho/dt = f(rho)
		//k1 = f(rho)
		//cudaDeviceSynchronize();
		total_ADO_dynamics(key, d_rho_copy, d_k1, streams);

		//k2 = f(rho + b[1][0]*h*k1)
		cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(key.step_size * 0.5, 0.0), d_k1, 1, d_rho_copy, 1);
		cudaDeviceSynchronize();
		total_ADO_dynamics(key, d_rho_copy, d_k2, streams);
		cudaMemcpy(d_rho_copy, key.d_rho, total_size * key.ado.size() * sizeof(data_type), cudaMemcpyDeviceToDevice);
		cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

		//k3 = f(rho + b[2][0]*h*k1 + b[2][1]*h*k2)
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(key.step_size * 0.5, 0.0), d_k2, 1, d_rho_copy, 1), __FILE__, __LINE__);
		cudaDeviceSynchronize();
		total_ADO_dynamics(key, d_rho_copy, d_k3, streams);
		cudaMemcpy(d_rho_copy, key.d_rho, total_size * key.ado.size() * sizeof(data_type), cudaMemcpyDeviceToDevice);
		cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

		//k4 = f(rho + b[3][0]*h*k1 + b[3][1]*h*k2+ b[3][2]*h*k3)
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(key.step_size, 0.0), d_k3, 1, d_rho_copy, 1), __FILE__, __LINE__);
		cudaDeviceSynchronize();
		total_ADO_dynamics(key, d_rho_copy, d_k4, streams);
		cudaMemcpy(d_rho_copy, key.d_rho, total_size * key.ado.size() * sizeof(data_type), cudaMemcpyDeviceToDevice);
		cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

		//new rho
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(key.step_size * b[0], 0.0), d_k1, 1, key.d_rho, 1), __FILE__, __LINE__);
		cudaDeviceSynchronize();
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(key.step_size * b[1], 0.0), d_k2, 1, key.d_rho, 1), __FILE__, __LINE__);
		cudaDeviceSynchronize();
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(key.step_size * b[2], 0.0), d_k3, 1, key.d_rho, 1), __FILE__, __LINE__);
		cudaDeviceSynchronize();
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(key.step_size * b[3], 0.0), d_k4, 1, key.d_rho, 1), __FILE__, __LINE__);
		cudaDeviceSynchronize();
	}
	for (int i = 0; i < stream_num; i++) cudaError(cudaStreamDestroy(streams[i]), __FILE__, __LINE__);
	for (int i = 0; i < files.size(); i++) files[i].close();
	cudaFree(d_k1);
	cudaFree(d_k2);
	cudaFree(d_k3);
	cudaFree(d_k4);
	cudaFree(d_rho_copy);
	cublasDestroy(cublasH);
}


void twoD_spectrum_solver(param& key, const data_type* d_H, const int nv1, const int nv2, const int nv3, std::vector<data_type>& polarization) {
	int sys_size = key.sys_size;
	int total_size = sys_size * sys_size;
	key.reset_rho();

	cublasHandle_t cublasH;
	cublasCreate(&cublasH);
	 

	std::vector<float> b = { 1.0 / 6, 2.0 / 6, 2.0 / 6, 1.0 / 6 };

	vector<data_type*> d_X(4);
	vector<data_type> X(total_size, make_cuComplex(0.0, 0.0));
	for (int i = 0; i < 4; i++) d_X[i] = To_Device(X);
	
	
	compute_pulse_interaction(key, d_X);

	dim3 tpb(ThreadPerBlock, ThreadPerBlock); // threads per block
	dim3 numblocks((sys_size + tpb.x - 1) / tpb.x, (sys_size + tpb.y - 1) / tpb.y);


	data_type* d_X_site_trans_2;
	cudaError(cudaMalloc(&d_X_site_trans_2, total_size * sizeof(data_type)), __FILE__, __LINE__);
	cudaError(cudaMemcpy(d_X_site_trans_2, d_X[1], total_size * sizeof(data_type), cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
	inplace_transpose << <numblocks, tpb >> > (d_X_site_trans_2, sys_size);
	cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
	data_type* d_X_site_trans_3;
	cudaError(cudaMalloc(&d_X_site_trans_3, total_size * sizeof(data_type)), __FILE__, __LINE__);
	cudaError(cudaMemcpy(d_X_site_trans_3, d_X[2], total_size * sizeof(data_type), cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
	inplace_transpose << <numblocks, tpb >> > (d_X_site_trans_3, sys_size);
	cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);


	
	//data_type* d_rho_copy = nullptr;
	//cudaError(cudaMalloc(&d_rho_copy, key.ado.size() * total_size * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_k1 = nullptr;
	cudaError(cudaMalloc(&d_k1, total_size * key.ado.size() * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_k2 = nullptr;
	cudaError(cudaMalloc(&d_k2, total_size * key.ado.size() * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_k3 = nullptr;
	cudaError(cudaMalloc(&d_k3, total_size * key.ado.size() * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_k4 = nullptr;
	cudaError(cudaMalloc(&d_k4, total_size * key.ado.size() * sizeof(data_type)), __FILE__, __LINE__);

	data_type* d_V1 = nullptr;
	cudaError(cudaMalloc(&d_V1, total_size * sizeof(data_type)), __FILE__, __LINE__);
	cudaError(cudaMemset(d_V1, 0.0, total_size * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_V2 = nullptr;
	cudaError(cudaMalloc(&d_V2, total_size * sizeof(data_type)), __FILE__, __LINE__);
	cudaError(cudaMemset(d_V2, 0.0, total_size * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_V3 = nullptr;
	cudaError(cudaMalloc(&d_V3, total_size * sizeof(data_type)), __FILE__, __LINE__);
	cudaError(cudaMemset(d_V3, 0.0, total_size * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_V = nullptr;
	cudaError(cudaMalloc(&d_V, total_size * sizeof(data_type)), __FILE__, __LINE__);
	cudaError(cudaMemset(d_V, 0.0, total_size * sizeof(data_type)), __FILE__, __LINE__);


	data_type* d_P = nullptr;
	cudaError(cudaMalloc(&d_P, total_size * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_ONE = nullptr;
	cudaError(cudaMalloc(&d_ONE, sys_size * sizeof(data_type)), __FILE__, __LINE__);
	init_cuComplex << <numblocks.x, tpb.x >> > (d_ONE, sys_size, ONE);
	cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
	data_type tr;
	data_type* d_tr = nullptr;
	cudaError(cudaMalloc(&d_tr, sizeof(data_type)), __FILE__, __LINE__);

	//std::ofstream file1("site1.txt");
	//std::ofstream file2("site2.txt");

	int nstep = (key.t_end - key.t_start) / key.step_size;
	float t = key.t_start;
	int print_step = (int)(key.print_step / key.step_size);

	for (int i = 0; i <= nstep; i++) {
		cudaError(cudaMemcpy(d_V, d_H, total_size * sizeof(data_type), cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
		// pulse 1
		if (nv1 == 1 && fabs(t - key.pulses[0].tau0) <= key.pulses[0].taup * 2) {
			build_V_matrix(cublasH, key, d_X[0], d_V1, t, 0);
			cublasError(cublasCaxpy_v2(cublasH, total_size, &MINUS_ONE, d_V1, 1, d_V, 1), __FILE__, __LINE__);
		}
		// pulse 2
		if (nv2 == 1 && fabs(t - key.pulses[1].tau0) <= key.pulses[1].taup * 2) {
			build_V_dagger_matrix(cublasH, key, d_X_site_trans_2, d_V2, t, 1);
			cublasError(cublasCaxpy_v2(cublasH, total_size, &MINUS_ONE, d_V2, 1, d_V, 1), __FILE__, __LINE__);
		}
		// pulse 3
		if (nv3 == 1 && fabs(t - key.pulses[2].tau0) <= key.pulses[2].taup * 2) {
			build_V_dagger_matrix(cublasH, key, d_X_site_trans_3, d_V3, t, 2);
			cublasError(cublasCaxpy_v2(cublasH, total_size, &MINUS_ONE, d_V3, 1, d_V, 1), __FILE__, __LINE__);
		}

		
		if (t >= key.pulses[2].tau0 && i % print_step == 0) {
			cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
				sys_size, sys_size, sys_size,
				&ONE, d_X[3], sys_size,
				key.d_rho, sys_size,
				&ZERO, d_P, sys_size), __FILE__, __LINE__);
			///////////////////////////////////////////////
			// Tr(X*rho)
			cublasError(cublasCdotu_v2(cublasH, sys_size, d_P, sys_size + 1, d_ONE, 1, d_tr), __FILE__, __LINE__);
			cudaError(cudaMemcpy(&tr, d_tr, sizeof(data_type), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
			polarization.push_back(tr);
			///////////////////////////////////////////////
		}

		//f = drho/dt
		//k1 = f(rho)
		total_ADO_dynamics_Ht(cublasH, key, key.d_rho, d_k1, d_V);

		//k2 = f(rho + b[1][0]*h*k1)
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(key.step_size * 0.5, 0.0), d_k1, 1, key.d_rho, 1), __FILE__, __LINE__);
		total_ADO_dynamics_Ht(cublasH, key, key.d_rho, d_k2, d_V);
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(-1 * key.step_size * 0.5, 0.0), d_k1, 1, key.d_rho, 1), __FILE__, __LINE__);

		//k3 = f(rho + b[2][0]*h*k1 + b[2][1]*h*k2)
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(key.step_size * 0.5, 0.0), d_k2, 1, key.d_rho, 1), __FILE__, __LINE__);
		total_ADO_dynamics_Ht(cublasH, key, key.d_rho, d_k3, d_V);
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(-1 * key.step_size * 0.5, 0.0), d_k2, 1, key.d_rho, 1), __FILE__, __LINE__);

		//k4 = f(rho + b[3][0]*h*k1 + b[3][1]*h*k2+ b[3][2]*h*k3)
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(key.step_size, 0.0), d_k3, 1, key.d_rho, 1), __FILE__, __LINE__);
		total_ADO_dynamics_Ht(cublasH, key, key.d_rho, d_k4, d_V);
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(-1 * key.step_size, 0.0), d_k3, 1, key.d_rho, 1), __FILE__, __LINE__);

		//new rho
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(key.step_size * b[0], 0.0), d_k1, 1, key.d_rho, 1), __FILE__, __LINE__);
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(key.step_size * b[1], 0.0), d_k2, 1, key.d_rho, 1), __FILE__, __LINE__);
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(key.step_size * b[2], 0.0), d_k3, 1, key.d_rho, 1), __FILE__, __LINE__);
		cublasError(cublasCaxpy_v2(cublasH, total_size * key.ado.size(), &make_cuComplex(key.step_size * b[3], 0.0), d_k4, 1, key.d_rho, 1), __FILE__, __LINE__);
		t += key.step_size;
	}

	cublasError(cublasDestroy(cublasH), __FILE__, __LINE__);
	for (int i = 0; i < 4; i++) cudaError(cudaFree(d_X[i]), __FILE__, __LINE__);
	cudaError(cudaFree(d_X_site_trans_2), __FILE__, __LINE__);
	cudaError(cudaFree(d_X_site_trans_3), __FILE__, __LINE__);
	cudaError(cudaFree(d_k1), __FILE__, __LINE__);
	cudaError(cudaFree(d_k2), __FILE__, __LINE__);
	cudaError(cudaFree(d_k3), __FILE__, __LINE__);
	cudaError(cudaFree(d_k4), __FILE__, __LINE__);
	//cudaError(cudaFree(d_rho_copy), __FILE__, __LINE__);
	cudaError(cudaFree(d_V1), __FILE__, __LINE__);
	cudaError(cudaFree(d_V2), __FILE__, __LINE__);
	cudaError(cudaFree(d_V3), __FILE__, __LINE__);
	cudaError(cudaFree(d_V), __FILE__, __LINE__);
	cudaError(cudaFree(d_P), __FILE__, __LINE__);
	cudaError(cudaFree(d_ONE), __FILE__, __LINE__);
	cudaError(cudaFree(d_tr), __FILE__, __LINE__);
	
	//file1.close();
	//file2.close();
	
}

void propagation_Ht_batch(param& key, const data_type* d_H, const int nv1, const int nv2, const int nv3, std::vector<data_type>& polarization) {
	int sys_size = key.sys_size;
	int total_size = sys_size * sys_size;
	key.reset_rho();
	key.reset_rho_batch();

	cublasHandle_t cublasH = NULL;
	cublasCreate(&cublasH);

	int stream_num = 16;
	vector<cudaStream_t> streams(stream_num);
	for (int i = 0; i < stream_num; i++) cudaError(cudaStreamCreate(&streams[i]), __FILE__, __LINE__);

	vector<float> b = { 1.0 / 6, 2.0 / 6, 2.0 / 6, 1.0 / 6 };

	vector<data_type*> d_X(4);
	vector<data_type> X(total_size, make_cuComplex(0.0, 0.0));
	for (int i = 0; i < 4; i++) d_X[i] = To_Device(X);


	compute_pulse_interaction(key, d_X);

	dim3 tpb(ThreadPerBlock, ThreadPerBlock); // threads per block
	dim3 numblocks((sys_size + tpb.x - 1) / tpb.x, (sys_size + tpb.y - 1) / tpb.y);


	data_type* d_X_site_trans_2;
	cudaError(cudaMalloc(&d_X_site_trans_2, total_size * sizeof(data_type)), __FILE__, __LINE__);
	cudaError(cudaMemcpy(d_X_site_trans_2, d_X[1], total_size * sizeof(data_type), cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
	inplace_transpose << <numblocks, tpb >> > (d_X_site_trans_2, sys_size);
	cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
	data_type* d_X_site_trans_3;
	cudaError(cudaMalloc(&d_X_site_trans_3, total_size * sizeof(data_type)), __FILE__, __LINE__);
	cudaError(cudaMemcpy(d_X_site_trans_3, d_X[2], total_size * sizeof(data_type), cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
	inplace_transpose << <numblocks, tpb >> > (d_X_site_trans_3, sys_size);
	cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);

	/*
	vector<data_type> A(total_size);
	To_Host(d_X[0], A);
	print_matrix(A, sys_size);
	puts("");
	*/

	//data_type* d_rho_copy = nullptr;
	//cudaError(cudaMalloc(&d_rho_copy, key.ado.size() * total_size * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_k1 = nullptr;
	cudaError(cudaMalloc(&d_k1, key.ado.size() * total_size * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_k2 = nullptr;
	cudaError(cudaMalloc(&d_k2, key.ado.size() * total_size * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_k3 = nullptr;
	cudaError(cudaMalloc(&d_k3, key.ado.size() * total_size * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_k4 = nullptr;
	cudaError(cudaMalloc(&d_k4, key.ado.size() * total_size * sizeof(data_type)), __FILE__, __LINE__);

	vector<data_type*> d_k1_ptr;
	for (int i = 0; i < key.ado.size(); i++) d_k1_ptr.push_back(d_k1 + total_size * i);
	vector<data_type*> d_k2_ptr;
	for (int i = 0; i < key.ado.size(); i++) d_k2_ptr.push_back(d_k2 + total_size * i);
	vector<data_type*> d_k3_ptr;
	for (int i = 0; i < key.ado.size(); i++) d_k3_ptr.push_back(d_k3 + total_size * i);
	vector<data_type*> d_k4_ptr;
	for (int i = 0; i < key.ado.size(); i++) d_k4_ptr.push_back(d_k4 + total_size * i);

	data_type** d_k1_batch = nullptr;
	data_type** d_k2_batch = nullptr;
	data_type** d_k3_batch = nullptr;
	data_type** d_k4_batch = nullptr;
	cudaError(cudaMalloc(&d_k1_batch, key.ado.size() * sizeof(data_type*)), __FILE__, __LINE__);
	cudaError(cudaMemcpy(d_k1_batch, d_k1_ptr.data(), key.ado.size() * sizeof(data_type*), cudaMemcpyHostToDevice), __FILE__, __LINE__);
	cudaError(cudaMalloc(&d_k2_batch, key.ado.size() * sizeof(data_type*)), __FILE__, __LINE__);
	cudaError(cudaMemcpy(d_k2_batch, d_k2_ptr.data(), key.ado.size() * sizeof(data_type*), cudaMemcpyHostToDevice), __FILE__, __LINE__);
	cudaError(cudaMalloc(&d_k3_batch, key.ado.size() * sizeof(data_type*)), __FILE__, __LINE__);
	cudaError(cudaMemcpy(d_k3_batch, d_k3_ptr.data(), key.ado.size() * sizeof(data_type*), cudaMemcpyHostToDevice), __FILE__, __LINE__);
	cudaError(cudaMalloc(&d_k4_batch, key.ado.size() * sizeof(data_type*)), __FILE__, __LINE__);
	cudaError(cudaMemcpy(d_k4_batch, d_k4_ptr.data(), key.ado.size() * sizeof(data_type*), cudaMemcpyHostToDevice), __FILE__, __LINE__);


	data_type* d_work;
	cudaError(cudaMalloc(&d_work, total_size * sizeof(data_type)), __FILE__, __LINE__);
	vector<data_type*> d_work_ptr;
	for (int i = 0; i < key.ado.size(); i++) d_work_ptr.push_back(d_work);
	data_type** d_work_batch = nullptr;
	cudaError(cudaMalloc(&d_work_batch, key.ado.size() * sizeof(data_type*)), __FILE__, __LINE__);
	cudaError(cudaMemcpy(d_work_batch, d_work_ptr.data(), key.ado.size() * sizeof(data_type*), cudaMemcpyHostToDevice), __FILE__, __LINE__);

	data_type* d_rho_work;
	cudaError(cudaMalloc(&d_rho_work, key.ado.size() * total_size * sizeof(data_type)), __FILE__, __LINE__);
	vector<data_type*> d_rho_work_ptr;
	for (int i = 0; i < key.ado.size(); i++) d_rho_work_ptr.push_back(d_rho_work);
	data_type** d_rho_work_batch = nullptr;
	cudaError(cudaMalloc(&d_rho_work_batch, key.ado.size() * sizeof(data_type*)), __FILE__, __LINE__);
	for (int i = 0; i < key.ado.size(); i++) cudaError(cudaMemcpy(d_rho_work_batch, d_rho_work_ptr.data(), key.ado.size() * sizeof(data_type*), cudaMemcpyHostToDevice), __FILE__, __LINE__);


	data_type* d_V1 = nullptr;
	cudaError(cudaMalloc(&d_V1, total_size * sizeof(data_type)), __FILE__, __LINE__);
	cudaError(cudaMemset(d_V1, 0.0, total_size * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_V2 = nullptr;
	cudaError(cudaMalloc(&d_V2, total_size * sizeof(data_type)), __FILE__, __LINE__);
	cudaError(cudaMemset(d_V2, 0.0, total_size * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_V3 = nullptr;
	cudaError(cudaMalloc(&d_V3, total_size * sizeof(data_type)), __FILE__, __LINE__);
	cudaError(cudaMemset(d_V3, 0.0, total_size * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_V = nullptr;
	cudaError(cudaMalloc(&d_V, total_size * sizeof(data_type)), __FILE__, __LINE__);
	cudaError(cudaMemset(d_V, 0.0, total_size * sizeof(data_type)), __FILE__, __LINE__);


	data_type* d_P = nullptr;
	cudaError(cudaMalloc(&d_P, total_size * sizeof(data_type)), __FILE__, __LINE__);
	data_type* d_ONE = nullptr;
	cudaError(cudaMalloc(&d_ONE, sys_size * sizeof(data_type)), __FILE__, __LINE__);
	init_cuComplex << <numblocks.x, tpb.x >> > (d_ONE, sys_size, ONE);
	cudaError(cudaDeviceSynchronize(), __FILE__, __LINE__);
	data_type tr;
	data_type* d_tr = nullptr;
	cudaError(cudaMalloc(&d_tr, sizeof(data_type)), __FILE__, __LINE__);

	//std::ofstream file1("site1.txt");
	//std::ofstream file2("site2.txt");

	int nstep = (key.t_end - key.t_start) / key.step_size;
	float t = key.t_start;
	int print_step = (int)(key.print_step / key.step_size);

	for (int i = 0; i <= nstep; i++) {
		cudaError(cudaMemcpy(d_V, d_H, total_size * sizeof(data_type), cudaMemcpyDeviceToDevice), __FILE__, __LINE__);
		// pulse 1
		if (nv1 == 1 && fabs(t - key.pulses[0].tau0) <= key.pulses[0].taup * 2) {
			build_V_matrix(cublasH, key, d_X[0], d_V1, t, 0);
			cublasError(cublasCaxpy_v2(cublasH, total_size, &MINUS_ONE, d_V1, 1, d_V, 1), __FILE__, __LINE__);
		}
		// pulse 2
		if (nv2 == 1 && fabs(t - key.pulses[1].tau0) <= key.pulses[1].taup * 2) {
			build_V_dagger_matrix(cublasH, key, d_X_site_trans_2, d_V2, t, 1);
			cublasError(cublasCaxpy_v2(cublasH, total_size, &MINUS_ONE, d_V2, 1, d_V, 1), __FILE__, __LINE__);
		}
		// pulse 3
		if (nv3 == 1 && fabs(t - key.pulses[2].tau0) <= key.pulses[2].taup * 2) {
			build_V_dagger_matrix(cublasH, key, d_X_site_trans_3, d_V3, t, 2);
			cublasError(cublasCaxpy_v2(cublasH, total_size, &MINUS_ONE, d_V3, 1, d_V, 1), __FILE__, __LINE__);
		}

		/*
		cout << "t = " << t << '\n';
		vector<data_type> X(total_size);
		To_Host(key.d_rho, X);
		print_matrix(X, 1);
		puts("");
		*/


		if (t >= key.pulses[2].tau0 && i % print_step == 0) {
			cublasError(cublasCgemm3m(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
				sys_size, sys_size, sys_size,
				&ONE, d_X[3], sys_size,
				key.d_rho, sys_size,
				&ZERO, d_P, sys_size), __FILE__, __LINE__);
			///////////////////////////////////////////////
			// Tr(X*rho)
			cublasError(cublasCdotu_v2(cublasH, sys_size, d_P, sys_size + 1, d_ONE, 1, d_tr), __FILE__, __LINE__);
			cudaError(cudaMemcpy(&tr, d_tr, sizeof(data_type), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
			polarization.push_back(tr);
			///////////////////////////////////////////////
		}

		//f = drho/dt
		//k1 = f(rho)
		total_ADO_dynamics_Ht_batch(cublasH, key, key.d_rho, d_k1, d_V, key.d_rho_batch, d_k1_batch, d_work, d_work_batch, d_rho_work, d_rho_work_batch, streams);

		//k2 = f(rho + b[1][0]*h*k1)
		cublasError(cublasCaxpy_v2(cublasH, key.ado.size() * total_size, &make_cuComplex(key.step_size * 0.5, 0.0), d_k1, 1, key.d_rho, 1), __FILE__, __LINE__);
		total_ADO_dynamics_Ht_batch(cublasH, key, key.d_rho, d_k2, d_V, key.d_rho_batch, d_k2_batch, d_work, d_work_batch, d_rho_work, d_rho_work_batch, streams);
		cublasError(cublasCaxpy_v2(cublasH, key.ado.size() * total_size, &make_cuComplex(-1 * key.step_size * 0.5, 0.0), d_k1, 1, key.d_rho, 1), __FILE__, __LINE__);

		//k3 = f(rho + b[2][0]*h*k1 + b[2][1]*h*k2)
		cublasError(cublasCaxpy_v2(cublasH, key.ado.size() * total_size, &make_cuComplex(key.step_size * 0.5, 0.0), d_k2, 1, key.d_rho, 1), __FILE__, __LINE__);
		total_ADO_dynamics_Ht_batch(cublasH, key, key.d_rho, d_k3, d_V, key.d_rho_batch, d_k3_batch, d_work, d_work_batch, d_rho_work, d_rho_work_batch, streams);
		cublasError(cublasCaxpy_v2(cublasH, key.ado.size() * total_size, &make_cuComplex(-1 * key.step_size * 0.5, 0.0), d_k2, 1, key.d_rho, 1), __FILE__, __LINE__);

		//k4 = f(rho + b[3][0]*h*k1 + b[3][1]*h*k2+ b[3][2]*h*k3)
		cublasError(cublasCaxpy_v2(cublasH, key.ado.size() * total_size, &make_cuComplex(key.step_size, 0.0), d_k3, 1, key.d_rho, 1), __FILE__, __LINE__);
		total_ADO_dynamics_Ht_batch(cublasH, key, key.d_rho, d_k4, d_V, key.d_rho_batch, d_k4_batch, d_work, d_work_batch, d_rho_work, d_rho_work_batch, streams);
		cublasError(cublasCaxpy_v2(cublasH, key.ado.size() * total_size, &make_cuComplex(-1 * key.step_size, 0.0), d_k3, 1, key.d_rho, 1), __FILE__, __LINE__);

		//new rho
		cublasError(cublasCaxpy_v2(cublasH, key.ado.size() * total_size, &make_cuComplex(key.step_size * b[0], 0.0), d_k1, 1, key.d_rho, 1), __FILE__, __LINE__);
		cublasError(cublasCaxpy_v2(cublasH, key.ado.size() * total_size, &make_cuComplex(key.step_size * b[1], 0.0), d_k2, 1, key.d_rho, 1), __FILE__, __LINE__);
		cublasError(cublasCaxpy_v2(cublasH, key.ado.size() * total_size, &make_cuComplex(key.step_size * b[2], 0.0), d_k3, 1, key.d_rho, 1), __FILE__, __LINE__);
		cublasError(cublasCaxpy_v2(cublasH, key.ado.size() * total_size, &make_cuComplex(key.step_size * b[3], 0.0), d_k4, 1, key.d_rho, 1), __FILE__, __LINE__);
		t += key.step_size;
	}

	cublasError(cublasDestroy(cublasH), __FILE__, __LINE__);
	for (int i = 0; i < stream_num; i++) cudaError(cudaStreamDestroy(streams[i]), __FILE__, __LINE__);
	for (int i = 0; i < 4; i++) cudaError(cudaFree(d_X[i]), __FILE__, __LINE__);
	cudaError(cudaFree(d_X_site_trans_2), __FILE__, __LINE__);
	cudaError(cudaFree(d_X_site_trans_3), __FILE__, __LINE__);
	cudaError(cudaFree(d_k1), __FILE__, __LINE__);
	cudaError(cudaFree(d_k2), __FILE__, __LINE__);
	cudaError(cudaFree(d_k3), __FILE__, __LINE__);
	cudaError(cudaFree(d_k4), __FILE__, __LINE__);
	cudaError(cudaFree(d_work), __FILE__, __LINE__);
	cudaError(cudaFree(d_k1_batch), __FILE__, __LINE__);
	cudaError(cudaFree(d_k2_batch), __FILE__, __LINE__);
	cudaError(cudaFree(d_k3_batch), __FILE__, __LINE__);
	cudaError(cudaFree(d_k4_batch), __FILE__, __LINE__);
	cudaError(cudaFree(d_work_batch), __FILE__, __LINE__);
	cudaError(cudaFree(d_rho_work), __FILE__, __LINE__);
	cudaError(cudaFree(d_rho_work_batch), __FILE__, __LINE__);
	cudaError(cudaFree(d_V1), __FILE__, __LINE__);
	cudaError(cudaFree(d_V2), __FILE__, __LINE__);
	cudaError(cudaFree(d_V3), __FILE__, __LINE__);
	cudaError(cudaFree(d_V), __FILE__, __LINE__);
	cudaError(cudaFree(d_P), __FILE__, __LINE__);
	cudaError(cudaFree(d_ONE), __FILE__, __LINE__);
	cudaError(cudaFree(d_tr), __FILE__, __LINE__);

	//file1.close();
	//file2.close();
}



