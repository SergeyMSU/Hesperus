// cooling.cu
#include "cooling.h"
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>

void load_cooling_tables(CoolingTablesHost& host)
{
    // ---- параметры сетки ----
    std::ifstream p("cooling_params.txt");
    if (!p) { std::cerr << "cannot open cooling_params.txt\n"; std::exit(1); }
    std::string key; double val;
    while (p >> key >> val) 
    {
        if (key == "logT_min")   host.logT_min = val;
        else if (key == "logT_max")   host.logT_max = val;
        else if (key == "Y_min")      host.Y_min = val;
        else if (key == "Y_max")      host.Y_max = val;
        else if (key == "T_ref")      host.T_ref = val;
        else if (key == "Lambda_ref") host.Lambda_ref = val;
        else if (key == "N")          host.n = (int)val;
        else if (key == "N_Y")        host.n_y = (int)val;
    }
    host.dlogT = (host.logT_max - host.logT_min) / (host.n - 1);
    host.dY = (host.Y_max - host.Y_min) / (host.n_y - 1);

    // ---- lambda_table.txt ----
    {
        std::ifstream f("lambda_table.txt");
        std::string line; int i = 0;
        while (std::getline(f, line) && i < host.n) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            double lT, lL;
            if (iss >> lT >> lL) host.logLambda[i++] = lL;
        }
    }
    // ---- Y_table.txt ----
    {
        std::ifstream f("Y_table.txt");
        std::string line; int i = 0;
        while (std::getline(f, line) && i < host.n) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            double lT, yv;
            if (iss >> lT >> yv) host.Y_of_T[i++] = yv;
        }
    }
    // ---- Yinv_table.txt ----
    {
        std::ifstream f("Yinv_table.txt");
        std::string line; int i = 0;
        while (std::getline(f, line) && i < host.n_y) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            double yv, tv;
            if (iss >> yv >> tv) host.T_of_Y[i++] = tv;
        }
    }
    std::cout << "[cooling] loaded tables: n=" << host.n
        << " n_y=" << host.n_y
        << " logT=[" << host.logT_min << ", " << host.logT_max << "]\n";
}

void free_host_tables(CoolingTablesHost& /*host*/) {}

CoolingTablesDevice to_device(const CoolingTablesHost& host)
{
    CoolingTablesDevice dev;
    dev.logT_min = host.logT_min;
    dev.dlogT = host.dlogT;
    dev.Y_min = host.Y_min;
    dev.dY = host.dY;
    dev.T_ref = host.T_ref;
    dev.Lambda_ref = host.Lambda_ref;
    dev.n = host.n;
    dev.n_y = host.n_y;

    double* d1, * d2, * d3;
    cudaMalloc(&d1, host.n * sizeof(double));
    cudaMalloc(&d2, host.n * sizeof(double));
    cudaMalloc(&d3, host.n_y * sizeof(double));

    cudaMemcpy(d1, host.logLambda, host.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d2, host.Y_of_T, host.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d3, host.T_of_Y, host.n_y * sizeof(double), cudaMemcpyHostToDevice);

    dev.logLambda = d1;
    dev.Y_of_T = d2;
    dev.T_of_Y = d3;
    return dev;
}

void free_device_tables(CoolingTablesDevice& dev)
{
    if (dev.logLambda) cudaFree(const_cast<double*>(dev.logLambda));
    if (dev.Y_of_T)    cudaFree(const_cast<double*>(dev.Y_of_T));
    if (dev.T_of_Y)    cudaFree(const_cast<double*>(dev.T_of_Y));
    dev.logLambda = nullptr;
    dev.Y_of_T = nullptr;
    dev.T_of_Y = nullptr;
}