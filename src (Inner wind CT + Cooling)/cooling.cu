#include "cooling.h"
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cmath>

void load_cooling_tables(CoolingTablesHost& host)
{
    auto fail = [](const std::string& msg) {
        std::cerr << "[cooling] ERROR: " << msg << "\n";
        std::exit(1);
        };

    std::ifstream p("cooling_params.txt");
    if (!p) fail("cannot open cooling_params.txt");
    std::string key; double val; int seen = 0;
    const int required = 5;  // logT_min, logT_max, T_ref, Lambda_ref, N
    while (p >> key >> val) {
        if (key == "logT_min") { host.logT_min = val; ++seen; }
        else if (key == "logT_max") { host.logT_max = val; ++seen; }
        else if (key == "T_ref") { host.T_ref = val; ++seen; }
        else if (key == "Lambda_ref") { host.Lambda_ref = val; ++seen; }
        else if (key == "N") { host.n = (int)val; ++seen; }
    }
    if (seen < required) fail("params: missing keys");
    if (host.n != N_TABLE) fail("N != N_TABLE");
    if (host.logT_max <= host.logT_min) fail("logT_max <= logT_min");
    if (host.T_ref <= 0.0 || host.Lambda_ref <= 0.0) fail("bad refs");

    host.dlogT = (host.logT_max - host.logT_min) / (host.n - 1);

    auto read_col = [&](const char* fname, int n, double* out) {
        std::ifstream f(fname);
        if (!f) fail(std::string("cannot open ") + fname);
        std::string line; int i = 0;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            double a, b;
            if (!(iss >> a >> b)) continue;
            if (i >= n) fail(std::string(fname) + ": too many rows");
            out[i++] = b;
        }
        if (i != n) fail(std::string(fname) + ": wrong row count");
        };
    read_col("lambda_table.txt", host.n, host.logLambda);
    read_col("Y_table.txt", host.n, host.Y_of_T);

    // Проверки
    for (int i = 0; i < host.n; ++i) {
        if (!std::isfinite(host.logLambda[i])) fail("logLambda non-finite");
        if (host.logLambda[i] < -30.0 || host.logLambda[i] > -10.0)
            fail("logLambda out of [-30,-10]");
        if (!std::isfinite(host.Y_of_T[i])) fail("Y_of_T non-finite");
    }
    for (int i = 1; i < host.n; ++i)
        if (host.Y_of_T[i] >= host.Y_of_T[i - 1])
            fail("Y_of_T not strictly decreasing at i=" + std::to_string(i));

    // Closure test с бинарным поиском (как в device)
    auto Y_of_T_h = [&](double T) {
        double lT = std::log10(T);
        double t = (lT - host.logT_min) / host.dlogT;
        int i = std::min(std::max((int)t, 0), host.n - 2);
        double w = t - i;
        return host.Y_of_T[i] * (1.0 - w) + host.Y_of_T[i + 1] * w;
        };
    auto T_of_Y_h = [&](double Y) {
        int lo = 0, hi = host.n - 1;
        if (Y >= host.Y_of_T[0]) return std::pow(10.0, host.logT_min);
        if (Y <= host.Y_of_T[hi]) return std::pow(10.0,
            host.logT_min + hi * host.dlogT);
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (host.Y_of_T[mid] > Y) lo = mid;
            else                       hi = mid;
        }
        double Y0 = host.Y_of_T[lo], Y1 = host.Y_of_T[hi];
        double w = (Y0 - Y) / (Y0 - Y1);
        return std::pow(10.0, host.logT_min + (lo + w) * host.dlogT);
        };

    const int n_chk = 8;
    const double T_chk[n_chk] = { 1.5e4, 5e4, 1.5e5, 5e5,
                                 1.5e6, 5e6, 2e7, 1e8 };
    double max_err = 0.0;
    for (int k = 0; k < n_chk; ++k) {
        double T = T_chk[k];
        double Y = Y_of_T_h(T);
        double Tb = T_of_Y_h(Y);
        double rel = std::abs(Tb - T) / T;
        max_err = std::max(max_err, rel);
    }

    std::cout << "[cooling] loaded tables OK\n";
    std::cout << "  N=" << host.n
        << "  logT=[" << host.logT_min << ","
        << host.logT_max << "]  dlogT=" << host.dlogT << "\n";
    std::cout << "  Y=[" << host.Y_of_T[host.n - 1] << ","
        << host.Y_of_T[0] << "]\n";
    std::cout << "  T_ref=" << host.T_ref
        << "  Lambda_ref=" << host.Lambda_ref << "\n";
    std::cout << "  closure max rel.err = " << max_err << "\n";

    
}

void free_host_tables(CoolingTablesHost&) {}

CoolingTablesDevice to_device(const CoolingTablesHost& host)
{
    CoolingTablesDevice dev;
    dev.logT_min = host.logT_min;
    dev.dlogT = host.dlogT;
    dev.T_ref = host.T_ref;
    dev.Lambda_ref = host.Lambda_ref;
    dev.n = host.n;

    double* d1, * d2;
    cudaMalloc(&d1, host.n * sizeof(double));
    cudaMalloc(&d2, host.n * sizeof(double));
    cudaMemcpy(d1, host.logLambda, host.n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d2, host.Y_of_T, host.n * sizeof(double), cudaMemcpyHostToDevice);
    dev.logLambda = d1;
    dev.Y_of_T = d2;
    return dev;
}

void free_device_tables(CoolingTablesDevice& dev)
{
    if (dev.logLambda) cudaFree(const_cast<double*>(dev.logLambda));
    if (dev.Y_of_T)    cudaFree(const_cast<double*>(dev.Y_of_T));
    dev.logLambda = nullptr;
    dev.Y_of_T = nullptr;
}