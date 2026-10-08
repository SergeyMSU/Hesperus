#pragma once
#include <string>

constexpr int N_TABLE = 1024;

struct CoolingTablesHost {
    double logLambda[N_TABLE];
    double Y_of_T[N_TABLE];
    double logT_min, logT_max, dlogT;
    double T_ref, Lambda_ref;
    int    n;
};

struct CoolingTablesDevice {
    const double* logLambda;
    const double* Y_of_T;
    double logT_min, dlogT;
    double T_ref, Lambda_ref;
    int    n;
};

void load_cooling_tables(CoolingTablesHost& host);
void free_host_tables(CoolingTablesHost& host);

CoolingTablesDevice to_device(const CoolingTablesHost& host);
void free_device_tables(CoolingTablesDevice& dev);