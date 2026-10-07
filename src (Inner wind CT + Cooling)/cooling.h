#pragma once
// cooling.h
#pragma once
#include <string>

// –азмеры фиксированы на этапе компил€ции, чтобы структуру можно
// было передавать в device по значению.
constexpr int N_TABLE = 1024;   // точек по log10(T)
constexpr int N_TABLE_Y = 1024;   // точек по Y

struct CoolingTablesHost 
{
    // –авномерна€ сетка по log10(T) от logT_min до logT_max
    double logLambda[N_TABLE];   // log10( Lambda(T) )
    double Y_of_T[N_TABLE];    // Y(T)

    // –авномерна€ сетка по Y от Y_min до Y_max
    double T_of_Y[N_TABLE_Y];  // T(Y)

    // ѕараметры сеток и нормировки
    double logT_min, logT_max, dlogT;
    double Y_min, Y_max, dY;
    double T_ref, Lambda_ref;
    int    n, n_y;
};

// Device-верси€: только указатели на device-массивы
struct CoolingTablesDevice 
{
    const double* logLambda;
    const double* Y_of_T;
    const double* T_of_Y;
    double logT_min, dlogT;
    double Y_min, dY;
    double T_ref, Lambda_ref;
    int    n, n_y;
};

// --- host ---
void load_cooling_tables(CoolingTablesHost& host);

void free_host_tables(CoolingTablesHost& host);

CoolingTablesDevice to_device(const CoolingTablesHost& host);

void free_device_tables(CoolingTablesDevice& dev);