// cooling_device.cuh
#pragma once
#include "cooling.h"
#include <cuda_runtime.h>
#include <math.h>

// ------------------------------------------------------------------
// Универсальный интерполятор по равномерной сетке.
// Читает ТОЛЬКО два соседних значения из global memory (через __ldg).
// ------------------------------------------------------------------
__device__ __forceinline__
double interp_uniform(double x, const double* __restrict__ vals,
    double x_min, double dx, int n)
{
    double t = (x - x_min) / dx;
    if (t <= 0.0)              return __ldg(&vals[0]);
    if (t >= (double)(n - 1))  return __ldg(&vals[n - 1]);
    int i = (int)t;
    double w = t - i;
    double v0 = __ldg(&vals[i]);
    double v1 = __ldg(&vals[i + 1]);
    return v0 + w * (v1 - v0);
}

// ------------------------------------------------------------------
// Lambda(T), erg cm^3 s^-1.
// T < 1e4  ->  floor
// T > 1e8  ->  аналитика  Lambda = 2.3e-19 * T^{-0.54}
// ------------------------------------------------------------------
__device__ __forceinline__
double Lambda_of_T(double T, const CoolingTablesDevice& tab)
{
    if (T < 1.0e4) T = 1.0e4;
    double logT = log10(T);

    if (logT > 8.0) {
        // log10(2.3e-19) = -18.638
        return pow(10.0, -18.638 - 0.54 * logT);
    }
    double logLam = interp_uniform(logT, tab.logLambda,
        tab.logT_min, tab.dlogT, tab.n);
    return pow(10.0, logLam);
}

// ------------------------------------------------------------------
// Y(T)  — temporal evolution function
// ------------------------------------------------------------------
__device__ __forceinline__
double Y_of_T_dev(double T, const CoolingTablesDevice& tab)
{
    double logT = log10(T);
    double logT_max = tab.logT_min + (tab.n - 1) * tab.dlogT;
    if (logT < tab.logT_min) logT = tab.logT_min;
    if (logT > logT_max)     logT = logT_max;
    return interp_uniform(logT, tab.Y_of_T,
        tab.logT_min, tab.dlogT, tab.n);
}

// ------------------------------------------------------------------
// Y^{-1}(Y) -> T.  Возвращает T_floor, если Y > Y_max
// (газ остыл бы ниже 10^4 K).
// ------------------------------------------------------------------
__device__ __forceinline__
double T_of_Y_dev(double Y, const CoolingTablesDevice& tab,
    double T_floor = 1.0e4)
{
    double Y_max = tab.Y_min + (tab.n_y - 1) * tab.dY;
    if (Y >= Y_max) return T_floor;
    if (Y <= tab.Y_min) Y = tab.Y_min;
    return interp_uniform(Y, tab.T_of_Y,
        tab.Y_min, tab.dY, tab.n_y);
}

// ------------------------------------------------------------------
// Полный шаг охлаждения для одной ячейки (EI-схема).
// Возвращает T^{n+1}.
//
// Все величины в CGS:
//   rho    - g/cm^3
//   mu, mu_e, mu_H - БЕЗРАЗМЕРНЫЕ средние молекулярные веса
//                    (в единицах m_p). Если у вас они уже в граммах,
//                    уберите m_p из формулы.
// ------------------------------------------------------------------
__device__ __forceinline__
double apply_cooling(double T_n, double dt, double rho,
    double gamma, double mu, double mu_e, double mu_H,
    const CoolingTablesDevice& tab,
    double T_floor = 1.0e4)
{
    constexpr double kB = 1.380649e-16;   // erg/K
    constexpr double m_p = 1.67262192e-24; // g

    // Y_star = Y(T_n) + C * dt,
    // C = (Lambda_ref/T_ref) * (gamma-1) * rho * mu / (kB * m_p * mu_e * mu_H)
    double C = (tab.Lambda_ref / tab.T_ref)
        * (gamma - 1.0) * rho * mu
        / (kB * m_p * mu_e * mu_H);

    double Y_n = Y_of_T_dev(T_n, tab);
    double Y_star = Y_n + C * dt;
    double T_new = T_of_Y_dev(Y_star, tab, T_floor);

    if (T_new < T_floor) T_new = T_floor;
    return T_new;
}