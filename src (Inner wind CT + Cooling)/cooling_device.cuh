// cooling_device.cuh
#pragma once
#include "cooling.h"
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__
double Y_of_T_dev(double T, const CoolingTablesDevice& tab)
{
    double logT = log10(T);
    double logT_max = tab.logT_min + (tab.n - 1) * tab.dlogT;
    if (logT < tab.logT_min) logT = tab.logT_min;
    if (logT > logT_max)     logT = logT_max;
    double t = (logT - tab.logT_min) / tab.dlogT;
    int i = (int)t;
    if (i < 0) i = 0;
    if (i > tab.n - 2) i = tab.n - 2;
    double w = t - i;
    double v0 = __ldg(&tab.Y_of_T[i]);
    double v1 = __ldg(&tab.Y_of_T[i + 1]);
    return v0 + w * (v1 - v0);
}

__device__ __forceinline__
double T_of_Y_dev(double Y_star, const CoolingTablesDevice& tab,
    double T_floor = 1.0e4)
{
    // Y_of_T строго убывает по индексу i (растёт T -> убывает Y)
    if (Y_star >= __ldg(&tab.Y_of_T[0]))
        return pow(10.0, tab.logT_min);
    if (Y_star <= __ldg(&tab.Y_of_T[tab.n - 1]))
        return pow(10.0, tab.logT_min + (tab.n - 1) * tab.dlogT);

    int lo = 0, hi = tab.n - 1;
#pragma unroll 1
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (__ldg(&tab.Y_of_T[mid]) > Y_star) lo = mid;
        else                                   hi = mid;
    }
    double Y0 = __ldg(&tab.Y_of_T[lo]);
    double Y1 = __ldg(&tab.Y_of_T[hi]);
    double w = (Y0 - Y_star) / (Y0 - Y1);
    double logT = tab.logT_min + (lo + w) * tab.dlogT;
    double T = pow(10.0, logT);
    return (T < T_floor) ? T_floor : T;
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