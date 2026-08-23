#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Cell.h"
#include "Header.h"
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "Konstruktor.h"
#include "Cuda_main.cu"
#include "Kyb.h"
#include <string>

#include <time.h> 


#define ER_S(x) printf("Standart error in kernel.cu: kod - x\n")
#define TVD_ true //false
#define TVQ_ true
#define kor_Sol true

#define sss 500000000

using namespace std;

cudaError_t addWithCuda(void);
cudaError_t addWithCuda_G_D(void);

__device__ void rotate_z(double& x0, double& y0, double& z0, double& x, double& y, double& z, double alph)
{
    x =x0*__cosf(alph)-y0*__sinf(alph);
    y =x0* __sinf(alph)+y0* __cosf(alph);
    z =z0;
}

__device__ void rotate_y(double& x0, double& y0, double& z0, double& x, double& y, double& z, double alph)
{
    z = z0 * __cosf(alph) + x0 * __sinf(alph);
    x = -z0 * __sinf(alph) + x0 * __cosf(alph);
    y = y0;
}

__device__ double polar_angle(double x, double y)
{
    if (x < 0)
    {
        return atan(y / x) + 1.0 * PI;
    }
    else if (x > 0 && y >= 0)
    {
        return atan(y / x);
    }
    else if (x > 0 && y < 0)
    {
        return atan(y / x) + 2.0 * PI;
    }
    else if (y > 0 && x >= 0 && x <= 0)
    {
        return PI / 2.0;
    }
    else if (y < 0 && x >= 0 && x <= 0)
    {
        return  3.0 * PI / 2.0;
    }
    return 0.0;
}

__device__ void spherical_skorost(const double& z, const double& x, const double& y, const double& Vz, const double& Vx, 
    const double& Vy, double& Vr, double& Vphi, double& Vtheta)
{
    double r_1, the_1, phi_1;

    r_1 = sqrt(x * x + y * y + z * z);
    the_1 = acos(z / r_1);
    phi_1 = polar_angle(x, y);

    Vr = Vx * sin(the_1) * cos(phi_1) + Vy * sin(the_1) * sin(phi_1) + Vz * cos(the_1);
    Vtheta = Vx * cos(the_1) * cos(phi_1) + Vy * cos(the_1) * sin(phi_1) - Vz * sin(the_1);
    Vphi = -Vx * sin(phi_1) + Vy * cos(phi_1);
}

__device__ void dekard_skorost(const double& z, const double& x, const double& y, const double& Vr,
    const double& Vphi, const double& Vtheta, double& Vz, double& Vx, double& Vy)
{
    double r_2, the_2, phi_2;

    r_2 = sqrt(x * x + y * y + z * z);
    the_2 = acos(z / r_2);
    phi_2 = polar_angle(x, y);

    Vx = Vr * sin(the_2) * cos(phi_2) + Vtheta * cos(the_2) * cos(phi_2) - Vphi * sin(phi_2);
    Vy = Vr * sin(the_2) * sin(phi_2) + Vtheta * cos(the_2) * sin(phi_2) + Vphi * cos(phi_2);
    Vz = Vr * cos(the_2) - Vtheta * sin(the_2);
}

__device__ double Lya(double T)
{
    if (T < 1.0)
    {
        return 0.0;
    }
    else if (T < 10.0)
    {
        return 570.0 * pow(T, 0.55);
    }
    else if (T < 4000)
    {
        return 8903.4 * pow(T, -0.6);
    }
    else
    {
        return 0.9006 * pow(T, 0.5);
    }
}



__device__ void transfer(double x0, double y0, double z0, double x1, double y1, double z1, double u, double v, double w,//
    double& uu, double& vv, double& ww)
{
    double alph1 = polar_angle(x0, y0);
    double alph2 = polar_angle(__dsqrt_rn(x0*x0 + y0 *y0), z0);
    double x, y, z;
    double x2, y2, z2;
    double X, Y, Z;
    double X2, Y2, Z2;
    double U, V, W;
    rotate_z(x0, y0, z0, x, y, z, -alph1);
    rotate_y(x, y, z, x2, y2, z2, -alph2);
    X = x2;
    Y = y2;
    Z = z2;
    rotate_z(x1, y1, z1, x, y, z, -alph1);
    rotate_y(x, y, z, x2, y2, z2, -alph2);
    X2 = x2;
    Y2 = y2;
    Z2 = z2;
    rotate_z(u, v, w, x, y, z, -alph1);
    rotate_y(x, y, z, x2, y2, z2, -alph2);
    U = x2;
    V = y2;
    W = z2;
    double alph3 = polar_angle(X2, Y2);
    double alph4 = polar_angle(__dsqrt_rn(X2 * X2 + Y2 * Y2), Z2);
    rotate_y(U, V, W, x, y, z, alph4);
    rotate_z(x, y, z, U, V, W, alph3);
    rotate_y(U, V, W, x, y, z, alph2);
    rotate_z(x, y, z, U, V, W, alph1);
    uu = U;
    vv = V;
    ww = W;
}


__device__ double HLLC_my(double& ro_L, double& p_L, double& v1_L, double& v2_L, double& v3_L,//
    double& ro_R, double& p_R, double& v1_R, double& v2_R, double& v3_R,//
    double* P, double& n1, double& n2, double& n3, double& rad)
{
    double t = 10000.0;

    double e_L, e_R;
    double Vkv_L, Vkv_R;
    double c_L, c_R;

    Vkv_L = v1_L * v1_L + v2_L * v2_L + v3_L * v3_L;
    Vkv_R = v1_R * v1_R + v2_R * v2_R + v3_R * v3_R;
    c_L = __dsqrt_rn(ggg * p_L / ro_L);
    c_R = __dsqrt_rn(ggg * p_R / ro_R);
    e_L = p_L / (ggg - 1.0) + ro_L * Vkv_L / 2.0;  /// ѕолна€ энерги€ слева
    e_R = p_R / (ggg - 1.0) + ro_R * Vkv_R / 2.0;  /// ѕолна€ энерги€ справа

    double Vn_L = v1_L * n1 + v2_L * n2 + v3_L * n3;
    double Vn_R = v1_R * n1 + v2_R * n2 + v3_R * n3;

    double D_L = min(Vn_L, Vn_R) - max(c_L, c_R);
    double D_R = max(Vn_L, Vn_R) + max(c_L, c_R);
    ///double D_L = min(Vn_L - c_L, Vn_R - c_R);
    ///double D_R = max(Vn_L + c_L, Vn_R + c_R);
    t = min(t, krit * rad / max(fabs(D_L), fabs(D_R)));
    //t = min(t,krit*rad_R/max(fabs(D_L),fabs(D_R)));

    double fx1 = ro_L * v1_L;
    double fx2 = ro_L * v1_L * v1_L + p_L;
    double fx3 = ro_L * v1_L * v2_L;
    double fx4 = ro_L * v1_L * v3_L;
    double fx5 = (e_L + p_L) * v1_L;

    double fy1 = ro_L * v2_L;
    double fy2 = ro_L * v1_L * v2_L;
    double fy3 = ro_L * v2_L * v2_L + p_L;
    double fy4 = ro_L * v2_L * v3_L;
    double fy5 = (e_L + p_L) * v2_L;

    double fz1 = ro_L * v3_L;
    double fz2 = ro_L * v1_L * v3_L;
    double fz3 = ro_L * v2_L * v3_L;
    double fz4 = ro_L * v3_L * v3_L + p_L;
    double fz5 = (e_L + p_L) * v3_L;

    double fl_1 = fx1 * n1 + fy1 * n2 + fz1 * n3;
    double fl_2 = fx2 * n1 + fy2 * n2 + fz2 * n3;
    double fl_3 = fx3 * n1 + fy3 * n2 + fz3 * n3;
    double fl_4 = fx4 * n1 + fy4 * n2 + fz4 * n3;
    double fl_5 = fx5 * n1 + fy5 * n2 + fz5 * n3;

    if (D_L > Omega)
    {
        P[0] = fl_1 - Omega * ro_L; /// Ќужно будет домножить на площадь грани и шаг по времени
        P[1] = fl_2 - Omega * ro_L * v1_L;
        P[2] = fl_3 - Omega * ro_L * v2_L;
        P[3] = fl_4 - Omega * ro_L * v3_L;
        P[4] = fl_5 - Omega * e_L;
        return t;
    }

    double hx1 = ro_R * v1_R;
    double hx2 = ro_R * v1_R * v1_R + p_R;
    double hx3 = ro_R * v1_R * v2_R;
    double hx4 = ro_R * v1_R * v3_R;
    double hx5 = (e_R + p_R) * v1_R;

    double hy1 = ro_R * v2_R;
    double hy2 = ro_R * v1_R * v2_R;
    double hy3 = ro_R * v2_R * v2_R + p_R;
    double hy4 = ro_R * v2_R * v3_R;
    double hy5 = (e_R + p_R) * v2_R;

    double hz1 = ro_R * v3_R;
    double hz2 = ro_R * v1_R * v3_R;
    double hz3 = ro_R * v2_R * v3_R;
    double hz4 = ro_R * v3_R * v3_R + p_R;
    double hz5 = (e_R + p_R) * v3_R;

    double fr_1 = hx1 * n1 + hy1 * n2 + hz1 * n3;
    double fr_2 = hx2 * n1 + hy2 * n2 + hz2 * n3;
    double fr_3 = hx3 * n1 + hy3 * n2 + hz3 * n3;
    double fr_4 = hx4 * n1 + hy4 * n2 + hz4 * n3;
    double fr_5 = hx5 * n1 + hy5 * n2 + hz5 * n3;

    if (D_R < Omega)
    {
        P[0] = fr_1 - Omega * ro_R; /// Ќужно будет домножить на площадь грани и шаг по времени
        P[1] = fr_2 - Omega * ro_R * v1_R;
        P[2] = fr_3 - Omega * ro_R * v2_R;
        P[3] = fr_4 - Omega * ro_R * v3_R;
        P[4] = fr_5 - Omega * e_R;
        return t;
    }

    /// _______________________________________________________________________________________________________________________________________

    double u_L = Vn_L;
    double u_R = Vn_R;

    double D_C = ((D_R - u_R) * ro_R * u_R - (D_L - u_L) * ro_L * u_L - p_R + p_L) / ((D_R - u_R) * ro_R - (D_L - u_L) * ro_L);

    double roro_L = ro_L * ((D_L - u_L) / (D_L - D_C));
    double roro_R = ro_R * ((D_R - u_R) / (D_R - D_C));

    /// Ќаходим давление в центральной области (оно одинаковое слева и справа)
    double P_T = (p_L * ro_R * (u_R - D_R) - p_R * ro_L * (u_L - D_L) - ro_L * ro_R * (u_L - D_L) * (u_R - D_R) * (u_R - u_L)) / (ro_R * (u_R - D_R) - ro_L * (u_L - D_L));

    if (D_L <= Omega && D_C >= Omega)  /// ѕопали во вторую область (слева)
    {
        double Vx = v1_L + (D_C - Vn_L) * n1;
        double Vy = v2_L + (D_C - Vn_L) * n2;
        double Vz = v3_L + (D_C - Vn_L) * n3;

        //double ee_L = P_T/(ggg - 1.0) + roro_L*(Vx*Vx + Vy*Vy + Vz*Vz)/2.0;
        //double ee_L = e_L - ((P_T - p_L)/2.0)*(1/roro_L - 1/ro_L);
        double ee_L = ((D_L - u_L) * e_L - p_L * u_L + P_T * D_C) / (D_L - D_C);

        double dq1 = roro_L - ro_L;
        double dq2 = roro_L * Vx - ro_L * v1_L;
        double dq3 = roro_L * Vy - ro_L * v2_L;
        double dq4 = roro_L * Vz - ro_L * v3_L;
        double dq5 = ee_L - e_L;

        P[0] = D_L * dq1 + fl_1 - Omega * roro_L; /// Ќужно будет домножить на площадь грани и шаг по времени
        P[1] = D_L * dq2 + fl_2 - Omega * roro_L * Vx;
        P[2] = D_L * dq3 + fl_3 - Omega * roro_L * Vy;
        P[3] = D_L * dq4 + fl_4 - Omega * roro_L * Vz;
        P[4] = D_L * dq5 + fl_5 - Omega * ee_L;
        return t;
    }
    else if (D_R >= Omega && D_C <= Omega)  /// ѕопали во вторую область (справа)
    {
        double Vx = v1_R + (D_C - Vn_R) * n1;
        double Vy = v2_R + (D_C - Vn_R) * n2;
        double Vz = v3_R + (D_C - Vn_R) * n3;

        //double ee_R = P_T/(ggg - 1.0) + roro_R*(Vx*Vx + Vy*Vy + Vz*Vz)/2.0;
        double ee_R = ((D_R - u_R) * e_R - p_R * u_R + P_T * D_C) / (D_R - D_C);

        double dq1 = roro_R - ro_R;
        double dq2 = roro_R * Vx - ro_R * v1_R;
        double dq3 = roro_R * Vy - ro_R * v2_R;
        double dq4 = roro_R * Vz - ro_R * v3_R;
        double dq5 = ee_R - e_R;

        P[0] = D_R * dq1 + fr_1 - Omega * roro_R; /// Ќужно будет домножить на площадь грани и шаг по времени
        P[1] = D_R * dq2 + fr_2 - Omega * roro_R * Vx;
        P[2] = D_R * dq3 + fr_3 - Omega * roro_R * Vy;
        P[3] = D_R * dq4 + fr_4 - Omega * roro_R * Vz;
        P[4] = D_R * dq5 + fr_5 - Omega * ee_R;
        return t;
    }

    return t;
}

__device__ double HLLC_Aleksashov(double& ro_L, double& p_L, double& v1_L, double& v2_L, double& v3_L,//
    double& ro_R, double& p_R, double& v1_R, double& v2_R, double& v3_R,//
    double* P, double& n1, double& n2, double& n3, double& rad)
{
    double n[3];
    n[0] = n1;
    n[1] = n2;
    n[2] = n3;
    //int id_bn = 1;
    //int n_state = 1;
    double FR[8], FL[8];
    double UL[8], UZ[8], UR[8];
    double UZL[8], UZR[8];

    double vL[3], vR[3], bL[3], bR[3];
    double vzL[3], vzR[3], bzL[3], bzR[3];
    double qv[3];
    double aco[3][3];

    double wv = 0.0;
    double r1 = ro_L;
    double u1 = v1_L;
    double v1 = v2_L;
    double w1 = v3_L;
    double p1 = p_L;
    double bx1 = 0.0;
    double by1 = 0.0;
    double bz1 = 0.0;


    double r2 = ro_R;
    double u2 = v1_R;
    double v2 = v2_R;
    double w2 = v3_R;
    double p2 = p_R;
    double bx2 = 0.0;
    double by2 = 0.0;
    double bz2 = 0.0;

    double ro = (r2 + r1) / 2.0;
    double ap = (p2 + p1) / 2.0;
    double abx = (bx2 + bx1) / 2.0;
    double aby = (by2 + by1) / 2.0;
    double abz = (bz2 + bz1) / 2.0;


    double bk = abx * n[0] + aby * n[1] + abz * n[2];
    double b2 = kv(abx) + kv(aby) + kv(abz);

    double d = b2 - kv(bk);
    aco[0][0] = n[0];
    aco[1][0] = n[1];
    aco[2][0] = n[2];
    if (d > eps)
    {
        d = __dsqrt_rn(d);
        aco[0][1] = (abx - bk * n[0]) / d;
        aco[1][1] = (aby - bk * n[1]) / d;
        aco[2][1] = (abz - bk * n[2]) / d;
        aco[0][2] = (aby * n[2] - abz * n[1]) / d;
        aco[1][2] = (abz * n[0] - abx * n[2]) / d;
        aco[2][2] = (abx * n[1] - aby * n[0]) / d;
    }
    else
    {
        double aix, aiy, aiz;
        if ((fabs(n[0]) < fabs(n[1])) && (fabs(n[0]) < fabs(n[2])))
        {
            aix = 1.0;
            aiy = 0.0;
            aiz = 0.0;
        }
        else if (fabs(n[1]) < fabs(n[2]))
        {
            aix = 0.0;
            aiy = 1.0;
            aiz = 0.0;
        }
        else
        {
            aix = 0.0;
            aiy = 0.0;
            aiz = 1.0;
        }

        double aik = aix * n[0] + aiy * n[1] + aiz * n[2];
        d = __dsqrt_rn(1.0 - kv(aik));
        aco[0][1] = (aix - aik * n[0]) / d;
        aco[1][1] = (aiy - aik * n[1]) / d;
        aco[2][1] = (aiz - aik * n[2]) / d;
        aco[0][2] = (aiy * n[2] - aiz * n[1]) / d;
        aco[1][2] = (aiz * n[0] - aix * n[2]) / d;
        aco[2][2] = (aix * n[1] - aiy * n[0]) / d;
    }

    for (int i = 0; i < 3; i++)
    {
        vL[i] = aco[0][i] * u1 + aco[1][i] * v1 + aco[2][i] * w1;
        vR[i] = aco[0][i] * u2 + aco[1][i] * v2 + aco[2][i] * w2;
        bL[i] = aco[0][i] * bx1 + aco[1][i] * by1 + aco[2][i] * bz1;
        bR[i] = aco[0][i] * bx2 + aco[1][i] * by2 + aco[2][i] * bz2;
    }

    double aaL = bL[0] / __dsqrt_rn(r1);
    double b2L = kv(bL[0]) + kv(bL[1]) + kv(bL[2]);
    double b21 = b2L / r1;
    double cL = __dsqrt_rn(ga * p1 / r1);
    double qp = __dsqrt_rn(b21 + cL * (cL + 2.0 * aaL));
    double qm = __dsqrt_rn(b21 + cL * (cL - 2.0 * aaL));
    double cfL = (qp + qm) / 2.0;
    double ptL = p1 + b2L / 2.0;

    double aaR = bR[0] / __dsqrt_rn(r2);
    double b2R = kv(bR[0]) + kv(bR[1]) + kv(bR[2]);
    double b22 = b2R / r2;
    double cR = __dsqrt_rn(ga * p2 / r2);
    qp = __dsqrt_rn(b22 + cR * (cR + 2.0 * aaR));
    qm = __dsqrt_rn(b22 + cR * (cR - 2.0 * aaR));
    double cfR = (qp + qm) / 2.0;
    double ptR = p2 + b2R / 2.0;

    double aC = (aaL + aaR) / 2.0;
    double b2o = (b22 + b21) / 2.0;
    double cC = __dsqrt_rn(ga * ap / ro);
    qp = __dsqrt_rn(b2o + cC * (cC + 2.0 * aC));
    qm = __dsqrt_rn(b2o + cC * (cC - 2.0 * aC));
    double cfC = (qp + qm) / 2.0;
    double vC1 = (vL[0] + vR[0]) / 2.0;

    double SL = min((vL[0] - cfL), (vR[0] - cfR));
    double SR = max((vL[0] + cfL), (vR[0] + cfR));

    double suR = SR - vR[0];
    double suL = SL - vL[0];
    double SM = (suR * r2 * vR[0] - ptR + ptL - suL * r1 * vL[0]) / (suR * r2 - suL * r1);

    if (SR <= SL)
    {
        printf("231\n");
    }

    double SM00 = SM;
    double SR00 = SR;
    double SL00 = SL;
    double SM01, SR01, SL01;
    if ((SM00 >= SR00) || (SM00 <= SL00))
    {
        SL = min((vL[0] - cfL), (vR[0] - cfR));
        SR = max((vL[0] + cfL), (vR[0] + cfR));
        suR = SR - vR[0];
        suL = SL - vL[0];
        SM = (suR * r2 * vR[0] - ptR + ptL - suL * r1 * vL[0]) / (suR * r2 - suL * r1);
        SM01 = SM;
        SR01 = SR;
        SL01 = SL;
        if ((SM01 >= SR01) || (SM01 <= SL01))
        {
            printf("251\n");
        }
    }


    double UU = max(fabs(SL), fabs(SR));
    double time = krit * rad / UU;

    double upt1 = (kv(u1) + kv(v1) + kv(w1)) / 2.0;
    double sbv1 = u1 * bx1 + v1 * by1 + w1 * bz1;

    double upt2 = (kv(u2) + kv(v2) + kv(w2)) / 2.0;
    double sbv2 = u2 * bx2 + v2 * by2 + w2 * bz2;

    double e1 = p1 / g1 + r1 * upt1 + b2L / 2.0;
    double e2 = p2 / g1 + r2 * upt2 + b2R / 2.0;

    FL[0] = r1 * vL[0];
    FL[1] = r1 * vL[0] * vL[0] + ptL - kv(bL[0]);
    FL[2] = r1 * vL[0] * vL[1] - bL[0] * bL[1];
    FL[3] = r1 * vL[0] * vL[2] - bL[0] * bL[2];
    FL[4] = (e1 + ptL) * vL[0] - bL[0] * sbv1;
    FL[5] = 0.0;
    FL[6] = vL[0] * bL[1] - vL[1] * bL[0];
    FL[7] = vL[0] * bL[2] - vL[2] * bL[0];

    FR[0] = r2 * vR[0];
    FR[1] = r2 * vR[0] * vR[0] + ptR - kv(bR[0]);
    FR[2] = r2 * vR[0] * vR[1] - bR[0] * bR[1];
    FR[3] = r2 * vR[0] * vR[2] - bR[0] * bR[2];
    FR[4] = (e2 + ptR) * vR[0] - bR[0] * sbv2;
    FR[5] = 0.0;
    FR[6] = vR[0] * bR[1] - vR[1] * bR[0];
    FR[7] = vR[0] * bR[2] - vR[2] * bR[0];

    UL[0] = r1;
    UL[4] = e1;
    UR[0] = r2;
    UR[4] = e2;


    for (int ik = 0; ik < 3; ik++)
    {
        UL[ik + 1] = r1 * vL[ik];
        UL[ik + 5] = bL[ik];
        UR[ik + 1] = r2 * vR[ik];
        UR[ik + 5] = bR[ik];
    }

    for (int ik = 0; ik < 8; ik++)
    {
        UZ[ik] = (SR * UR[ik] - SL * UL[ik] + FL[ik] - FR[ik]) / (SR - SL);
    }

    double suRm = suR / (SR - SM);
    double suLm = suL / (SL - SM);
    double rzR = r2 * suRm;
    double rzL = r1 * suLm;
    vzR[0] = SM;
    vzL[0] = SM;
    double ptzR = ptR + r2 * suR * (SM - vR[0]);
    double ptzL = ptL + r1 * suL * (SM - vL[0]);
    double ptz = (ptzR + ptzL) / 2.0;
    bzR[0] = UZ[5];
    bzL[0] = UZ[5];

    vzR[1] = UZ[2] / UZ[0];
    vzR[2] = UZ[3] / UZ[0];
    vzL[1] = vzR[1];
    vzL[2] = vzR[2];

    vzR[1] = vR[1] + UZ[5] * (bR[1] - UZ[6]) / suR / r2;
    vzR[2] = vR[2] + UZ[5] * (bR[2] - UZ[7]) / suR / r2;
    vzL[1] = vL[1] + UZ[5] * (bL[1] - UZ[6]) / suL / r1;
    vzL[2] = vL[2] + UZ[5] * (bL[2] - UZ[7]) / suL / r1;

    bzR[1] = UZ[6];
    bzR[2] = UZ[7];
    bzL[1] = bzR[1];
    bzL[2] = bzR[2];

    double sbvz = (UZ[5] * UZ[1] + UZ[6] * UZ[2] + UZ[7] * UZ[3]) / UZ[0];

    double ezR = e2 * suRm + (ptz * SM - ptR * vR[0] + UZ[5] * (sbv2 - sbvz)) / (SR - SM);
    double ezL = e1 * suLm + (ptz * SM - ptL * vL[0] + UZ[5] * (sbv1 - sbvz)) / (SL - SM);

    if (fabs(UZ[5]) < eps)
    {
        vzR[1] = vR[1];
        vzR[2] = vR[2];
        vzL[1] = vL[1];
        vzL[2] = vL[2];
        bzR[1] = bR[1] * suRm;
        bzR[2] = bR[2] * suRm;
        bzL[1] = bL[1] * suLm;
        bzL[2] = bL[2] * suLm;
    }
    UZL[0] = rzL;
    UZL[4] = ezL;
    UZR[0] = rzR;
    UZR[4] = ezR;

    for (int ik = 0; ik < 3; ik++)
    {
        UZL[ik + 1] = vzL[ik] * rzL;
        UZL[ik + 5] = bzL[ik];
        UZR[ik + 1] = vzR[ik] * rzR;
        UZR[ik + 5] = bzR[ik];
    }

    if (SL > wv)
    {
        P[0] = FL[0] - wv * UL[0];
        P[4] = FL[4] - wv * UL[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FL[ik] - wv * UL[ik];
        }
    }
    else if ((SL <= wv) && (SM >= wv))
    {
        P[0] = FL[0] + SL * (rzL - r1) - wv * UZL[0];
        P[4] = FL[4] + SL * (ezL - e1) - wv * UZL[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
        }
    }
    else if ((SM <= wv) && (SR >= wv))
    {
        P[0] = FR[0] + SR * (rzR - r2) - wv * UZR[0];
        P[4] = FR[4] + SR * (ezR - e2) - wv * UZR[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];
        }
    }
    else if (SR < wv)
    {
        P[0] = FR[0] - wv * UR[0];
        P[4] = FR[4] - wv * UR[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = FR[ik] + -wv * UR[ik];
        }
    }
    else
    {
        printf("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD\n");
    }


    P[1] = aco[0][0] * qv[0] + aco[0][1] * qv[1] + aco[0][2] * qv[2];
    P[2] = aco[1][0] * qv[0] + aco[1][1] * qv[1] + aco[1][2] * qv[2];
    P[3] = aco[2][0] * qv[0] + aco[2][1] * qv[1] + aco[2][2] * qv[2];

    return time;
}

__device__ double HLLD_Solver_Alexashov(double& ro_L, double& p_L, double& v1_L, double& v2_L, double& v3_L,//
    double& Bx_L, double& By_L, double& Bz_L,  double& ro_R, double& p_R, double& v1_R, double& v2_R, double& v3_R,//
    double& Bx_R, double& By_R, double& Bz_R, double* P, double& n1, double& n2, double& n3, double& rad)
{
    //int id_bn = 1;
    int n_state = 1;    // 1 - HLL,  3 - HLLD
    double FR[8], FL[8]; 
    double FW[8], UL[8], UZ[8], UR[8];
    double UZL[8], UZR[8];
    double UZZL[8], UZZR[8];

    double vL[3], vR[3], bL[3], bR[3];
    double vzL[3], vzR[3], bzL[3], bzR[3];
    double vzzL[3], vzzR[3], bzzL[3], bzzR[3];
    double qv[3], qb[3];
    double aco[3][3];
    double n[3];
    n[0] = n1;
    n[1] = n2;
    n[2] = n3;

    double wv = 0.0;
    double r1 = ro_L;
    double u1 = v1_L;
    double v1 = v2_L;
    double w1 = v3_L;
    double p1 = p_L;
    double bx1 = Bx_L / spi4;
    double by1 = By_L / spi4;
    double bz1 = Bz_L / spi4;


    double r2 = ro_R;
    double u2 = v1_R;
    double v2 = v2_R;
    double w2 = v3_R;
    double p2 = p_R;
    double bx2 = Bx_R / spi4;
    double by2 = By_R / spi4;
    double bz2 = Bz_R / spi4;

    double ro = (r2 + r1) / 2.0;
    //double au = (u2 + u1) / 2.0;
    //double av = (v2 + v1) / 2.0;
    //double aw = (w2 + w1) / 2.0;
    double ap = (p2 + p1) / 2.0;
    double abx = (bx2 + bx1) / 2.0;
    double aby = (by2 + by1) / 2.0;
    double abz = (bz2 + bz1) / 2.0;


    double bk = abx * n1 + aby * n2 + abz * n3;
    double b2 = kv(abx) + kv(aby) + kv(abz);

    double d = b2 - kv(bk);
    aco[0][0] = n1;
    aco[1][0] = n2;
    aco[2][0] = n3;
    if (d > eps)
    {
        d = __dsqrt_rn(d);
        aco[0][1] = (abx - bk * n[0]) / d;
        aco[1][1] = (aby - bk * n[1]) / d;
        aco[2][1] = (abz - bk * n[2]) / d;
        aco[0][2] = (aby * n[2] - abz * n[1]) / d;
        aco[1][2] = (abz * n[0] - abx * n[2]) / d;
        aco[2][2] = (abx * n[1] - aby * n[0]) / d;


    }
    else
    {
        double aix, aiy, aiz;
        if ((fabs(n[0]) < fabs(n[1])) && (fabs(n[0]) < fabs(n[2])))
        {
            aix = 1.0;
            aiy = 0.0;
            aiz = 0.0;
        }
        else if (fabs(n[1]) < fabs(n[2]))
        {
            aix = 0.0;
            aiy = 1.0;
            aiz = 0.0;
        }
        else
        {
            aix = 0.0;
            aiy = 0.0;
            aiz = 1.0;
        }

        double aik = aix * n[0] + aiy * n[1] + aiz * n[2];
        d = __dsqrt_rn(1.0 - kv(aik));
        aco[0][1] = (aix - aik * n[0]) / d;
        aco[1][1] = (aiy - aik * n[1]) / d;
        aco[2][1] = (aiz - aik * n[2]) / d;
        aco[0][2] = (aiy * n[2] - aiz * n[1]) / d;
        aco[1][2] = (aiz * n[0] - aix * n[2]) / d;
        aco[2][2] = (aix * n[1] - aiy * n[0]) / d;
    }

    for (int i = 0; i < 3; i++)
    {
        vL[i] = aco[0][i] * u1 + aco[1][i] * v1 + aco[2][i] * w1;
        vR[i] = aco[0][i] * u2 + aco[1][i] * v2 + aco[2][i] * w2;
        bL[i] = aco[0][i] * bx1 + aco[1][i] * by1 + aco[2][i] * bz1;
        bR[i] = aco[0][i] * bx2 + aco[1][i] * by2 + aco[2][i] * bz2;
    }

    double aaL = bL[0] / __dsqrt_rn(r1);
    double b2L = kv(bL[0]) + kv(bL[1]) + kv(bL[2]);
    double b21 = b2L / r1;
    double cL = __dsqrt_rn(ga * p1 / r1);
    double qp = __dsqrt_rn(b21 + cL * (cL + 2.0 * aaL));
    double qm = __dsqrt_rn(b21 + cL * (cL - 2.0 * aaL));
    double cfL = (qp + qm) / 2.0;
    double ptL = p1 + b2L / 2.0;

    double aaR = bR[0] / __dsqrt_rn(r2);
    double b2R = kv(bR[0]) + kv(bR[1]) + kv(bR[2]);
    double b22 = b2R / r2;
    double cR = __dsqrt_rn(ga * p2 / r2);
    qp = __dsqrt_rn(b22 + cR * (cR + 2.0 * aaR));
    qm = __dsqrt_rn(b22 + cR * (cR - 2.0 * aaR));
    double cfR = (qp + qm) / 2.0;
    double ptR = p2 + b2R / 2.0;

    double aC = (aaL + aaR) / 2.0;
    double b2o = (b22 + b21) / 2.0;
    double cC = __dsqrt_rn(ga * ap / ro);
    qp = __dsqrt_rn(b2o + cC * (cC + 2.0 * aC));
    qm = __dsqrt_rn(b2o + cC * (cC - 2.0 * aC));
    //double cfC = (qp + qm) / 2.0;
    //double vC1 = (vL[0] + vR[0]) / 2.0;

    /*double SL = min((vL[0] - cfL), (vR[0] - cfR));
    double SR = max((vL[0] + cfL), (vR[0] + cfR));*/

    double SL = min(vL[0], vR[0]) - max(cfL, cfR);
    double SR = min(vL[0], vR[0]) + max(cfL, cfR);

    double suR = SR - vR[0];
    double suL = SL - vL[0];
    double SM = (suR * r2 * vR[0] - ptR + ptL - suL * r1 * vL[0]) / (suR * r2 - suL * r1);

    if (SR <= SL)
    {
        printf("ERROR -  231\n");
        printf("%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",//
            vL[0], vR[0], cfL, cfR, ro_L, ro_R, p_L, p_R, suR, suL);
    }

    double SM00 = SM;
    double SR00 = SR;
    double SL00 = SL;
    double SM01, SR01, SL01;
    if ((SM00 >= SR00) || (SM00 <= SL00))
    {
        SL = min((vL[0] - cfL), (vR[0] - cfR));
        SR = max((vL[0] + cfL), (vR[0] + cfR));
        suR = SR - vR[0];
        suL = SL - vL[0];
        SM = (suR * r2 * vR[0] - ptR + ptL - suL * r1 * vL[0]) / (suR * r2 - suL * r1);
        SM01 = SM;
        SR01 = SR;
        SL01 = SL;
        if ((SM01 >= SR01) || (SM01 <= SL01))
        {
            printf("ERROR -  251\n");
        }
    }

    double dsl, dsp;
    dsl = SL;
    //dsc = SM;
    dsp = SR;

    double UU = max(fabs(dsl), fabs(dsp));
    double time = krit * rad / UU;

    double upt1 = (kv(u1) + kv(v1) + kv(w1)) / 2.0;
    double sbv1 = u1 * bx1 + v1 * by1 + w1 * bz1;

    double upt2 = (kv(u2) + kv(v2) + kv(w2)) / 2.0;
    double sbv2 = u2 * bx2 + v2 * by2 + w2 * bz2;

    double e1 = p1 / g1 + r1 * upt1 + b2L / 2.0;
    double e2 = p2 / g1 + r2 * upt2 + b2R / 2.0;

    FL[0] = r1 * vL[0];
    FL[1] = r1 * vL[0] * vL[0] + ptL - kv(bL[0]);
    FL[2] = r1 * vL[0] * vL[1] - bL[0] * bL[1];
    FL[3] = r1 * vL[0] * vL[2] - bL[0] * bL[2];
    FL[4] = (e1 + ptL) * vL[0] - bL[0] * sbv1;
    FL[5] = 0.0;
    FL[6] = vL[0] * bL[1] - vL[1] * bL[0];
    FL[7] = vL[0] * bL[2] - vL[2] * bL[0];

    FR[0] = r2 * vR[0];
    FR[1] = r2 * vR[0] * vR[0] + ptR - kv(bR[0]);
    FR[2] = r2 * vR[0] * vR[1] - bR[0] * bR[1];
    FR[3] = r2 * vR[0] * vR[2] - bR[0] * bR[2];
    FR[4] = (e2 + ptR) * vR[0] - bR[0] * sbv2;
    FR[5] = 0.0;
    FR[6] = vR[0] * bR[1] - vR[1] * bR[0];
    FR[7] = vR[0] * bR[2] - vR[2] * bR[0];

    UL[0] = r1;
    UL[4] = e1;
    UR[0] = r2;
    UR[4] = e2;


    for (int ik = 0; ik < 3; ik++)
    {
        UL[ik + 1] = r1 * vL[ik];
        UL[ik + 5] = bL[ik];
        UR[ik + 1] = r2 * vR[ik];
        UR[ik + 5] = bR[ik];
    }

    for (int ik = 0; ik < 8; ik++)
    {
        UZ[ik] = (SR * UR[ik] - SL * UL[ik] + FL[ik] - FR[ik]) / (SR - SL);
    }


    // if(id_bn == 1)
    // {
    //     UZ[5] = 0.0;
    // }
    if (n_state == 0)  // HLL new
    {
        double dq[8];
        for (int ik = 0; ik < 8; ik++)
        {
            dq[ik] = UR[ik] - UL[ik];
        }

        double TL = SL;
        double TR = SR;

        if (SL > wv)
        {
            TL = 0.0;
            for (int ik = 0; ik < 8; ik++)
            {
                FW[ik] = wv * UL[ik];
            }
        }
        else if ( ( SL <= wv)&&(wv <= SR ) )
        {
            for (int ik = 0; ik < 8; ik++)
            {
                FW[ik] = wv * UZ[ik];
            }
        }
        else if (SR < wv)
        {
            TR = 0.0;
            for (int ik = 0; ik < 8; ik++)
            {
                FW[ik] = wv * UR[ik];
            }
        }
        double a = TR * TL;
        double b = TR - TL;

        P[0] = (TR * FL[0] - TL * FR[0] + a * dq[0]) / b - FW[0];
        P[4] = (TR * FL[4] - TL * FR[4] + a * dq[4]) / b - FW[4];
        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik] = (TR * FL[ik] - TL * FR[ik] + a * dq[ik]) / b - FW[ik];
        }
        for (int ik = 5; ik < 8; ik++)
        {
            qb[ik - 5] = (TR * FL[ik] - TL * FR[ik] + a * dq[ik]) / b - FW[ik];
        }
        for (int ik = 0; ik < 3; ik++)
        {
            P[ik + 1] = aco[ik][0] * qv[0] + aco[ik][1] * qv[1] + aco[ik][2] * qv[2];
            P[ik + 5] = aco[ik][0] * qb[0] + aco[ik][1] * qb[1] + aco[ik][2] * qb[2];
            P[ik + 5] = spi4 * P[ik + 5];
        }
        double SWAP = P[4];
        P[4] = P[5];
        P[5] = P[6];
        P[6] = P[7];
        P[7] = SWAP;
        return time;
    }
    else if (n_state == 1)  /// HLL
    {
        double dq[8];
        for (int ik = 0; ik < 8; ik++)
        {
            dq[ik] = UR[ik] - UL[ik];
        }

        double TL = SL;
        double TR = SR;
        if (SL > wv)
        {
            TL = 0.0;
            for (int ik = 0; ik < 8; ik++)
            {
                FW[ik] = wv * UL[ik];
            }
        }
        if ((SL <= wv) && (wv <= SR))
        {
            for (int ik = 0; ik < 8; ik++)
            {
                FW[ik] = wv * UZ[ik];
            }
        }
        if (SR < wv)
        {
            TR = 0.0;
            for (int ik = 0; ik < 8; ik++)
            {
                FW[ik] = wv * UR[ik];
            }
        }


        double a = TR * TL;
        double b = TR - TL;

        P[0] = (TR * FL[0] - TL * FR[0] + a * dq[0]) / b - FW[0];
        P[4] = (TR * FL[4] - TL * FR[4] + a * dq[4]) / b - FW[4];

        for (int ik = 1; ik < 4; ik++)
        {
            qv[ik - 1] = (TR * FL[ik] - TL * FR[ik] + a * dq[ik]) / b - FW[ik];
        }
        for (int ik = 5; ik < 8; ik++)
        {
            qb[ik - 5] = (TR * FL[ik] - TL * FR[ik] + a * dq[ik]) / b - FW[ik];
        }

        for (int i = 0; i < 3; i++)
        {
            P[i + 1] = aco[i][0] * qv[0] + aco[i][1] * qv[1] + aco[i][2] * qv[2];
            P[i + 5] = aco[i][0] * qb[0] + aco[i][1] * qb[1] + aco[i][2] * qb[2];
            P[i + 5] = spi4 * P[i + 5];
        }

        double SWAP = P[4];
        P[4] = P[5];
        P[5] = P[6];
        P[6] = P[7];
        P[7] = SWAP;

        return time;
    }
    else if (n_state == 3)     /// HLLD
    {
        double ptz = (suR * r2 * ptL - suL * r1 * ptR + r1 * r2 * suR * suL * (vR[0] - vL[0])) / (suR * r2 - suL * r1);

        vzL[0] = SM;
        vzR[0] = SM;
        vzzL[0] = SM;
        vzzR[0] = SM;
        //double ptzL = ptz;
        //double ptzR = ptz;
        //double ptzzL = ptz;
        //double ptzzR = ptz;

        double suRm = suR / (SR - SM);
        double suLm = suL / (SL - SM);
        double rzR = r2 * suRm;
        double rzL = r1 * suLm;

        double bn = UZ[5];
        double bn2 = bn * bn;
        bzL[0] = bn;
        bzR[0] = bn;
        bzzL[0] = bn;
        bzzR[0] = bn;

        double ttR = r2 * suR * (SR - SM) - bn2;
        double tvR, tbR;

        if (fabs(ttR) <= eps)
        {
            tvR = 0.0;
            tbR = 0.0;
        }
        else
        {
            tvR = (SM - vR[0]) / ttR;
            tbR = (r2 * suR * suR - bn2) / ttR;
        }

        double ttL = r1 * suL * (SL - SM) - bn2;
        double tvL, tbL;

        if (fabs(ttL) <= eps)
        {
            tvL = 0.0;
            tbL = 0.0;
        }
        else
        {
            tvL = (SM - vL[0]) / ttL;
            tbL = (r1 * suL * suL - bn2) / ttL;
        }

        vzL[1] = vL[1] - bn * bL[1] * tvL;
        vzL[2] = vL[2] - bn * bL[2] * tvL;
        vzR[1] = vR[1] - bn * bR[1] * tvR;
        vzR[2] = vR[2] - bn * bR[2] * tvR;

        bzL[1] = bL[1] * tbL;
        bzL[2] = bL[2] * tbL;
        bzR[1] = bR[1] * tbR;
        bzR[2] = bR[2] * tbR;

        double sbvL = bzL[0] * vzL[0] + bzL[1] * vzL[1] + bzL[2] * vzL[2];
        double sbvR = bzR[0] * vzR[0] + bzR[1] * vzR[1] + bzR[2] * vzR[2];

        double ezR = e2 * suRm + (ptz * SM - ptR * vR[0] + bn * (sbv2 - sbvR)) / (SR - SM);
        double ezL = e1 * suLm + (ptz * SM - ptL * vL[0] + bn * (sbv1 - sbvL)) / (SL - SM);

        double rzzR = rzR;
        double rzzL = rzL;
        double rzRs = __dsqrt_rn(rzR);
        double rzLs = __dsqrt_rn(rzL);
        double rzss = rzRs + rzLs;
        double rzps = rzRs * rzLs;

        double SZL = SM - fabs(bn) / rzLs;
        double SZR = SM + fabs(bn) / rzRs;

        int ibn = 0;
        double sbn;
        if (fabs(bn) > epsb)
        {
            sbn = fabs(bn) / bn;
            ibn = 1;
        }
        else
        {
            sbn = 0.0;
            ibn = 0;
            SZL = SM;
            SZR = SM;
        }

        vzzL[1] = (rzLs * vzL[1] + rzRs * vzR[1] + sbn * (bzR[1] - bzL[1])) / rzss;
        vzzL[2] = (rzLs * vzL[2] + rzRs * vzR[2] + sbn * (bzR[2] - bzL[2])) / rzss;
        vzzR[1] = vzzL[1];
        vzzR[2] = vzzL[2];

        bzzL[1] = (rzLs * bzR[1] + rzRs * bzL[1] + sbn * rzps * (vzR[1] - vzL[1])) / rzss;
        bzzL[2] = (rzLs * bzR[2] + rzRs * bzL[2] + sbn * rzps * (vzR[2] - vzL[2])) / rzss;
        bzzR[1] = bzzL[1];
        bzzR[2] = bzzL[2];

        double sbzz = bzzL[0] * vzzL[0] + bzzL[1] * vzzL[1] + bzzL[2] * vzzL[2];

        double ezzR = ezR + rzRs * sbn * (sbvR - sbzz);
        double ezzL = ezL - rzLs * sbn * (sbvL - sbzz);

        UZL[0] = rzL;
        UZL[4] = ezL;
        UZR[0] = rzR;
        UZR[4] = ezR;

        for (int ik = 0; ik < 3; ik++)
        {
            UZL[ik + 1] = vzL[ik] * rzL;
            UZL[ik + 5] = bzL[ik];
            UZR[ik + 1] = vzR[ik] * rzR;
            UZR[ik + 5] = bzR[ik];
        }

        UZZL[0] = rzzL;
        UZZL[4] = ezzL;
        UZZR[0] = rzzR;
        UZZR[4] = ezzR;

        for (int ik = 0; ik < 3; ik++)
        {
            UZZL[ik + 1] = vzzL[ik] * rzzL;
            UZZL[ik + 5] = bzzL[ik];
            UZZR[ik + 1] = vzzR[ik] * rzzR;
            UZZR[ik + 5] = bzzR[ik];
        }

        int j_ccs = -1;

        if (SL > wv)
        {
            P[0] = FL[0] - wv * UL[0];
            P[4] = FL[4] - wv * UL[4];
            for (int ik = 1; ik < 4; ik++)
            {
                qv[ik - 1] = FL[ik] - wv * UL[ik];
            }
            for (int ik = 5; ik < 8; ik++)
            {
                qb[ik - 5] = FL[ik] - wv * UL[ik];
            }
            j_ccs = 1;
        }

        if ((SL <= wv) && (SZL >= wv))
        {
            int ik = 0;
            P[ik] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
            ik = 4;
            P[ik] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
            for (ik = 1; ik < 4; ik++)
            {
                qv[ik - 1] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
            }
            for (ik = 5; ik < 8; ik++)
            {
                qb[ik - 5] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
            }
            j_ccs = 2;
        }

        if (ibn == 1)
        {
            if ((SZL <= wv) && (SM >= wv))
            {
                int ik = 0;
                P[ik] = FL[ik] + SZL * (UZZL[ik] - UZL[ik]) + SL * (UZL[ik] - UL[ik]) - wv * UZZL[ik];
                ik = 4;
                P[ik] = FL[ik] + SZL * (UZZL[ik] - UZL[ik]) + SL * (UZL[ik] - UL[ik]) - wv * UZZL[ik];

                for (ik = 1; ik < 4; ik++)
                {
                    qv[ik - 1] = FL[ik] + SZL * (UZZL[ik] - UZL[ik]) + SL * (UZL[ik] - UL[ik]) - wv * UZZL[ik];
                }
                for (ik = 5; ik < 8; ik++)
                {
                    qb[ik - 5] = FL[ik] + SZL * (UZZL[ik] - UZL[ik]) + SL * (UZL[ik] - UL[ik]) - wv * UZZL[ik];
                }
                j_ccs = 3;
            }

            if ((SM <= wv) && (SZR >= wv))
            {
                int ik = 1;
                P[ik] = FR[ik] + SZR * (UZZR[ik] - UZR[ik]) + SR * (UZR[ik] - UR[ik]) - wv * UZZR[ik];
                ik = 5;
                P[ik] = FR[ik] + SZR * (UZZR[ik] - UZR[ik]) + SR * (UZR[ik] - UR[ik]) - wv * UZZR[ik];

                for (ik = 1; ik < 4; ik++)
                {
                    qv[ik - 1] = FR[ik] + SZR * (UZZR[ik] - UZR[ik]) + SR * (UZR[ik] - UR[ik]) - wv * UZZR[ik];
                }
                for (ik = 5; ik < 8; ik++)
                {
                    qb[ik - 5] = FR[ik] + SZR * (UZZR[ik] - UZR[ik]) + SR * (UZR[ik] - UR[ik]) - wv * UZZR[ik];
                }
                j_ccs = 4;
            }
        }

        if ((SZR <= wv) && (SR >= wv))
        {
            int ik = 1;
            P[ik] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];
            ik = 5;
            P[ik] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];
            for (ik = 1; ik < 4; ik++)
            {
                qv[ik - 1] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];
            }
            for (ik = 5; ik < 8; ik++)
            {
                qb[ik - 5] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];
            }
            j_ccs = 5;
        }

        if (SR < wv)
        {
            P[0] = FR[0] - wv * UR[0];
            P[4] = FR[4] - wv * UR[4];
            for (int ik = 1; ik < 4; ik++)
            {
                qv[ik - 1] = FR[ik] - wv * UR[ik];
            }
            for (int ik = 5; ik < 8; ik++)
            {
                qb[ik - 5] = FR[ik] - wv * UR[ik];
            }
            j_ccs = 6;
        }

        if (j_ccs == -1)
        {
            printf("ERROR -  559\n");
            /*watch(SL);
            watch(r1);
            watch(p1);
            watch(v1);
            watch(u1);
            watch(w1);
            watch(bx1);
            watch(by1);
            watch(bz1);
            watch(r2);
            watch(p2);
            watch(v2);
            watch(u2);
            watch(w2);
            watch(bx2);
            watch(by2);
            watch(bz2);*/
        }


        double SN = -max(fabs(SL), fabs(SR)) / 2.0;

        double wbn = 0.0;
        if (wv >= SR)
        {
            wbn = wv * bR[0];
        }
        else if (wv <= SL)
        {
            wbn = wv * bL[0];
        }
        else
        {
            wbn = wv * (bL[0] + bR[0]) / 2.0;
        }

        qb[0] = SN * (bR[0] - bL[0]) - wbn;



        for (int i = 0; i < 3; i++)
        {
            P[i + 1] = aco[i][0] * qv[0] + aco[i][1] * qv[1] + aco[i][2] * qv[2];
            P[i + 5] = aco[i][0] * qb[0] + aco[i][1] * qb[1] + aco[i][2] * qb[2];
            P[i + 5] = spi4 * P[i + 5];
        }

        double SWAP = P[4];
        P[4] = P[5];
        P[5] = P[6];
        P[6] = P[7];
        P[7] = SWAP;

        return time;
    }
    return time;
}

__device__ double HLL(double& ro_L, double& p_L, double& v1_L, double& v2_L, double& v3_L,//
    double& ro_R, double& p_R, double& v1_R, double& v2_R, double& v3_R,//
    double* P, double& n1, double& n2, double& n3, double& rad)
{

    double t = 99999999999.0;

    double e_L, e_R;
    double Vkv_L, Vkv_R;
    double c_L, c_R;

    Vkv_L = v1_L * v1_L + v2_L * v2_L + v3_L * v3_L;
    Vkv_R = v1_R * v1_R + v2_R * v2_R + v3_R * v3_R;
    if (ro_L <= 0)
    {
        c_L = 0.0;
    }
    else
    {
        c_L = __dsqrt_rn(ggg * p_L / ro_L);
    }

    if (ro_R <= 0)
    {
        c_R = 0.0;
    }
    else
    {
        c_R = __dsqrt_rn(ggg * p_R / ro_R);
    }
    e_L = p_L / (ggg - 1.0) + ro_L * Vkv_L / 2.0;  /// ѕолна€ энерги€ слева
    e_R = p_R / (ggg - 1.0) + ro_R * Vkv_R / 2.0;  /// ѕолна€ энерги€ справа

    double Vn_L = v1_L * n1 + v2_L * n2 + v3_L * n3;
    double Vn_R = v1_R * n1 + v2_R * n2 + v3_R * n3;
    double D_L = min(Vn_L, Vn_R) - max(c_L, c_R);
    double D_R = max(Vn_L, Vn_R) + max(c_L, c_R);
    t = min(t, krit * rad / max(fabs(D_L), fabs(D_R)));

    double fx1 = ro_L * v1_L;
    double fx2 = ro_L * v1_L * v1_L + p_L;
    double fx3 = ro_L * v1_L * v2_L;
    double fx4 = ro_L * v1_L * v3_L;
    double fx5 = (e_L + p_L) * v1_L;

    double fy1 = ro_L * v2_L;
    double fy2 = ro_L * v1_L * v2_L;
    double fy3 = ro_L * v2_L * v2_L + p_L;
    double fy4 = ro_L * v2_L * v3_L;
    double fy5 = (e_L + p_L) * v2_L;

    double fz1 = ro_L * v3_L;
    double fz2 = ro_L * v1_L * v3_L;
    double fz3 = ro_L * v2_L * v3_L;
    double fz4 = ro_L * v3_L * v3_L + p_L;
    double fz5 = (e_L + p_L) * v3_L;

    double fl_1 = fx1 * n1 + fy1 * n2 + fz1 * n3;
    double fl_2 = fx2 * n1 + fy2 * n2 + fz2 * n3;
    double fl_3 = fx3 * n1 + fy3 * n2 + fz3 * n3;
    double fl_4 = fx4 * n1 + fy4 * n2 + fz4 * n3;
    double fl_5 = fx5 * n1 + fy5 * n2 + fz5 * n3;

    /*double U_L1 = ro_L;
    double U_L2 = ro_L * v1_L;
    double U_L3 = ro_L * v2_L;
    double U_L4 = ro_L * v3_L;
    double U_L5 = e_L;*/

    if (D_L > 0.0)
    {
        P[0] = fl_1; /// Ќужно будет домножить на площадь грани и шаг по времени
        P[1] = fl_2;
        P[2] = fl_3;
        P[3] = fl_4;
        P[4] = fl_5;
        return t;
    }
    else
    {
        double hx1 = ro_R * v1_R;
        double hx2 = ro_R * v1_R * v1_R + p_R;
        double hx3 = ro_R * v1_R * v2_R;
        double hx4 = ro_R * v1_R * v3_R;
        double hx5 = (e_R + p_R) * v1_R;

        double hy1 = ro_R * v2_R;
        double hy2 = ro_R * v1_R * v2_R;
        double hy3 = ro_R * v2_R * v2_R + p_R;
        double hy4 = ro_R * v2_R * v3_R;
        double hy5 = (e_R + p_R) * v2_R;

        double hz1 = ro_R * v3_R;
        double hz2 = ro_R * v1_R * v3_R;
        double hz3 = ro_R * v2_R * v3_R;
        double hz4 = ro_R * v3_R * v3_R + p_R;
        double hz5 = (e_R + p_R) * v3_R;

        double fr_1 = hx1 * n1 + hy1 * n2 + hz1 * n3;
        double fr_2 = hx2 * n1 + hy2 * n2 + hz2 * n3;
        double fr_3 = hx3 * n1 + hy3 * n2 + hz3 * n3;
        double fr_4 = hx4 * n1 + hy4 * n2 + hz4 * n3;
        double fr_5 = hx5 * n1 + hy5 * n2 + hz5 * n3;

        /*double U_R1 = ro_R;
        double U_R2 = ro_R * v1_R;
        double U_R3 = ro_R * v2_R;
        double U_R4 = ro_R * v3_R;
        double U_R5 = e_R;*/

        if (D_R < 0.0)
        {
            P[0] = fr_1; /// Ќужно будет домножить на площадь грани и шаг по времени
            P[1] = fr_2;
            P[2] = fr_3;
            P[3] = fr_4;
            P[4] = fr_5;
            return t;
        }
        else
        {
            double dq1 = ro_R - ro_L;
            double dq2 = ro_R * v1_R - ro_L * v1_L;
            double dq3 = ro_R * v2_R - ro_L * v2_L;
            double dq4 = ro_R * v3_R - ro_L * v3_L;
            double dq5 = e_R - e_L;

            /*double U1 = (D_R * U_R1 - D_L * U_L1 - fr_1 + fl_1) / (D_R - D_L);
            double U2 = (D_R * U_R2 - D_L * U_L2 - fr_2 + fl_2) / (D_R - D_L);
            double U3 = (D_R * U_R3 - D_L * U_L3 - fr_3 + fl_3) / (D_R - D_L);
            double U4 = (D_R * U_R4 - D_L * U_L4 - fr_4 + fl_4) / (D_R - D_L);
            double U5 = (D_R * U_R5 - D_L * U_L5 - fr_5 + fl_5) / (D_R - D_L);*/


            P[0] = (D_R * fl_1 - D_L * fr_1 + D_L * D_R * dq1) / (D_R - D_L); /// Ќужно будет домножить на площадь грани и шаг по времени
            P[1] = (D_R * fl_2 - D_L * fr_2 + D_L * D_R * dq2) / (D_R - D_L);
            P[2] = (D_R * fl_3 - D_L * fr_3 + D_L * D_R * dq3) / (D_R - D_L);
            P[3] = (D_R * fl_4 - D_L * fr_4 + D_L * D_R * dq4) / (D_R - D_L);
            P[4] = (D_R * fl_5 - D_L * fr_5 + D_L * D_R * dq5) / (D_R - D_L);
            return t;
        }
    }
}

__device__ double get_square(const double& x1, const double& y1, const double& z1,//
    const double& dx1, const double& dy1, const double& dz1, const double& x2, const double& y2, const double& z2,//
    const double& dx2, const double& dy2, const double& dz2, double& n1, double& n2, double& n3, double& dist)
{
    if (fabs(fabs(x1 - x2) - dx1 - dx2) < 0.0004)
    {
        n1 = (x2 - x1) / fabs(x1 - x2);
        n2 = 0.0;
        n3 = 0.0;
        dist = min(dx1, dx2);
        return 4.0 * min(dy1, dy2) * min(dz1, dz2);
    }
    else if (fabs(fabs(y1 - y2) - dy1 - dy2) < 0.0004)
    {
        n2 = (y2 - y1) / fabs(y1 - y2);
        n1 = 0.0;
        n3 = 0.0;
        dist = min(dy1, dy2);
        return 4.0 * min(dx1, dx2) * min(dz1, dz2);
    }
    else if (fabs(fabs(z1 - z2) - dz1 - dz2) < 0.0004)
    {
        n3 = (z2 - z1) / fabs(z1 - z2);
        n2 = 0.0;
        n1 = 0.0;
        dist = min(dz1, dz2);
        return 4.0 * min(dy1, dy2) * min(dx1, dx2);
    }
    else
    {
        printf("Error:  get_square: %lf, %lf, %lf, %lf, %lf,  %lf,  %lf,  %lf,  %lf, %lf,  %lf,  %lf\n", //
            x1, y1, z1, x2, y2, z2, dx1, dy1, dz1, dx2, dy2, dz2);
    }
    return 0.0;
}

__global__ void funk_time(double* T, double* T_do, double* TT, int* i)
{
    *T_do = *T;
    *TT = *TT + *T_do;
    *T = 10000000;
    *i = *i + 1;
    if (*i % 10000 == 0)
    {
        printf("i = %d,  TT = %lf years,  dT = %lf hours \n", *i, *TT/0.00791429, *T_do / 9.03458E-7);
    }
    return;
}

__global__ void perekluch(double* X, double* Y, double* Z, double* RO1, double* P1, double* U1, double* V1,//
    double* W1, double* TT)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; // √лобальный индекс текущего потока
    if (index == 0)
    {
        *TT = 0.0;
    }
    double x = X[index];
    double y = Y[index];
    double z = Z[index];
    double dist = sqrt( kv(x) + kv(y) + kv(z) );
    if (dist < 0.05)
    {
        RO1[index] = 0.0;
        P1[index] = 0.0;
        U1[index] = 0.0;
        V1[index] = 0.0;
        W1[index] = 0.0;
    }
    else if (dist <= 0.1)
    {
        RO1[index] = RO1[index]/100.0;
        U1[index] = U1[index] * 100.0;
        V1[index] = V1[index] * 100.0;
        W1[index] = W1[index] * 100.0;
    }
}

__global__ void Cuda_main_HLL(double* X, double* Y, double* Z, double* DX, double* DY, double* DZ,//
    double* RO1, double* RO2, double*P1, double*P2, double*U1, double*U2, double*V1, double*V2,//
    double* W1, double*W2, int* SOSED, int*L, int*R, double* T, double* T_do)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; // √лобальный индекс текущего потока
    double x, y, z, dx, dy, dz, ro, p, u, v, w;
    int l = L[index];
    int r = R[index];
    x = X[index];
    y = Y[index];
    z = Z[index];
    dx = DX[index];
    dy = DY[index];
    dz = DZ[index];
    ro = RO1[index];
    p = P1[index];
    u = U1[index];
    v = V1[index];
    w = W1[index];

    if (__dsqrt_rn(x * x + y * y + z * z) <= 0.1)
    {
        RO2[index] = ro;
        P2[index] = p;
        U2[index] = u;
        V2[index] = v;
        W2[index] = w;
    }
    else
    {
        double n1 = 0.0;
        double n2 = 0.0;
        double n3 = 0.0;
        double dist = 0.0;
        double P[5] = { 0.0 };
        double Potok[5] = { 0.0 };
        Potok[0] = Potok[1] = Potok[2] = Potok[3] = Potok[4] = 0.0;
        double tmin = 1000;
        double Volume = dx * dy * dz * 8.0;
        int ii = 0;
        double x2, y2, z2, dx2, dy2, dz2, ro2, p2, u2, v2, w2;
        double U, V, W, UU, VV, WW;
        for (int i = l; i <= r; i++)
        {
            ii = SOSED[i];
            if (ii >= 0)
            {
                x2 = X[ii];
                y2 = Y[ii];
                z2 = Z[ii];
                dx2 = DX[ii];
                dy2 = DY[ii];
                dz2 = DZ[ii];
                ro2 = RO1[ii];
                p2 = P1[ii];
                u2 = U1[ii];
                v2 = V1[ii];
                w2 = W1[ii];
                double S = get_square(x, y, z, dx, dy, dz, x2, y2, z2, dx2, dy2, dz2, n1, n2, n3, dist);

                transfer(x, y, z, x + n1 * dx, y + n2 * dy, z + n3 * dz, u, v, w, U, V, W);
                transfer(x2, y2, z2, x2 - n1 * dx2, y2 - n2 * dy2, z2 - n3 * dz2, u2, v2, w2, UU, VV, WW);

                tmin = min(tmin, HLLC_Aleksashov(ro, p, U, V, W, ro2, p2, UU, VV, WW, P, n1, n2, n3, dist));

                for (int k = 0; k < 5; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -1)
            {
                double S = dy * dz * 4.0;
                n1 = 1.0;
                n2 = 0.0;
                n3 = 0.0;
                dist = dx;
                tmin = min(tmin, HLLC_Aleksashov(ro, p, u, v, w, ro, p, u, v, w, P, n1, n2, n3, dist));
                for (int k = 0; k < 5; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -2)
            {
                double S = dy * dz * 4.0;
                n1 = -1.0;
                n2 = 0.0;
                n3 = 0.0;
                dist = dx;
                tmin = min(tmin, HLLC_Aleksashov(ro, p, u, v, w, ro, p, u, v, w, P, n1, n2, n3, dist));
                for (int k = 0; k < 5; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -3)
            {
                double S = dx * dz * 4.0;
                n1 = 0.0;
                n2 = 1.0;
                n3 = 0.0;
                dist = dy;
                tmin = min(tmin, HLLC_Aleksashov(ro, p, u, v, w, ro, p, u, v, w, P, n1, n2, n3, dist));
                for (int k = 0; k < 5; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -4)
            {
                double S = dx * dz * 4.0;
                n1 = 0.0;
                n2 = -1.0;
                n3 = 0.0;
                dist = dy;
                tmin = min(tmin, HLLC_Aleksashov(ro, p, u, v, w, ro, p, u, v, w, P, n1, n2, n3, dist));
                for (int k = 0; k < 5; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -5)
            {
                double S = dy * dx * 4.0;
                n1 = 0.0;
                n2 = 0.0;
                n3 = 1.0;
                dist = dz;
                tmin = min(tmin, HLLC_Aleksashov(ro, p, u, v, w, ro, p, u, v, w, P, n1, n2, n3, dist));
                for (int k = 0; k < 5; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -6)
            {
                double S = dy * dx * 4.0;
                n1 = 0.0;
                n2 = 0.0;
                n3 = -1.0;
                dist = dz;
                tmin = min(tmin, HLLC_Aleksashov(ro, p, u, v, w, ro, p, u, v, w, P, n1, n2, n3, dist));
                for (int k = 0; k < 5; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
        }

        double ro3, p3, u3, v3, w3;

        ro3 = ro - *T_do * Potok[0] / Volume;
        if (ro3 <= 0)
        {
            printf("ERROR -  errhtryr436kokjyhtd\n");
            ro3 = 0.0001;
        }
        u3 = (ro * u - *T_do * Potok[1] / Volume) / ro3;
        v3 = (ro * v - *T_do * Potok[2] / Volume) / ro3;
        w3 = (ro * w - *T_do * Potok[3] / Volume) / ro3;
        p3 = ((p / (ggg - 1.0) + 0.5 * ro * (u * u + v * v + w * w)) //
            - *T_do * (Potok[4] / Volume + ro * ro * Lya(p / ro * 0.8)) - 0.5 * ro3 * (u3 * u3 + v3 * v3 + w3 * w3)) * (ggg - 1.0);

        if (p3 <= 0)
        {
           p3 = 0.000001;
        }

        RO2[index] = ro3;
        P2[index] = p3;
        U2[index] = u3;
        V2[index] = v3;
        W2[index] = w3;

        if (*T > tmin)
        {
            *T = tmin;
            __threadfence();
        }
    }

}

__global__ void Cuda_main_HLLD(int* NN, double* X, double* Y, double* Z, double* DX, double* DY, double* DZ,//
    double* RO1, double* RO2, double* P1, double* P2, double* U1, double* U2, double* V1, double* V2,//
    double* W1, double* W2, double* BX1, double* BY1, double* BZ1, double* BX2, double* BY2, double* BZ2,//
    int* SOSED, int* L, int* R, double* T, double* T_do, bool mgd = true, bool diver = true, int metod = 0)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; // √лобальный индекс текущего потока
    if (index > *NN - 1)
    {
        return;
    }
    double x, y, z, dx, dy, dz, ro, p, u, v, w, bx, by, bz;
    int l = L[index];
    int r = R[index];
    x = X[index];
    y = Y[index];
    z = Z[index];
    dx = DX[index];
    dy = DY[index];
    dz = DZ[index];
    ro = RO1[index];
    p = P1[index];
    u = U1[index];
    v = V1[index];
    w = W1[index];
    if (mgd == true)
    {
        bx = BX1[index];
        by = BY1[index];
        bz = BZ1[index];
    }
    else
    {
        bx = 0.0;
        by = 0.0;
        bz = 0.0;
    }




    double dist3 = __dsqrt_rn(kv(x + 155.0) + kv(y) + z * z);

    if ( (x * x + y * y + z * z) <= 13924.0 || dist3 < 69.0)
    {
        RO2[index] = ro;
        P2[index] = p;
        U2[index] = u;
        V2[index] = v;
        W2[index] = w;
        BX2[index] = bx;
        BY2[index] = by;
        BZ2[index] = bz;
    }
    else
    {
        double n1 = 0.0;
        double n2 = 0.0;
        double n3 = 0.0;
        double dist = 0.0;
        double P[8] = { 0.0 };
        P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;
        double Potok[9] = { 0.0 };
        Potok[0] = Potok[1] = Potok[2] = Potok[3] = Potok[4] = Potok[5] = Potok[6] = Potok[7] = Potok[8] =  0.0;
        double tmin = 1000;
        double Volume = dx * dy * dz * 8.0;
        int ii = 0;
        double x2, y2, z2, dx2, dy2, dz2, ro2, p2, u2, v2, w2, bx2, by2, bz2, sks;
        double roC = 1.0;
        double pC = 1.0 / (ggg * M_inf * M_inf);
        double uC = -1.0;
        double vC = 0.0;
        double wC = 0.0;
        double bxC, byC, bzC;
        if (mgd == true)
        {
            bxC = -spi4 * (1.0 / (M_alf)) * cos(alpha * pi / 180.0);
            byC = -spi4 * (1.0 / (M_alf)) * sin(alpha * pi / 180.0);
            bzC = 0.0;
        }
        else
        {
            bxC = 0.0;
            byC = 0.0;
            bzC = 0.0;
        }


        for (int i = l; i <= r; i++)
        {
            ii = SOSED[i];
            if (ii >= 0)
            {
                x2 = X[ii];
                y2 = Y[ii];
                z2 = Z[ii];
                dx2 = DX[ii];
                dy2 = DY[ii];
                dz2 = DZ[ii];
                /*if ( (__dsqrt_rn(kv(x - x2) + kv(y - y2) + kv(z - z2)) > max(dx2, dy2) * 2 + 0.0001)|| (__dsqrt_rn(kv(x - x2) + kv(y - y2) + kv(z - z2)) < min(dx2, dy2) * 2 - 0.0001))
                {
                    printf("Error 1736\n");
                }*/
                ro2 = RO1[ii];
                p2 = P1[ii];
                u2 = U1[ii];
                v2 = V1[ii];
                w2 = W1[ii];
                if (mgd == true)
                {
                    bx2 = BX1[ii];
                    by2 = BY1[ii];
                    bz2 = BZ1[ii];
                }
                else
                {
                    bx2 = 0.0;
                    by2 = 0.0;
                    bz2 = 0.0;
                }
                double S = get_square(x, y, z, dx, dy, dz, x2, y2, z2, dx2, dy2, dz2, n1, n2, n3, dist);
                if (diver == true)
                {
                    sks = n1 * (bx + bx2) / 2.0 + n2 * (by + by2) / 2.0 + n3 * (bz + bz2) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }
                Potok[8] = Potok[8] + sks * S;
                if (ro == 0||ro2 == 0)
                {
                    printf("ewfwddasxxesdwedwhvb   %d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", index, ii, x2, y2, z2, dx2, dy2, dz2, x, y, z, dx, dy, dz);
                }
                tmin = min(tmin, HLLD_Alexashov(ro, p, u, v, w, bx, by, bz, ro2, p2, u2, v2, w2, bx2, by2, bz2, P, n1, n2, n3, dist, metod));
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -1)
            {
                double S = dy * dz * 4.0;
                n1 = 1.0;
                n2 = 0.0;
                n3 = 0.0;
                dist = dx;
                if (diver == true)
                {
                    sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }
                Potok[8] = Potok[8] + sks * S;
                if (ro == 0)
                {
                    printf("ewfwddasxxesdwedwhvb   %d,%lf,%lf,%lf,%lf,%lf,%lf\n", ii, x, y, z, dx, dy, dz);
                }
                tmin = min(tmin, HLLD_Alexashov(ro, p, u, v, w, bx, by, bz, roC, pC, uC, vC, wC, bxC, byC, bzC, P, n1, n2, n3, dist, metod));
                //  ћожно вручную выписать потоки дл€ ускорени€ времени
                /*double b2R = kv(bxC) + kv(byC) + kv(bzC);
                double ptR = pC + b2R / 2.0;
                double upt2 = (kv(uC) + kv(vC) + kv(wC)) / 2.0;
                double sbv2 = uC * bxC + vC * byC + wC * bzC;
                double e2 = pC / g1 + roC * upt2 + b2R / 2.0;

                P[0] = roC * uC;
                P[1] = roC * uC * uC + ptR - kv(bxC);
                P[2] = roC * uC * vC - bxC * byC;
                P[3] = roC * uC * wC - bxC * bzC;
                P[7] = (e2 + ptR) * uC - bxC * sbv2;
                P[4] = 0.0;
                P[5] = uC * byC - vC * bxC;
                P[6] = uC * bzC - wC * bxC;*/

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -2)
            {
                double S = dy * dz * 4.0;
                n1 = -1.0;
                n2 = 0.0;
                n3 = 0.0;
                dist = dx;
                if (diver == true)
                {
                    sks = n1 * bx + n2 * by + n3 * bz;
                }
                else
                {
                    sks = 0.0;
                }
                Potok[8] = Potok[8] + sks * S;
                double uu = u;
                if ( (uu >= -0.1)&&(__dsqrt_rn(y*y + z * z) < 150.0) )
                {
                    uu = -1.0;
                }
                /*if ((bx > 0.0))
                {
                    bx = -0.1;
                }*/
                tmin = min(tmin, HLLD_Alexashov(ro, p, u, v, w, bx, by, bz, ro, p, uu, v, w, bx, by, bz, P, n1, n2, n3, dist, metod));
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -3)
            {
                double S = dx * dz * 4.0;
                n1 = 0.0;
                n2 = 1.0;
                n3 = 0.0;
                dist = dy;
                if (diver == true)
                {
                    sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }
                Potok[8] = Potok[8] + sks * S;
                tmin = min(tmin, HLLD_Alexashov(ro, p, u, v, w, bx, by, bz, roC, pC, uC, vC, wC, bxC, byC, bzC, P, n1, n2, n3, dist, metod));
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -4)
            {
                double S = dx * dz * 4.0;
                n1 = 0.0;
                n2 = -1.0;
                n3 = 0.0;
                dist = dy;
                if (diver == true)
                {
                    sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }
                
                Potok[8] = Potok[8] + sks * S;
                tmin = min(tmin, HLLD_Alexashov(ro, p, u, v, w, bx, by, bz, roC, pC, uC, vC, wC, bxC, byC, bzC, P, n1, n2, n3, dist, metod));
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -5)
            {
                double S = dy * dx * 4.0;
                n1 = 0.0;
                n2 = 0.0;
                n3 = 1.0;
                dist = dz;
                if (diver == true)
                {
                    sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }
                
                Potok[8] = Potok[8] + sks * S;
                tmin = min(tmin, HLLD_Alexashov(ro, p, u, v, w, bx, by, bz, roC, pC, uC, vC, wC, bxC, byC, bzC, P, n1, n2, n3, dist, metod));
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -6)
            {
                double S = dy * dx * 4.0;
                n1 = 0.0;
                n2 = 0.0;
                n3 = -1.0;
                dist = dz;
                if (diver == true)
                {
                    sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }
                
                Potok[8] = Potok[8] + sks * S;
                tmin = min(tmin, HLLD_Alexashov(ro, p, u, v, w, bx, by, bz, roC, pC, uC, vC, wC, bxC, byC, bzC, P, n1, n2, n3, dist, metod));
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else
            {
                printf("Error 12438jdyu. Ne doljni suda popadat = %d \n", ii);
            }
        }

        double ro3, p3, u3, v3, w3, bx3, by3, bz3;

        ro3 = ro - *T_do * Potok[0] / Volume;
        if (ro3 <= 0.0)
        {
            printf("ERROR -  dssdbfhfshjskfutytqqazz\n");
            printf("%lf, %lf, %lf, %lf\n", x, y, z, ro3);
            ro3 = 0.0001;
        }
        u3 = ( ro * u - *T_do * (Potok[1] + (bx / cpi4) * Potok[8]) / Volume ) / ro3;
        v3 = (ro * v - *T_do * (Potok[2] + (by / cpi4) * Potok[8]) / Volume) / ro3;
        w3 = (ro * w - *T_do * (Potok[3] + (bz / cpi4) * Potok[8]) / Volume) / ro3;
        bx3 = (bx - *T_do * (Potok[4] + u * Potok[8]) / Volume);
        by3 = (by - *T_do * (Potok[5] + v * Potok[8]) / Volume);
        bz3 = (bz - *T_do * (Potok[6] + w * Potok[8]) / Volume);
        p3 = ((U8(ro, p, u, v, w, bx, by, bz) - *T_do * (Potok[7] + (skk(u, v, w, bx, by, bz)/cpi4) * Potok[8])//
            / Volume) - 0.5 * ro3 * kvv(u3, v3, w3) - kvv(bx3,by3,bz3) / cpi8) * (ggg - 1.0);
        //u3 = (ro * u - *T_do * (Potok[1] + (bx) * Potok[8]) / Volume) / ro3;
        //v3 = (ro * v - *T_do * (Potok[2] + (by) * Potok[8]) / Volume) / ro3;
        //w3 = (ro * w - *T_do * (Potok[3] + (bz) * Potok[8]) / Volume) / ro3;
        //bx3 = (bx - *T_do * (Potok[4] + u * Potok[8]) / Volume);
        //by3 = (by - *T_do * (Potok[5] + v * Potok[8]) / Volume);
        //bz3 = (bz - *T_do * (Potok[6] + w * Potok[8]) / Volume);
        //p3 = ((U8(ro, p, u, v, w, bx, by, bz) - *T_do * (Potok[7] + (skk(u, v, w, bx, by, bz)) * Potok[8])//
        //    / Volume) - 0.5 * ro3 * kvv(u3, v3, w3) - kvv(bx3, by3, bz3)) * (ggg - 1.0);
        if (p3 <= 0)
        {
            p3 = 0.000001;
        }

        RO2[index] = ro3;
        P2[index] = p3;
        U2[index] = u3;
        V2[index] = v3;
        W2[index] = w3;
        BX2[index] = bx3;
        BY2[index] = by3;
        BZ2[index] = bz3;

        if (*T > tmin)
        {
            *T = tmin;
            __threadfence();
        }
    }

}


__global__ void Cuda_main_HLLDQ(int* NN, double* X, double* Y, double* Z, double* DX, double* DY, double* DZ,//
    double* RO1, double* RO2, double* Q1, double* Q2, double* P1, double* P2, double* U1, double* U2, double* V1, double* V2,//
    double* W1, double* W2, double* BX1, double* BY1, double* BZ1, double* BX2, double* BY2, double* BZ2,//
    int* SOSED, int* L, int* R, double* T, double* T_do, int step_, double M_inf_, bool mgd = true, bool diver = true, int metod = 0, bool istoch = false)
{
    // istoch - источники дл€ атомов (видимо делал расчЄты дл€ ќфер и добавил
    int index = blockIdx.x * blockDim.x + threadIdx.x; // √лобальный индекс текущего потока
    if (index > * NN - 1)
    {
        return;
    }
    double x, y, z, dx, dy, dz, ro, p, u, v, w, bx, by, bz, Q;
    int l = L[index];
    int r = R[index];
    x = X[index];
    y = Y[index];
    z = Z[index];
    dx = DX[index];
    dy = DY[index];
    dz = DZ[index];
    ro = RO1[index];
    p = P1[index];
    u = U1[index];
    v = V1[index];
    w = W1[index];
    Q = Q1[index];
    if (mgd == true)
    {
        bx = BX1[index];
        by = BY1[index];
        bz = BZ1[index];
    }
    else
    {
        bx = 0.0;
        by = 0.0;
        bz = 0.0;
    }



    //double ddd = kv(y) + kv(z);
    double ddd2 = kv(x) + kv(y) + kv(z);
    //double dist3 = kv(x + 1.0) / kv(1.6) + kv(y) / kv(1.6) + kv(z) / kv(1.6);
    //double dist3 = kv(x + 1.08) / kv(2.4) + kv(y) / kv(2.0) + kv(z) / kv(2.0);
    double dist3 = kv(x + 0.15) / kv(0.35) + kv(y) / kv(0.35) + kv(z) / kv(0.35);

    if (ddd2 < (ddist * ddist)) // (dist3 < 1.0001)//( ddd2 <= (ddist*ddist))
    {
        RO2[index] = ro;
        P2[index] = p;
        U2[index] = u;
        V2[index] = v;
        W2[index] = w;
        BX2[index] = bx;
        BY2[index] = by;
        BZ2[index] = bz;
        Q2[index] = Q;
    }
    else
    {
        //metod = 3;
        //if (ddd2 <= 0.73 * 0.73)
        //{
        //    metod = 2;
        //}
        

        double PQ = 0.0;
        double n1 = 0.0;
        double n2 = 0.0;
        double n3 = 0.0;
        double dist = 0.0;
        double P[8] = { 0.0 };
        P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;
        double Potok[10] = { 0.0 };
        Potok[0] = Potok[1] = Potok[2] = Potok[3] = Potok[4] = Potok[5] = Potok[6] = Potok[7] = Potok[8] = Potok[9] = 0.0;
        double tmin = 1000;
        double Volume = dx * dy * dz * 8.0;
        int ii = 0;
        double x2, y2, z2, dx2, dy2, dz2, ro2, p2, u2, v2, w2, bx2, by2, bz2, sks, Q_2;
        double su1, sv1, sw1, su2, sv2, sw2, sro1, sro2, sp1, sp2;
        double ur, up, uz;
        double roC = 1.0; 
        double rosred = 0.0; 
        double pC = 1.0 / (ggg); // 1.0 / (ggg * M_inf * M_inf);
        double uC = M_infty; // -1.0;
        double vC = 0.0;
        double wC = 0.0;
        double QC = 100.0;
        double bxC, byC, bzC;
        bxC = 0.0; // -spi4 * (1.0 / (M_alf)) * cos(alpha * pi / 180.0);
        byC = 0.0; // -spi4 * (1.0 / (M_alf)) * sin(alpha * pi / 180.0);
        bzC = B_inf;


        for (int i = l; i <= r; i++)
        {
            ii = SOSED[i];
            if (ii >= 0)
            {
                x2 = X[ii];
                y2 = Y[ii];
                z2 = Z[ii];
                dx2 = DX[ii];
                dy2 = DY[ii];
                dz2 = DZ[ii];
                ro2 = RO1[ii];
                rosred = rosred + ro2;
                p2 = P1[ii];
                u2 = U1[ii];
                v2 = V1[ii];
                w2 = W1[ii];
                Q_2 = Q1[ii];
                if (mgd == true)
                {
                    bx2 = BX1[ii];
                    by2 = BY1[ii];
                    bz2 = BZ1[ii];
                }
                else
                {
                    bx2 = 0.0;
                    by2 = 0.0;
                    bz2 = 0.0;
                }
                double S = get_square(x, y, z, dx, dy, dz, x2, y2, z2, dx2, dy2, dz2, n1, n2, n3, dist);
                if (diver == true)
                {
                    sks = n1 * (bx + bx2) / 2.0 + n2 * (by + by2) / 2.0 + n3 * (bz + bz2) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }
                Potok[8] = Potok[8] + sks * S;

                su1 = u;
                sv1 = v;
                sw1 = w;

                su2 = u2;
                sv2 = v2;
                sw2 = w2;

                sro1 = ro;
                sro2 = ro2;

                sp1 = p;
                sp2 = p2;

                // ƒелаем перенос в сферической —  
                ddd2 = kv((z + z2) / 2.0) + kv((x + x2) / 2.0) + kv((y + y2) / 2.0);
                if (ddd2 <= (ddist2 * ddist2))
                //if (kvv(u, v, w)/(ggg * p/ro) > 20.0  &&  kvv(u2, v2, w2) / (ggg * p2 / ro2) > 20.0)
                {
                    //metod = 1;

                    spherical_skorost(z, x, y, w, u, v, ur, up, uz);
                    dekard_skorost((z + z2) / 2.0, (x + x2) / 2.0, (y + y2) / 2.0, ur, up, uz, sw1, su1, sv1);

                    spherical_skorost(z2, x2, y2, w2, u2, v2, ur, up, uz);
                    dekard_skorost((z + z2) / 2.0, (x + x2) / 2.0, (y + y2) / 2.0, ur, up, uz, sw2, su2, sv2);

                    sro1 = ro * (kv(z) + kv(x) + kv(y)) / ddd2;
                    sro2 = ro2 * (kv(z2) + kv(x2) + kv(y2)) / ddd2;

                    sp1 = p * pow((kv(z) + kv(x) + kv(y)) / ddd2, ggg);
                    sp2 = p2 * pow((kv(z2) + kv(x2) + kv(y2)) / ddd2, ggg);
                }

                if (metod <= 1|| metod == 3)//(y * y + z * z < 225 && y2 * y2 + z2 * z2 < 225 && x > -15 && x2 > -15 && x < 8 && x2 < 8  && step_ > 10000)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(sro1, Q, sp1, su1, sv1, sw1, bx, by, bz, sro2, Q_2, sp2, su2, sv2, sw2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(sro1, Q, sp1, su1, sv1, sw1, bx, by, bz, sro2, Q_2, sp2, su2, sv2, sw2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, metod));
                }
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -1)
            {
                double S = dy * dz * 4.0;
                n1 = 1.0;
                n2 = 0.0;
                n3 = 0.0;
                dist = dx;
                if (diver == true)
                {
                    sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }
                /*double uu = u;
                if (uu < 0.0)
                {
                    uu = 0.0;
                }*/
                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || metod <= 1 || metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, metod));
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, metod));
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
                }
                //  ћожно вручную выписать потоки дл€ ускорени€ времени
                /*double b2R = kv(bxC) + kv(byC) + kv(bzC);
                double ptR = pC + b2R / 2.0;
                double upt2 = (kv(uC) + kv(vC) + kv(wC)) / 2.0;
                double sbv2 = uC * bxC + vC * byC + wC * bzC;
                double e2 = pC / g1 + roC * upt2 + b2R / 2.0;

                P[0] = roC * uC;
                P[1] = roC * uC * uC + ptR - kv(bxC);
                P[2] = roC * uC * vC - bxC * byC;
                P[3] = roC * uC * wC - bxC * bzC;
                P[7] = (e2 + ptR) * uC - bxC * sbv2;
                P[4] = 0.0;
                P[5] = uC * byC - vC * bxC;
                P[6] = uC * bzC - wC * bxC;*/

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -2)
            {
                double S = dy * dz * 4.0;
                n1 = -1.0;
                n2 = 0.0;
                n3 = 0.0;
                dist = dx;
                if (diver == true)
                {
                    sks = n1 * bx + n2 * by + n3 * bz;
                }
                else
                {
                    sks = 0.0;
                }
                Potok[8] = Potok[8] + sks * S;
                double uu = u;
                if (uu > M_infty/3.0)
                {
                    uu = M_infty/3.0;
                }

                if (!kor_Sol || metod <= 1 || metod == 3)
                {
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, metod));
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, uu, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
                       
                }
                else
                {
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, metod));
                    
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, uu, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, uu, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
                }

                /*double t1, t2, t3, m1, m2, m3;
                double bx_L = bx / spi4;
                double by_L = by / spi4;
                double bz_L = bz / spi4;
                t1 = 0.0;
                t2 = 0.0;
                t3 = 1.0;
                m1 = 0.0;
                m2 = 1.0;
                m3 = 0.0;
                double u1 = uu * n1 + v * n2 + w * n3;
                double v1 = uu * t1 + v * t2 + w * t3;
                double w1 = uu * m1 + v * m2 + w * m3;
                double bn1, bt1, bm1;
                bn1 = bx_L * n1 + by_L * n2 + bz_L * n3;
                bt1 = bx_L * t1 + by_L * t2 + bz_L * t3;
                bm1 = bx_L * m1 + by_L * m2 + bz_L * m3;
                double uu_L = (kv(uu) + kv(v) + kv(w)) / 2.0;
                double bb_L = kv(bx_L) + kv(by_L) + kv(bz_L);
                double e1 = p / g1 + ro * uu_L + bb_L / 2.0;
                double pTL = p + bb_L / 2.0;

                double PO[9];

                PO[0] = ro * u1;
                PO[1] = ro * u1 * u1 + pTL - kv(bn1);
                PO[2] = ro * u1 * v1 - bn1 * bt1;
                PO[3] = ro * u1 * w1 - bn1 * bm1;
                PO[4] = (e1 + pTL) * u1 - bn1 * (u1 * bn1 + v1 * bt1 + w1 * bm1);
                PO[5] = 0.0;
                PO[6] = u1 * bt1 - v1 * bn1;
                PO[7] = u1 * bm1 - w1 * bn1;
                PO[8] = Q * u1;


                P[1] = n1 * PO[1] + t1 * PO[2] + m1 * PO[3];
                P[2] = n2 * PO[1] + t2 * PO[2] + m2 * PO[3];
                P[3] = n3 * PO[1] + t3 * PO[2] + m3 * PO[3];
                P[5] = spi4 * (n1 * PO[5] + t1 * PO[6] + m1 * PO[7]);
                P[6] = spi4 * (n2 * PO[5] + t2 * PO[6] + m2 * PO[7]);
                P[7] = spi4 * (n3 * PO[5] + t3 * PO[6] + m3 * PO[7]);
                P[0] = PO[0];
                P[4] = PO[4];
                PQ = PO[8];

                double SWAP = P[4];
                P[4] = P[5];
                P[5] = P[6];
                P[6] = P[7];
                P[7] = SWAP;*/

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -3)
            {
                double S = dx * dz * 4.0;
                n1 = 0.0;
                n2 = 1.0;
                n3 = 0.0;
                dist = dy;
                double uu = 0.0;// v;
                /*if (uu < 0.0)
                {
                    uu = 0.0;
                }*/

                if (diver == true)
                {
                    //sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;
                    sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }

                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || metod <= 1 || metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, metod));
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, metod));
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
                }
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -4)
            {
                double S = dx * dz * 4.0;
                n1 = 0.0;
                n2 = -1.0;
                n3 = 0.0;
                dist = dy;
                if (diver == true)
                {
                    sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
                    //sks =  n2 * (by + by) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }

                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || metod <= 1 || metod == 3)
                {
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, -v, w, -bx, by, -bz, P, PQ, n1, n2, n3, dist, metod));
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, metod)); // ѕочему тут так?
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod)); // ѕочему тут так?
                    
                     //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
                }
                else
                {
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, -v, w, -bx, by, -bz, P, PQ, n1, n2, n3, dist, metod));
                   
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, metod));
                    // была эта
                    
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
                
                }

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -5)
            {
                double S = dy * dx * 4.0;
                n1 = 0.0;
                n2 = 0.0;
                n3 = 1.0;
                dist = dz;
                if (diver == true)
                {
                    //sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;
                    sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }
                
                //double ww = w;
                //if (ww < 0.0)
                //{
                //    ww = 0.0;
                //}

                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || metod <= 1 || metod == 3)
                {
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, ww, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, metod));
                }
                else
                {
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, ww, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, metod));
                }

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -6)
            {
                double S = dy * dx * 4.0;
                n1 = 0.0;
                n2 = 0.0;
                n3 = -1.0;
                dist = dz;
                if (diver == true)
                {
                    sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;
                    //sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }

                //double ww = 

                su1 = u;
                sv1 = v;
                sw1 = w;


                // ƒелаем перенос в сферической —  
                ddd2 = kv(0.0) + kv((x)) + kv((y));
                if (false)//(ddd2 <= (ddist2 * ddist2))
                {
                    spherical_skorost(z, x, y, w, u, v, ur, up, uz);
                    dekard_skorost(0.0, x, y, ur, up, uz, sw1, su1, sv1);
                }
                

                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || metod <= 1 || metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, -w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod)); // —имметри€
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, metod));
                }
                else
                {
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, metod));
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, -w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
                }
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else
            {
                printf("Error 12438jdyu. Ne doljni suda popadat = %d \n", ii);
            }
        }

        double q2_1 = 0.0, q2_2 = 0.0, q2_3 = 0.0, q3 = 0.0;

        if (true)//(istoch == true)
        {
            double u_H4 = M_infty, v_H4 = 0.0, w_H4 = 0.0, ro_H4 = 1.0, p_H4 = 1.0 / (2.0 * ggg);

            double U_M_H4 = sqrt(kv(u - u_H4) + kv(v - v_H4) + kv(w - w_H4) + (64.0 / (9.0 * pi)) //
                * (p / ro + 2.0 * p_H4 / ro_H4));

            double U_H4 = sqrt(kv(u - u_H4) + kv(v - v_H4) + kv(w - w_H4) + (4.0 / pi) //
                * (p / ro + 2.0 * p_H4 / ro_H4));

            double sigma_H4 = kv(1.0 - a_2 * log(U_M_H4));

            double nu_H4 = ro * ro_H4 * U_M_H4 * sigma_H4;

            q2_1 = (3.0 / Kn_) * (nu_H4 * (u_H4 - u));
            q2_2 = (3.0 / Kn_) * (nu_H4 * (v_H4 - v));
            q2_3 = (3.0 / Kn_) * (nu_H4 * (w_H4 - w));


            q3 = (3.0 / Kn_) * (nu_H4 * ((kv(u_H4) + kv(v_H4) + kv(w_H4) - kv(u) - kv(v) - kv(w)) / 2.0 + //
                (U_H4 / U_M_H4) * (2.0 * p_H4 / ro_H4 - p / ro)));
        }

        double ro3, p3, u3, v3, w3, bx3, by3, bz3, Q33;

        Q33 = Q - *T_do * Potok[9] / Volume;
        ro3 = ro - *T_do * Potok[0] / Volume;
        if (ro3 <= 0.0)
        {
            printf("Rho ERROR  %lf, %lf, %lf, %lf, %lf\n", x, y, z, ro3, ro);
            ro3 = rosred/(r - l + 1);
        }
        u3 = (ro * u - *T_do * (Potok[1] + (bx / cpi4) * Potok[8] - q2_1 * Volume) / Volume) / ro3;
        v3 = (ro * v - *T_do * (Potok[2] + (by / cpi4) * Potok[8] - q2_2 * Volume) / Volume) / ro3;
        w3 = (ro * w - *T_do * (Potok[3] + (bz / cpi4) * Potok[8] - q2_3 * Volume) / Volume) / ro3;
        bx3 = (bx - *T_do * (Potok[4] + u * Potok[8]) / Volume);
        by3 = (by - *T_do * (Potok[5] + v * Potok[8]) / Volume);
        bz3 = (bz - *T_do * (Potok[6] + w * Potok[8]) / Volume);
        p3 = ((U8(ro, p, u, v, w, bx, by, bz) - *T_do * (Potok[7] + (skk(u, v, w, bx, by, bz) / cpi4) * Potok[8] - q3 * Volume)//
            / Volume) - 0.5 * ro3 * kvv(u3, v3, w3) - kvv(bx3, by3, bz3) / cpi8) * (ggg - 1.0);
        //u3 = (ro * u - *T_do * (Potok[1] + (bx) * Potok[8]) / Volume) / ro3;
        //v3 = (ro * v - *T_do * (Potok[2] + (by) * Potok[8]) / Volume) / ro3;
        //w3 = (ro * w - *T_do * (Potok[3] + (bz) * Potok[8]) / Volume) / ro3;
        //bx3 = (bx - *T_do * (Potok[4] + u * Potok[8]) / Volume);
        //by3 = (by - *T_do * (Potok[5] + v * Potok[8]) / Volume);
        //bz3 = (bz - *T_do * (Potok[6] + w * Potok[8]) / Volume);
        //p3 = ((U8(ro, p, u, v, w, bx, by, bz) - *T_do * (Potok[7] + (skk(u, v, w, bx, by, bz)) * Potok[8])//
        //    / Volume) - 0.5 * ro3 * kvv(u3, v3, w3) - kvv(bx3, by3, bz3)) * (ggg - 1.0);
        if (p3 <= 0)
        {
            p3 = 0.000001;
        }

        Q2[index] = Q33;
        RO2[index] = ro3;
        P2[index] = p3;
        U2[index] = u3;
        V2[index] = v3;
        W2[index] = w3;
        /*if (Q33 / ro3 > 50)
        {
            BX2[index] = 0.0;
            BY2[index] = 0.0;
            BZ2[index] = 0.0;
        }
        else 
        {
            BX2[index] = bx3;
            BY2[index] = by3;
            BZ2[index] = bz3;
        }*/
        BX2[index] = bx3;
        BY2[index] = by3;
        BZ2[index] = bz3;

        if (*T > tmin)
        {
            *T = tmin;
            __threadfence();
        }
    }

}

__global__ void Cuda_main_HLLD_TVD(int* NN, double* X, double* Y, double* Z, double* DX, double* DY, double* DZ,//
    double* RO1, double* RO2, double* P1, double* P2, double* U1, double* U2, double* V1, double* V2,//
    double* W1, double* W2, double* BX1, double* BY1, double* BZ1, double* BX2, double* BY2, double* BZ2,//
    int* SOSED, int* SOSED2, int* L, int* R, double* T, double* T_do, int step_, bool mgd = true, bool diver = true, int metod = 0)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; // √лобальный индекс текущего потока
    if (index > * NN - 1)
    {
        return;
    }
    double x, y, z, dx, dy, dz, ro, p, u, v, w, bx, by, bz;
    int l = L[index];
    int r = R[index];
    x = X[index];
    y = Y[index];
    z = Z[index];
    dx = DX[index];
    dy = DY[index];
    dz = DZ[index];
    ro = RO1[index];
    p = P1[index];
    u = U1[index];
    v = V1[index];
    w = W1[index];
    if (mgd == true)
    {
        bx = BX1[index];
        by = BY1[index];
        bz = BZ1[index];
    }
    else
    {
        bx = 0.0;
        by = 0.0;
        bz = 0.0;
    }





    if ((x * x + y * y + z * z) <= 0.1 )
    {
        RO2[index] = ro;
        P2[index] = p;
        U2[index] = u;
        V2[index] = v;
        W2[index] = w;
        BX2[index] = bx;
        BY2[index] = by;
        BZ2[index] = bz;
    }
    else
    {
        double n1 = 0.0;
        double n2 = 0.0;
        double n3 = 0.0;
        double dist = 0.0;
        double P[8] = { 0.0 };
        P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;
        double Potok[9] = { 0.0 };
        Potok[0] = Potok[1] = Potok[2] = Potok[3] = Potok[4] = Potok[5] = Potok[6] = Potok[7] = Potok[8] = 0.0;
        double tmin = 1000;
        double Volume = dx * dy * dz * 8.0;
        int ii = 0;
        double x2, y2, z2, dx2, dy2, dz2, ro2, p2, u2, v2, w2, bx2, by2, bz2, sks;
        double x3, y3, z3, dx3, dy3, dz3, ro3, p3, u3, v3, w3, bx3, by3, bz3;
        double x4, y4, z4, dx4, dy4, dz4, ro4, p4, u4, v4, w4, bx4, by4, bz4;
        double x12, y12, z12, dx12, dy12, dz12, ro12, p12, u12, v12, w12, bx12, by12, bz12;
        double x21, y21, z21, dx21, dy21, dz21, ro21, p21, u21, v21, w21, bx21, by21, bz21;
        double roC = 8.2598; // 1.0;
        double pC = 3.5252755; //  1.0 / (ggg * M_inf * M_inf);
        double uC = 0.0; // -1.0;
        double vC = 0.0;
        double wC = 0.0;
        double bxC, byC, bzC;
        if (mgd == true)
        {
            bxC = 0.0; // -spi4 * (1.0 / (M_alf)) * cos(alpha * pi / 180.0);
            byC = 0.0; // -spi4 * (1.0 / (M_alf)) * sin(alpha * pi / 180.0);
            bzC = 0.0;
        }
        else
        {
            bxC = 0.0;
            byC = 0.0;
            bzC = 0.0;
        }

        int kk, kk2, l2, r2;
        for (int i = l; i <= r; i++)
        {
            ii = SOSED[i];
            if (ii >= 0)
            {
                x2 = X[ii];
                y2 = Y[ii];
                z2 = Z[ii];
                dx2 = DX[ii];
                dy2 = DY[ii];
                dz2 = DZ[ii];
                ro2 = RO1[ii];
                p2 = P1[ii];
                u2 = U1[ii];
                v2 = V1[ii];
                w2 = W1[ii];
                if (mgd == true)
                {
                    bx2 = BX1[ii];
                    by2 = BY1[ii];
                    bz2 = BZ1[ii];
                }
                else
                {
                    bx2 = 0.0;
                    by2 = 0.0;
                    bz2 = 0.0;
                }

                kk = SOSED2[i];
                if (kk >= 0)
                {
                    x3 = X[kk];
                    y3 = Y[kk];
                    z3 = Z[kk];
                    ro3 = RO1[kk];
                    p3 = P1[kk];
                    u3 = U1[kk];
                    v3 = V1[kk];
                    w3 = W1[kk];
                    if (mgd == true)
                    {
                        bx3 = BX1[kk];
                        by3 = BY1[kk];
                        bz3 = BZ1[kk];
                    }
                    else
                    {
                        bx3 = 0.0;
                        by3 = 0.0;
                        bz3 = 0.0;
                    }
                }
                else if(kk != -2)
                {
                    if (kk == -1)
                    {
                        x3 = x + 100.0;
                        y3 = y;
                        z3 = z;
                    }
                    else if (kk == -3)
                    {
                        x3 = x;
                        y3 = y + 100.0;
                        z3 = z;
                    }
                    else if (kk == -4)
                    {
                        x3 = x;
                        y3 = y - 100.0;
                        z3 = z;
                    }
                    else if (kk == -5)
                    {
                        x3 = x;
                        y3 = y;
                        z3 = z + 100.0;
                    }
                    else if (kk == -6)
                    {
                        x3 = x;
                        y3 = y;
                        z3 = z - 100.0;
                    }
                    ro3 = roC;
                    p3 = pC;
                    u3 = uC;
                    v3 = vC;
                    w3 = wC;
                    if (mgd == true)
                    {
                        bx3 = bxC;
                        by3 = byC;
                        bz3 = bzC;
                    }
                    else
                    {
                        bx3 = 0.0;
                        by3 = 0.0;
                        bz3 = 0.0;
                    }
                }
                else
                {
                    x3 = x - 100.0;
                    y3 = y;
                    z3 = z;
                    ro3 = ro;
                    p3 = p;
                    u3 = u;
                    v3 = v;
                    w3 = w;
                    bx3 = bx;
                    by3 = by;
                    bz3 = bz;
                }


                l2 = L[ii];
                r2 = R[ii];
                for (int ij = l2; ij <= r2; ij++)
                {
                    if (SOSED[ij] == ii)
                    {
                        kk2 = SOSED2[ij];
                        break;
                    }
                }


                if (kk2 >= 0)
                {
                    x4 = X[kk2];
                    y4 = Y[kk2];
                    z4 = Z[kk2];
                    ro4 = RO1[kk2];
                    p4 = P1[kk2];
                    u4 = U1[kk2];
                    v4 = V1[kk2];
                    w4 = W1[kk2];
                    if (mgd == true)
                    {
                        bx4 = BX1[kk2];
                        by4 = BY1[kk2];
                        bz4 = BZ1[kk2];
                    }
                    else
                    {
                        bx4 = 0.0;
                        by4 = 0.0;
                        bz4 = 0.0;
                    }
                }
                else if (kk2 != -2)
                {
                    if (kk2 == -1)
                    {
                        x4 = x + 100.0;
                        y4 = y;
                        z4 = z;
                    }
                    else if (kk2 == -4)
                    {
                        x4 = x;
                        y4 = y + 100.0;
                        z4 = z;
                    }
                    else if (kk2 == -4)
                    {
                        x4 = x;
                        y4 = y - 100.0;
                        z4 = z;
                    }
                    else if (kk2 == -5)
                    {
                        x4 = x;
                        y4 = y;
                        z4 = z + 100.0;
                    }
                    else if (kk2 == -6)
                    {
                        x4 = x;
                        y4 = y;
                        z4 = z - 100.0;
                    }
                    ro4 = roC;
                    p4 = pC;
                    u4 = uC;
                    v4 = vC;
                    w4 = wC;
                    if (mgd == true)
                    {
                        bx4 = bxC;
                        by4 = byC;
                        bz4 = bzC;
                    }
                    else
                    {
                        bx4 = 0.0;
                        by4 = 0.0;
                        bz4 = 0.0;
                    }
                }
                else
                {
                    x4 = x - 100.0;
                    y4 = y;
                    z4 = z;
                    ro4 = ro;
                    p4 = p;
                    u4 = u;
                    v4 = v;
                    w4 = w;
                    bx4 = bx;
                    by4 = by;
                    bz4 = bz;
                }



                double S = get_square(x, y, z, dx, dy, dz, x2, y2, z2, dx2, dy2, dz2, n1, n2, n3, dist);
                double dd = 0.0;
                if (n1 != 0)
                {
                    dd = dx;
                }
                else if (n2 != 0)
                {
                    dd = dy;
                }
                else if (n3 != 0)
                {
                    dd = dz;
                }
                else
                {
                    printf("Errrrrr 2323132214243\n");
                }

                double s1 = __dsqrt_rn(kv(x - x3) + kv(y - y3) + kv(z - z3));
                double s2 = __dsqrt_rn(kv(x - x2) + kv(y - y2) + kv(z - z2));
                double s3 = __dsqrt_rn(kv(x4 - x2) + kv(y4 - y2) + kv(z4 - z2));
                // p3, p, p2, p4
                f_TVD(dd,  p, p2, p3, p4, p12, p21, s1, s2, s3);
                f_TVD(dd,  ro, ro2, ro3, ro4, ro12, ro21, s1, s2, s3);
                f_TVD(dd,  u, u2, u3, u4, u12, u21, s1, s2, s3);
                f_TVD(dd,  v, v2, v3, v4, v12, v21, s1, s2, s3);
                f_TVD(dd,  w, w2, w3, w4, w12, w21, s1, s2, s3);
                f_TVD(dd,  bx, bx2, bx3, bx4, bx12, bx21, s1, s2, s3);
                f_TVD(dd,  by, by2, by3, by4, by12, by21, s1, s2, s3);
                f_TVD(dd,  bz, bz2, bz3, bz4, bz12, bz21, s1, s2, s3);

                if (ro12 <= 0.0)
                {
                    ro12 = ro;
                }
                if (p12 <= 0.0)
                {
                    p12 = p;
                }
                if (ro21 <= 0.0)
                {
                    ro21 = ro2;
                }
                if (p21 <= 0.0)
                {
                    p21 = p2;
                }


                if (diver == true)
                {
                    sks = n1 * (bx12 + bx21) / 2.0 + n2 * (by12 + by21) / 2.0 + n3 * (bz12 + bz21) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }
                Potok[8] = Potok[8] + sks * S;
                /*if (ro == 0 || ro2 == 0)
                {
                    printf("ewfwddasxxesdwedwhvb   %d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", index, ii, x2, y2, z2, dx2, dy2, dz2, x, y, z, dx, dy, dz);
                }*/
                tmin = min(tmin, HLLD_Alexashov(ro12, p12, u12, v12, w12, bx12, by12, bz12, ro21, p21, u21, v21, w21, bx21, by21, bz21, P, n1, n2, n3, dist, metod));
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -1)
            {
                double S = dy * dz * 4.0;
                n1 = 1.0;
                n2 = 0.0;
                n3 = 0.0;
                dist = dx;
                if (diver == true)
                {
                    sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }
                Potok[8] = Potok[8] + sks * S;
                if (ro == 0)
                {
                    printf("ewfwddasxxesdwedwhvb   %d,%lf,%lf,%lf,%lf,%lf,%lf\n", ii, x, y, z, dx, dy, dz);
                }
                tmin = min(tmin, HLLD_Alexashov(ro, p, u, v, w, bx, by, bz, roC, pC, uC, vC, wC, bxC, byC, bzC, P, n1, n2, n3, dist, metod));
                //  ћожно вручную выписать потоки дл€ ускорени€ времени
                /*double b2R = kv(bxC) + kv(byC) + kv(bzC);
                double ptR = pC + b2R / 2.0;
                double upt2 = (kv(uC) + kv(vC) + kv(wC)) / 2.0;
                double sbv2 = uC * bxC + vC * byC + wC * bzC;
                double e2 = pC / g1 + roC * upt2 + b2R / 2.0;

                P[0] = roC * uC;
                P[1] = roC * uC * uC + ptR - kv(bxC);
                P[2] = roC * uC * vC - bxC * byC;
                P[3] = roC * uC * wC - bxC * bzC;
                P[7] = (e2 + ptR) * uC - bxC * sbv2;
                P[4] = 0.0;
                P[5] = uC * byC - vC * bxC;
                P[6] = uC * bzC - wC * bxC;*/

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -2)
            {
                double S = dy * dz * 4.0;
                n1 = -1.0;
                n2 = 0.0;
                n3 = 0.0;
                dist = dx;
                if (diver == true)
                {
                    sks = n1 * bx + n2 * by + n3 * bz;
                }
                else
                {
                    sks = 0.0;
                }
                Potok[8] = Potok[8] + sks * S;
                double uu = u;
                if ((uu >= -0.1) && (__dsqrt_rn(y * y + z * z) < 150.0))
                {
                    uu = -1.0;
                }
                /*if ((bx > 0.0))
                {
                    bx = -0.1;
                }*/
                tmin = min(tmin, HLLD_Alexashov(ro, p, u, v, w, bx, by, bz, ro, p, uu, v, w, bx, by, bz, P, n1, n2, n3, dist, metod));
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -3)
            {
                double S = dx * dz * 4.0;
                n1 = 0.0;
                n2 = 1.0;
                n3 = 0.0;
                dist = dy;
                if (diver == true)
                {
                    sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }
                Potok[8] = Potok[8] + sks * S;
                tmin = min(tmin, HLLD_Alexashov(ro, p, u, v, w, bx, by, bz, roC, pC, uC, vC, wC, bxC, byC, bzC, P, n1, n2, n3, dist, metod));
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -4)
            {
                double S = dx * dz * 4.0;
                n1 = 0.0;
                n2 = -1.0;
                n3 = 0.0;
                dist = dy;
                if (diver == true)
                {
                    sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }

                Potok[8] = Potok[8] + sks * S;
                tmin = min(tmin, HLLD_Alexashov(ro, p, u, v, w, bx, by, bz, roC, pC, uC, vC, wC, bxC, byC, bzC, P, n1, n2, n3, dist, metod));
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -5)
            {
                double S = dy * dx * 4.0;
                n1 = 0.0;
                n2 = 0.0;
                n3 = 1.0;
                dist = dz;
                if (diver == true)
                {
                    sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }

                Potok[8] = Potok[8] + sks * S;
                tmin = min(tmin, HLLD_Alexashov(ro, p, u, v, w, bx, by, bz, roC, pC, uC, vC, wC, bxC, byC, bzC, P, n1, n2, n3, dist, metod));
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else if (ii == -6)
            {
                double S = dy * dx * 4.0;
                n1 = 0.0;
                n2 = 0.0;
                n3 = -1.0;
                dist = dz;
                if (diver == true)
                {
                    sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
                }
                else
                {
                    sks = 0.0;
                }

                Potok[8] = Potok[8] + sks * S;
                tmin = min(tmin, HLLD_Alexashov(ro, p, u, v, w, bx, by, bz, roC, pC, uC, vC, wC, bxC, byC, bzC, P, n1, n2, n3, dist, metod));
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
            }
            else
            {
                printf("Error 12438jdyu. Ne doljni suda popadat = %d \n", ii);
            }
        }

        double ro33, p33, u33, v33, w33, bx33, by33, bz33;

        ro33 = ro - *T_do * Potok[0] / Volume;
        if (ro33 <= 0.0)
        {
            printf("ERROR -  dssdbfhfshjskfutytqqazz\n");
            printf("%lf, %lf, %lf, %lf\n", x, y, z, ro3);
            ro33 = 0.0001;
        }
        u33 = (ro * u - *T_do * (Potok[1] + (bx / cpi4) * Potok[8]) / Volume) / ro33;
        v33 = (ro * v - *T_do * (Potok[2] + (by / cpi4) * Potok[8]) / Volume) / ro33;
        w33 = (ro * w - *T_do * (Potok[3] + (bz / cpi4) * Potok[8]) / Volume) / ro33;
        bx33 = (bx - *T_do * (Potok[4] + u * Potok[8]) / Volume);
        by33 = (by - *T_do * (Potok[5] + v * Potok[8]) / Volume);
        bz33 = (bz - *T_do * (Potok[6] + w * Potok[8]) / Volume);
        p33 = ((U8(ro, p, u, v, w, bx, by, bz) - *T_do * (Potok[7] + (skk(u, v, w, bx, by, bz) / cpi4) * Potok[8])//
            / Volume) - 0.5 * ro33 * kvv(u33, v33, w33) - kvv(bx33, by33, bz33) / cpi8) * (ggg - 1.0);
        //u3 = (ro * u - *T_do * (Potok[1] + (bx) * Potok[8]) / Volume) / ro3;
        //v3 = (ro * v - *T_do * (Potok[2] + (by) * Potok[8]) / Volume) / ro3;
        //w3 = (ro * w - *T_do * (Potok[3] + (bz) * Potok[8]) / Volume) / ro3;
        //bx3 = (bx - *T_do * (Potok[4] + u * Potok[8]) / Volume);
        //by3 = (by - *T_do * (Potok[5] + v * Potok[8]) / Volume);
        //bz3 = (bz - *T_do * (Potok[6] + w * Potok[8]) / Volume);
        //p3 = ((U8(ro, p, u, v, w, bx, by, bz) - *T_do * (Potok[7] + (skk(u, v, w, bx, by, bz)) * Potok[8])//
        //    / Volume) - 0.5 * ro3 * kvv(u3, v3, w3) - kvv(bx3, by3, bz3)) * (ggg - 1.0);
        if (p33 <= 0)
        {
            p33 = 0.000001;
        }

        RO2[index] = ro33;
        P2[index] = p33;
        U2[index] = u33;
        V2[index] = v33;
        W2[index] = w33;
        BX2[index] = bx33;
        BY2[index] = by33;
        BZ2[index] = bz33;

        if (*T > tmin)
        {
            *T = tmin;
            __threadfence();
        }
    }

}

__global__ void Cuda_main_HLLDQ_TVD2(int* NN, double* X, double* Y, double* Z, double* DX, double* DY, double* DZ,//
    double* RO1, double* RO2, double* Q1, double* Q2, double* P1, double* P2, double* U1, double* U2, double* V1, double* V2,//
    double* W1, double* W2, double* BX1, double* BY1, double* BZ1, double* BX2, double* BY2, double* BZ2,//
    int* SOSED, int* SOSED2, int* SOSED3, int* L, int* R, double* T, double* T_do, int step_, double M_inf_, bool mgd = true, bool diver = true, int metod = 0)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; // √лобальный индекс текущего потока
    if (index > * NN - 1)
    {
        return;
    }
    double x, y, z, dx, dy, dz, ro, p, u, v, w, bx, by, bz, Q;
    int l = L[index];
    int r = R[index];
    int my_metod = metod;
    x = X[index];
    y = Y[index];
    z = Z[index];
    dx = DX[index];
    dy = DY[index];
    dz = DZ[index];
    ro = RO1[index];
    p = P1[index];
    u = U1[index];
    v = V1[index];
    w = W1[index];
    Q = Q1[index];
    bx = BX1[index];
    by = BY1[index];
    bz = BZ1[index];




    //double ddd = kv(y) + kv(z);
    double ddd2 = kv(x) + kv(y) + kv(z);

    if (ddd2 <= ddist * ddist) // || (ddd <= 4.0 && x > -5 && x < 0) ) //(ddd < 5.76 || ddd2 <= 2.0) //1.5
    {
        RO2[index] = ro;
        P2[index] = p;
        U2[index] = u;
        V2[index] = v;
        W2[index] = w;
        BX2[index] = bx;
        BY2[index] = by;
        BZ2[index] = bz;
        Q2[index] = Q;
    }
    else
    {
        /*metod = 3;
        if (ddd2 <= 0.73 * 0.73)
        {
            metod = 2;
        }*/

        double n1 = 0.0;
        double n2 = 0.0;
        double n3 = 0.0;
        double dist = 0.0;
        double P[8] = { 0.0 };
        P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;
        double Potok[10] = { 0.0 };
        Potok[0] = Potok[1] = Potok[2] = Potok[3] = Potok[4] = Potok[5] = Potok[6] = Potok[7] = Potok[8] = Potok[9] = 0.0;
        double PQ;
        double tmin = 1000;
        double Volume = dx * dy * dz * 8.0;
        int ii = 0;
        double Q_2, x2, y2, z2, dx2, dy2, dz2, ro2, p2, u2, v2, w2, bx2, by2, bz2, sks;
        double Q_3, x3, y3, z3, dx3, dy3, dz3, ro3, p3, u3, v3, w3, bx3, by3, bz3;
        double Q_4, x4, y4, z4, dx4, dy4, dz4, ro4, p4, u4, v4, w4, bx4, by4, bz4;
        double x12, y12, z12, dx12, dy12, dz12, ro12, p12, u12, v12, w12, bx12, by12, bz12, Q12, Q21;
        double x21, y21, z21, dx21, dy21, dz21, ro21, p21, u21, v21, w21, bx21, by21, bz21;
        double su1, sv1, sw1, su2, sv2, sw2, sro1, sro2, sp1, sp2;
        double ur, up, uz;
        double roC = 1.0; // 8.2598; //  1.0;
        double pC = 1.0 / (ggg); // 1.0 / (ggg * M_inf * M_inf);
        double uC = M_infty; // -1.0;
        double vC = 0.0;
        double wC = 0.0;
        double QC = 100.0;
        double bxC, byC, bzC;
        bxC = 0.0; 
        byC = 0.0; 
        bzC = B_inf;

        int kk, kk2, l2, r2;
        for (int i = l; i <= r; i++)
        {
            my_metod = metod;
            ii = SOSED[i];
            if (ii >= 0)
            {
                x2 = X[ii];
                y2 = Y[ii];
                z2 = Z[ii];
                dx2 = DX[ii];
                dy2 = DY[ii];
                dz2 = DZ[ii];
                ro2 = RO1[ii];
                p2 = P1[ii];
                u2 = U1[ii];
                v2 = V1[ii];
                w2 = W1[ii];
                Q_2 = Q1[ii];
                bx2 = BX1[ii];
                by2 = BY1[ii];
                bz2 = BZ1[ii];

                double ddd3 = kv((z + z2) / 2.0) + kv((x + x2) / 2.0) + kv((y + y2) / 2.0);

                kk = SOSED2[i];
                if (kk != -1 && ddd3 > (ddist2 * ddist2)) //&& ddd2 > 0.8
                {
                    dx3 = DX[kk];
                    x3 = X[kk];
                    y3 = Y[kk];
                    z3 = Z[kk];
                    ro3 = RO1[kk];
                    p3 = P1[kk];
                    u3 = U1[kk];
                    v3 = V1[kk];
                    w3 = W1[kk];
                    Q_3 = Q1[kk];
                    bx3 = BX1[kk];
                    by3 = BY1[kk];
                    bz3 = BZ1[kk];

                    kk2 = SOSED3[i];

                    dx4 = DX[kk2];
                    x4 = X[kk2];
                    y4 = Y[kk2];
                    z4 = Z[kk2];
                    ro4 = RO1[kk2];
                    p4 = P1[kk2];
                    u4 = U1[kk2];
                    v4 = V1[kk2];
                    w4 = W1[kk2];
                    Q_4 = Q1[kk2];
                    bx4 = BX1[kk2];
                    by4 = BY1[kk2];
                    bz4 = BZ1[kk2];


                    double S = get_square(x, y, z, dx, dy, dz, x2, y2, z2, dx2, dy2, dz2, n1, n2, n3, dist);
                    double dd = 0.0;
                    if (n1 != 0)
                    {
                        dd = dx;
                    }
                    else if (n2 != 0)
                    {
                        dd = dy;
                    }
                    else if (n3 != 0)
                    {
                        dd = dz;
                    }
                    else
                    {
                        printf("Errrrrr 2323132214243\n");
                    }

                    double s1 = __dsqrt_rn(kv(x - x3) + kv(y - y3) + kv(z - z3));
                    double s2 = __dsqrt_rn(kv(x - x2) + kv(y - y2) + kv(z - z2));
                    double s3 = __dsqrt_rn(kv(x4 - x2) + kv(y4 - y2) + kv(z4 - z2));
                    // p3, p, p2, p4
                    f_TVD(dd, Q, Q_2, Q_3, Q_4, Q12, Q21, s1, s2, s3);
                    f_TVD(dd, p, p2, p3, p4, p12, p21, s1, s2, s3);
                    f_TVD(dd, ro, ro2, ro3, ro4, ro12, ro21, s1, s2, s3);
                    f_TVD(dd, u, u2, u3, u4, u12, u21, s1, s2, s3);
                    f_TVD(dd, v, v2, v3, v4, v12, v21, s1, s2, s3);
                    f_TVD(dd, w, w2, w3, w4, w12, w21, s1, s2, s3);
                    f_TVD(dd, bx, bx2, bx3, bx4, bx12, bx21, s1, s2, s3);
                    f_TVD(dd, by, by2, by3, by4, by12, by21, s1, s2, s3);
                    f_TVD(dd, bz, bz2, bz3, bz4, bz12, bz21, s1, s2, s3);
                    if (Q12 <= 0.0)
                    {
                        Q12 = Q;
                    }
                    if (Q21 <= 0.0)
                    {
                        Q21 = Q_2;
                    }
                    if (ro12 <= 0.0)
                    {
                        ro12 = ro;
                    }
                    if (p12 <= 0.0)
                    {
                        p12 = p;
                    }
                    if (ro21 <= 0.0)
                    {
                        ro21 = ro2;
                    }
                    if (p21 <= 0.0)
                    {
                        p21 = p2;
                    }

                    //if (kvv(u, v, w) / (ggg * p / ro) > 100.0 || kvv(u2, v2, w2) / (ggg * p2 / ro2) > 100.0 || 
                    //    kvv(u3, v3, w3) / (ggg * p3 / ro3) > 100.0 || kvv(u4, v4, w4) / (ggg * p4 / ro4) > 100.0)
                    if(false)
                    {
                        my_metod = 1;

                        ro12 = ro;
                        p12 = p;
                        Q12 = Q;
                        u12 = u;
                        v12 = v;
                        w12 = w;
                        bx12 = bx;
                        by12 = by;
                        bz12 = bz;

                        ro21 = ro2;
                        p21 = p2;
                        Q21 = Q_2;
                        u21 = u2;
                        v21 = v2;
                        w21 = w2;
                        bx21 = bx2;
                        by21 = by2;
                        bz21 = bz2;
                    }


                    //if(ddd2 < 0.8) my_metod = 1;


                    sks = n1 * (bx12 + bx21) / 2.0 + n2 * (by12 + by21) / 2.0 + n3 * (bz12 + bz21) / 2.0;

                    Potok[8] = Potok[8] + sks * S;



                    if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                    {
                        tmin = min(tmin, HLLDQ_Alexashov(ro12, Q12, p12, u12, v12, w12, bx12, by12, bz12, ro21, Q21, p21, u21, v21, w21, bx21, by21, bz21, P, PQ, n1, n2, n3, dist, my_metod));
                    }
                    else
                    {
                        tmin = min(tmin, HLLDQ_Korolkov(ro12, Q12, p12, u12, v12, w12, bx12, by12, bz12, ro21, Q21, p21, u21, v21, w21, bx21, by21, bz21, P, PQ, n1, n2, n3, dist, my_metod));
                    }


                    for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                    {
                        Potok[k] = Potok[k] + P[k] * S;
                    }
                    Potok[9] = Potok[9] + PQ * S;
                }
                else
                {
                    double S = get_square(x, y, z, dx, dy, dz, x2, y2, z2, dx2, dy2, dz2, n1, n2, n3, dist);

                    sks = n1 * (bx + bx2) / 2.0 + n2 * (by + by2) / 2.0 + n3 * (bz + bz2) / 2.0;
                    Potok[8] = Potok[8] + sks * S;

                    su1 = u;
                    sv1 = v;
                    sw1 = w;

                    su2 = u2;
                    sv2 = v2;
                    sw2 = w2;

                    sro1 = ro;
                    sro2 = ro2;

                    sp1 = p;
                    sp2 = p2;

                    // ƒелаем перенос в сферической —  
                    if (ddd3 <= (ddist2 * ddist2))
                    //if (kvv(u, v, w) / (ggg * p / ro) > 100.0 && kvv(u2, v2, w2) / (ggg * p2 / ro2) > 100.0)
                    {
                        //my_metod = 1;

                        spherical_skorost(z, x, y, w, u, v, ur, up, uz);
                        dekard_skorost((z + z2) / 2.0, (x + x2) / 2.0, (y + y2) / 2.0, ur, up, uz, sw1, su1, sv1);

                        spherical_skorost(z2, x2, y2, w2, u2, v2, ur, up, uz);
                        dekard_skorost((z + z2) / 2.0, (x + x2) / 2.0, (y + y2) / 2.0, ur, up, uz, sw2, su2, sv2);

                        sro1 = ro * (kv(z) + kv(x) + kv(y)) / ddd3;
                        sro2 = ro2 * (kv(z2) + kv(x2) + kv(y2)) / ddd3;

                        sp1 = p * pow((kv(z) + kv(x) + kv(y)) / ddd3, ggg);
                        sp2 = p2 * pow((kv(z2) + kv(x2) + kv(y2)) / ddd3, ggg);
                    }



                    /*if (!kor_Sol || metod == 1 || metod == 3)
                    {
                        tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro2, Q_2, p2, u2, v2, w2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, metod));
                    }
                    else
                    {
                        tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro2, Q_2, p2, u2, v2, w2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, metod));
                    }*/

                    if (my_metod <= 1 || my_metod == 3)//(y * y + z * z < 225 && y2 * y2 + z2 * z2 < 225 && x > -15 && x2 > -15 && x < 8 && x2 < 8  && step_ > 10000)
                    {
                        tmin = min(tmin, HLLDQ_Alexashov(sro1, Q, sp1, su1, sv1, sw1, bx, by, bz, sro2, Q_2, sp2, su2, sv2, sw2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, my_metod));
                    }
                    else
                    {
                        tmin = min(tmin, HLLDQ_Korolkov(sro1, Q, sp1, su1, sv1, sw1, bx, by, bz, sro2, Q_2, sp2, su2, sv2, sw2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, my_metod));
                    }


                    for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                    {
                        Potok[k] = Potok[k] + P[k] * S;
                    }
                    Potok[9] = Potok[9] + PQ * S;
                }

            }
            else if (ii == -1)
            {
                double S = dy * dz * 4.0;
                n1 = 1.0;
                n2 = 0.0;
                n3 = 0.0;
                dist = dx;
                sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;

                /*double uu = u;
                if (uu < 0.0)
                {
                    uu = 0.0;
                }*/
                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, my_metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, my_metod));
                }

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -2)
            {
                double S = dy * dz * 4.0;
                n1 = -1.0;
                n2 = 0.0;
                n3 = 0.0;
                dist = dx;
                sks = n1 * bx + n2 * by + n3 * bz;

                Potok[8] = Potok[8] + sks * S;
                double uu = u;
                if (uu > M_infty/3.0)
                {
                    uu = M_infty;
                }
                /*else if (uu > -0.01)
                {
                    uu = -0.01;
                }*/

                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, uu, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, uu, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }

                /*double t1, t2, t3, m1, m2, m3;
                double bx_L = bx / spi4;
                double by_L = by / spi4;
                double bz_L = bz / spi4;
                t1 = 0.0;
                t2 = 0.0;
                t3 = 1.0;
                m1 = 0.0;
                m2 = 1.0;
                m3 = 0.0;
                double u1 = uu * n1 + v * n2 + w * n3;
                double v1 = uu * t1 + v * t2 + w * t3;
                double w1 = uu * m1 + v * m2 + w * m3;
                double bn1, bt1, bm1;
                bn1 = bx_L * n1 + by_L * n2 + bz_L * n3;
                bt1 = bx_L * t1 + by_L * t2 + bz_L * t3;
                bm1 = bx_L * m1 + by_L * m2 + bz_L * m3;
                double uu_L = (kv(uu) + kv(v) + kv(w)) / 2.0;
                double bb_L = kv(bx_L) + kv(by_L) + kv(bz_L);
                double e1 = p / g1 + ro * uu_L + bb_L / 2.0;
                double pTL = p + bb_L / 2.0;

                double PO[9];

                PO[0] = ro * u1;
                PO[1] = ro * u1 * u1 + pTL - kv(bn1);
                PO[2] = ro * u1 * v1 - bn1 * bt1;
                PO[3] = ro * u1 * w1 - bn1 * bm1;
                PO[4] = (e1 + pTL) * u1 - bn1 * (u1 * bn1 + v1 * bt1 + w1 * bm1);
                PO[5] = 0.0;
                PO[6] = u1 * bt1 - v1 * bn1;
                PO[7] = u1 * bm1 - w1 * bn1;
                PO[8] = Q * u1;


                P[1] = n1 * PO[1] + t1 * PO[2] + m1 * PO[3];
                P[2] = n2 * PO[1] + t2 * PO[2] + m2 * PO[3];
                P[3] = n3 * PO[1] + t3 * PO[2] + m3 * PO[3];
                P[5] = spi4 * (n1 * PO[5] + t1 * PO[6] + m1 * PO[7]);
                P[6] = spi4 * (n2 * PO[5] + t2 * PO[6] + m2 * PO[7]);
                P[7] = spi4 * (n3 * PO[5] + t3 * PO[6] + m3 * PO[7]);
                P[0] = PO[0];
                P[4] = PO[4];
                PQ = PO[8];

                double SWAP = P[4];
                P[4] = P[5];
                P[5] = P[6];
                P[6] = P[7];
                P[7] = SWAP;*/

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -3)
            {
                double S = dx * dz * 4.0;
                n1 = 0.0;
                n2 = 1.0;
                n3 = 0.0;
                dist = dy;

                sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;

                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -4)
            {
                double S = dx * dz * 4.0;
                n1 = 0.0;
                n2 = -1.0;
                n3 = 0.0;
                dist = dy;
                sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;

                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, -v, w, -bx, by, -bz, P, PQ, n1, n2, n3, dist, my_metod));
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, pC, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                    
                }
                else
                {
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, -v, w, -bx, by, -bz, P, PQ, n1, n2, n3, dist, my_metod));
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                    
                }

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -5)
            {
                double S = dy * dx * 4.0;
                n1 = 0.0;
                n2 = 0.0;
                n3 = 1.0;
                dist = dz;
                sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;


                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -6)
            {
                double S = dy * dx * 4.0;
                n1 = 0.0;
                n2 = 0.0;
                n3 = -1.0;
                dist = dz;
                sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;


                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, pC, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else
            {
                printf("Error 12438jdyu. Ne doljni suda popadat = %d \n", ii);
            }
        }


        double q2_1 = 0.0, q2_2 = 0.0, q2_3 = 0.0, q3 = 0.0;

        if (true)//(istoch == true)
        {
            double u_H4 = M_infty, v_H4 = 0.0, w_H4 = 0.0, ro_H4 = 1.0, p_H4 = 1.0 / (2.0 * ggg);

            double U_M_H4 = sqrt(kv(u - u_H4) + kv(v - v_H4) + kv(w - w_H4) + (64.0 / (9.0 * pi)) //
                * (p / ro + 2.0 * p_H4 / ro_H4));

            double U_H4 = sqrt(kv(u - u_H4) + kv(v - v_H4) + kv(w - w_H4) + (4.0 / pi) //
                * (p / ro + 2.0 * p_H4 / ro_H4));

            double sigma_H4 = kv(1.0 - a_2 * log(U_M_H4));

            double nu_H4 = ro * ro_H4 * U_M_H4 * sigma_H4;

            q2_1 = (1.5 / Kn_) * (nu_H4 * (u_H4 - u));
            q2_2 = (1.5 / Kn_) * (nu_H4 * (v_H4 - v));
            q2_3 = (1.5 / Kn_) * (nu_H4 * (w_H4 - w));


            q3 = (1.5 / Kn_) * (nu_H4 * ((kv(u_H4) + kv(v_H4) + kv(w_H4) - kv(u) - kv(v) - kv(w)) / 2.0 + //
                (U_H4 / U_M_H4) * (2.0 * p_H4 / ro_H4 - p / ro)));
        }


        double ro33, p33, u33, v33, w33, bx33, by33, bz33, Q33;


        Q33 = Q - *T_do * Potok[9] / Volume;
        ro33 = ro - *T_do * Potok[0] / Volume;
        if (ro33 <= 0.0)
        {
            printf("ERROR -  dssdbfhfshjskfutytqqazz\n");
            printf("%lf, %lf, %lf, %lf\n", x, y, z, ro33);
            ro33 = ro;
        }
        u33 = (ro * u - *T_do * (Potok[1] + (bx / cpi4) * Potok[8] - q2_1 * Volume) / Volume) / ro33;
        v33 = (ro * v - *T_do * (Potok[2] + (by / cpi4) * Potok[8] - q2_2 * Volume) / Volume) / ro33;
        w33 = (ro * w - *T_do * (Potok[3] + (bz / cpi4) * Potok[8] - q2_3 * Volume) / Volume) / ro33;
        bx33 = (bx - *T_do * (Potok[4] + u * Potok[8]) / Volume);
        by33 = (by - *T_do * (Potok[5] + v * Potok[8]) / Volume);
        bz33 = (bz - *T_do * (Potok[6] + w * Potok[8]) / Volume);
        p33 = ((U8(ro, p, u, v, w, bx, by, bz) - *T_do * (Potok[7] + (skk(u, v, w, bx, by, bz) / cpi4) * Potok[8] - q3 * Volume)//
            / Volume) - 0.5 * ro33 * kvv(u33, v33, w33) - kvv(bx33, by33, bz33) / cpi8) * (ggg - 1.0);
        //u3 = (ro * u - *T_do * (Potok[1] + (bx) * Potok[8]) / Volume) / ro3;
        //v3 = (ro * v - *T_do * (Potok[2] + (by) * Potok[8]) / Volume) / ro3;
        //w3 = (ro * w - *T_do * (Potok[3] + (bz) * Potok[8]) / Volume) / ro3;
        //bx3 = (bx - *T_do * (Potok[4] + u * Potok[8]) / Volume);
        //by3 = (by - *T_do * (Potok[5] + v * Potok[8]) / Volume);
        //bz3 = (bz - *T_do * (Potok[6] + w * Potok[8]) / Volume);
        //p3 = ((U8(ro, p, u, v, w, bx, by, bz) - *T_do * (Potok[7] + (skk(u, v, w, bx, by, bz)) * Potok[8])//
        //    / Volume) - 0.5 * ro3 * kvv(u3, v3, w3) - kvv(bx3, by3, bz3)) * (ggg - 1.0);
        if (p33 <= 0)
        {
            p33 = 0.000001;
        }

        Q2[index] = Q33;
        RO2[index] = ro33;
        P2[index] = p33;
        U2[index] = u33;
        V2[index] = v33;
        W2[index] = w33;
        BX2[index] = bx33;
        BY2[index] = by33;
        BZ2[index] = bz33;

        if (*T > tmin)
        {
            *T = tmin;
            __threadfence();
        }
    }

}

__global__ void Cuda_main_HLLDQ_TVD(int* NN, double* X, double* Y, double* Z, double* DX, double* DY, double* DZ,//
    double* RO1, double* RO2, double* Q1, double* Q2, double* P1, double* P2, double* U1, double* U2, double* V1, double* V2,//
    double* W1, double* W2, double* BX1, double* BY1, double* BZ1, double* BX2, double* BY2, double* BZ2,//
    int* SOSED, int* SOSED2, int* L, int* R, double* T, double* T_do, int step_, double M_inf_, bool mgd = true, bool diver = true, int metod = 0)
{
    // Ќадо подправить граничные услови€ здесь!!!
    int index = blockIdx.x * blockDim.x + threadIdx.x; // √лобальный индекс текущего потока
    if (index > * NN - 1)
    {                
        return;     
    }
    double x, y, z, dx, dy, dz, ro, p, u, v, w, bx, by, bz, Q;
    int l = L[index];
    int r = R[index];
    x = X[index];
    y = Y[index];
    z = Z[index];
    dx = DX[index];
    dy = DY[index]; 
    dz = DZ[index];
    ro = RO1[index];
    p = P1[index];
    u = U1[index];
    v = V1[index];
    w = W1[index];
    Q = Q1[index];
    if (mgd == true)
    {
        bx = BX1[index];
        by = BY1[index];
        bz = BZ1[index];
    }
    else
    {
        bx = 0.0;
        by = 0.0;
        bz = 0.0;
    }




    //double ddd = kv(y) + kv(z);
    double ddd2 = kv(x) + kv(y) + kv(z);

    if (ddd2 <= ddist * ddist) // || (ddd <= 4.0 && x > -5 && x < 0) ) //(ddd < 5.76 || ddd2 <= 2.0) //1.5
    {
        RO2[index] = ro;
        P2[index] = p;
        U2[index] = u;
        V2[index] = v;
        W2[index] = w;
        BX2[index] = bx;
        BY2[index] = by;
        BZ2[index] = bz;
        Q2[index] = Q;
    }
    else
    {
        if (ddd2 < 1.0)
        {
            metod = 2;
        }
        double n1 = 0.0;
        double n2 = 0.0;
        double n3 = 0.0;
        double dist = 0.0;
        double P[8] = { 0.0 };
        P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;
        double Potok[10] = { 0.0 };
        Potok[0] = Potok[1] = Potok[2] = Potok[3] = Potok[4] = Potok[5] = Potok[6] = Potok[7] = Potok[8] = Potok[9] = 0.0;
        double PQ;
        double tmin = 1000;
        double Volume = dx * dy * dz * 8.0;
        int ii = 0;
        double Q_2, x2, y2, z2, dx2, dy2, dz2, ro2, p2, u2, v2, w2, bx2, by2, bz2, sks;
        double Q_3, x3, y3, z3, dx3, dy3, dz3, ro3, p3, u3, v3, w3, bx3, by3, bz3;
        double Q_4, x4, y4, z4, dx4, dy4, dz4, ro4, p4, u4, v4, w4, bx4, by4, bz4;
        double x12, y12, z12, dx12, dy12, dz12, ro12, p12, u12, v12, w12, bx12, by12, bz12, Q12, Q21;
        double x21, y21, z21, dx21, dy21, dz21, ro21, p21, u21, v21, w21, bx21, by21, bz21;
        double roC = 1.0; // 8.2598; //  1.0;
        double pC = 1.0 / (ggg); // 1.0 / (ggg * M_inf * M_inf);
        double uC = 0.0; // -1.0;
        double vC = 0.0;
        double wC = 0.0;
        double QC = 100.0;
        double bxC, byC, bzC;
        if (mgd == true)
        {
            bxC = BX1[index];
            byC = BY1[index];
            bzC = BZ1[index];
        }
        else
        {
            bxC = 0.0;
            byC = 0.0;
            bzC = 0.0;
        }

        int kk, kk2, l2, r2;
        for (int i = l; i <= r; i++)
        {
            ii = SOSED[i];
            if (ii >= 0)
            {
                x2 = X[ii];
                y2 = Y[ii];
                z2 = Z[ii];
                dx2 = DX[ii];
                dy2 = DY[ii];
                dz2 = DZ[ii];
                ro2 = RO1[ii];
                p2 = P1[ii];
                u2 = U1[ii];
                v2 = V1[ii];
                w2 = W1[ii];
                Q_2 = Q1[ii];
                if (mgd == true)
                {
                    bx2 = BX1[ii];
                    by2 = BY1[ii];
                    bz2 = BZ1[ii];
                }
                else
                {
                    bx2 = 0.0;
                    by2 = 0.0;
                    bz2 = 0.0;
                }

                kk = SOSED2[i];
                if (kk >= 0)
                {
                    dx3 = DX[kk];
                    x3 = X[kk];
                    y3 = Y[kk];
                    z3 = Z[kk];
                    ro3 = RO1[kk];
                    p3 = P1[kk];
                    u3 = U1[kk];
                    v3 = V1[kk];
                    w3 = W1[kk];
                    Q_3 = Q1[kk];
                    if (mgd == true)
                    {
                        bx3 = BX1[kk];
                        by3 = BY1[kk];
                        bz3 = BZ1[kk];
                    }
                    else
                    {
                        bx3 = 0.0;
                        by3 = 0.0;
                        bz3 = 0.0;
                    }
                }
                else if (kk != -2)
                {
                    if (kk == -1)
                    {
                        x3 = x + 100.0;
                        y3 = y;
                        z3 = z;
                    }
                    else if (kk == -3)
                    {
                        x3 = x;
                        y3 = y + 100.0;
                        z3 = z;
                    }
                    else if (kk == -4)
                    {
                        x3 = x;
                        y3 = y - 100.0;
                        z3 = z;
                    }
                    else if (kk == -5)
                    {
                        x3 = x;
                        y3 = y;
                        z3 = z + 100.0;
                    }
                    else if (kk == -6)
                    {
                        x3 = x;
                        y3 = y;
                        z3 = z - 100.0;
                    }
                    ro3 = roC;
                    p3 = pC;
                    u3 = uC;
                    v3 = vC;
                    w3 = wC;
                    Q_3 = QC;
                    if (mgd == true)
                    {
                        bx3 = bxC;
                        by3 = byC;
                        bz3 = bzC;
                    }
                    else
                    {
                        bx3 = 0.0;
                        by3 = 0.0;
                        bz3 = 0.0;
                    }
                }
                else
                {
                    x3 = x - 100.0;
                    y3 = y;
                    z3 = z;
                    Q_3 = Q;
                    ro3 = ro;
                    p3 = p;
                    u3 = u;
                    v3 = v;
                    w3 = w;
                    bx3 = bx;
                    by3 = by;
                    bz3 = bz;
                }


                l2 = L[ii];
                r2 = R[ii];
                for (int ij = l2; ij <= r2; ij++)
                {
                    if (SOSED[ij] == ii)
                    {
                        kk2 = SOSED2[ij];
                        break;
                    }
                }


                if (kk2 >= 0)
                {
                    dx4 = DX[kk2];
                    x4 = X[kk2];
                    y4 = Y[kk2];
                    z4 = Z[kk2];
                    ro4 = RO1[kk2];
                    p4 = P1[kk2];
                    u4 = U1[kk2];
                    v4 = V1[kk2];
                    w4 = W1[kk2];
                    Q_4 = Q1[kk2];
                    if (mgd == true)
                    {
                        bx4 = BX1[kk2];
                        by4 = BY1[kk2];
                        bz4 = BZ1[kk2];
                    }
                    else
                    {
                        bx4 = 0.0;
                        by4 = 0.0;
                        bz4 = 0.0;
                    }
                }
                else if (kk2 != -2)
                {
                    if (kk2 == -1)
                    {
                        x4 = x + 100.0;
                        y4 = y;
                        z4 = z;
                    }
                    else if (kk2 == -3)
                    {
                        x4 = x;
                        y4 = y + 100.0;
                        z4 = z;
                    }
                    else if (kk2 == -4)
                    {
                        x4 = x;
                        y4 = y - 100.0;
                        z4 = z;
                    }
                    else if (kk2 == -5)
                    {
                        x4 = x;
                        y4 = y;
                        z4 = z + 100.0;
                    }
                    else if (kk2 == -6)
                    {
                        x4 = x;
                        y4 = y;
                        z4 = z - 100.0;
                    }
                    ro4 = roC;
                    p4 = pC;
                    u4 = uC;
                    v4 = vC;
                    w4 = wC;
                    Q_4 = QC;
                    if (mgd == true)
                    {
                        bx4 = bxC;
                        by4 = byC;
                        bz4 = bzC;
                    }
                    else
                    {
                        bx4 = 0.0;
                        by4 = 0.0;
                        bz4 = 0.0;
                    }
                }
                else
                {
                    x4 = x - 100.0;
                    y4 = y;
                    z4 = z;
                    ro4 = ro;
                    p4 = p;
                    u4 = u;
                    v4 = v;
                    w4 = w;
                    bx4 = bx;
                    by4 = by;
                    bz4 = bz;
                    Q_4 = Q;
                }


                if (fabs(dx - dx2) < 0.001 && fabs(dx - dx3) < 0.001 && fabs(dx - dx4) < 0.001)
                {
                    double S = get_square(x, y, z, dx, dy, dz, x2, y2, z2, dx2, dy2, dz2, n1, n2, n3, dist);
                    double dd = 0.0;
                    if (n1 != 0)
                    {
                        dd = dx;
                    }
                    else if (n2 != 0)
                    {
                        dd = dy;
                    }
                    else if (n3 != 0)
                    {
                        dd = dz;
                    }
                    else
                    {
                        printf("Errrrrr 2323132214243\n");
                    }

                    double s1 = __dsqrt_rn(kv(x - x3) + kv(y - y3) + kv(z - z3));
                    double s2 = __dsqrt_rn(kv(x - x2) + kv(y - y2) + kv(z - z2));
                    double s3 = __dsqrt_rn(kv(x4 - x2) + kv(y4 - y2) + kv(z4 - z2));
                    // p3, p, p2, p4
                    f_TVD(dd, Q, Q_2, Q_3, Q_4, Q12, Q21, s1, s2, s3);
                    f_TVD(dd, p, p2, p3, p4, p12, p21, s1, s2, s3);
                    f_TVD(dd, ro, ro2, ro3, ro4, ro12, ro21, s1, s2, s3);
                    f_TVD(dd, u, u2, u3, u4, u12, u21, s1, s2, s3);
                    f_TVD(dd, v, v2, v3, v4, v12, v21, s1, s2, s3);
                    f_TVD(dd, w, w2, w3, w4, w12, w21, s1, s2, s3);
                    f_TVD(dd, bx, bx2, bx3, bx4, bx12, bx21, s1, s2, s3);
                    f_TVD(dd, by, by2, by3, by4, by12, by21, s1, s2, s3);
                    f_TVD(dd, bz, bz2, bz3, bz4, bz12, bz21, s1, s2, s3);
                    if (Q12 <= 0.0)
                    {
                        Q12 = Q;
                    }
                    if (Q21 <= 0.0)
                    {
                        Q21 = Q_2;
                    }
                    if (ro12 <= 0.0)
                    {
                        ro12 = ro;
                    }
                    if (p12 <= 0.0)
                    {
                        p12 = p;
                    }
                    if (ro21 <= 0.0)
                    {
                        ro21 = ro2;
                    }
                    if (p21 <= 0.0)
                    {
                        p21 = p2;
                    }


                    if (diver == true)
                    {
                        sks = n1 * (bx12 + bx21) / 2.0 + n2 * (by12 + by21) / 2.0 + n3 * (bz12 + bz21) / 2.0;
                    }
                    else
                    {
                        sks = 0.0;
                    }
                    Potok[8] = Potok[8] + sks * S;

                    if (!kor_Sol || metod == 1)
                    {
                        tmin = min(tmin, HLLDQ_Alexashov(ro12, Q12, p12, u12, v12, w12, bx12, by12, bz12, ro21, Q21, p21, u21, v21, w21, bx21, by21, bz21, P, PQ, n1, n2, n3, dist, metod));
                    }
                    else
                    {
                        tmin = min(tmin, HLLDQ_Korolkov(ro12, Q12, p12, u12, v12, w12, bx12, by12, bz12, ro21, Q21, p21, u21, v21, w21, bx21, by21, bz21, P, PQ, n1, n2, n3, dist, metod));
                    }

                    //tmin = min(tmin, HLLDQ_Alexashov(ro12, Q12, p12, u12, v12, w12, bx12, by12, bz12, ro21, Q21, p21, u21, v21, w21, bx21, by21, bz21, P, PQ, n1, n2, n3, dist, metod));
                    
                    for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                    {
                        Potok[k] = Potok[k] + P[k] * S;
                    }
                    Potok[9] = Potok[9] + PQ * S;
                }
                else
                {
                    double S = get_square(x, y, z, dx, dy, dz, x2, y2, z2, dx2, dy2, dz2, n1, n2, n3, dist);

                    if (diver == true)
                    {
                        sks = n1 * (bx + bx2) / 2.0 + n2 * (by + by2) / 2.0 + n3 * (bz + bz2) / 2.0;
                    }
                    else
                    {
                        sks = 0.0;
                    }
                    Potok[8] = Potok[8] + sks * S;

                    if (!kor_Sol || metod == 1)
                    {
                        tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro2, Q_2, p2, u2, v2, w2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, metod));
                    }
                    else
                    {
                        tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro2, Q_2, p2, u2, v2, w2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, metod));
                    }
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro2, Q_2, p2, u2, v2, w2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, metod));
                    
                    for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                    {
                        Potok[k] = Potok[k] + P[k] * S;
                    }
                    Potok[9] = Potok[9] + PQ * S;
                }

            }
            else if (ii == -1)
            {
            double S = dy * dz * 4.0;
            n1 = 1.0;
            n2 = 0.0;
            n3 = 0.0;
            dist = dx;
            if (diver == true)
            {
                sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;
            }
            else
            {
                sks = 0.0;
            }
            /*double uu = u;
            if (uu < 0.0)
            {
                uu = 0.0;
            }*/
            Potok[8] = Potok[8] + sks * S;
            if (!kor_Sol || metod == 1)
            {
                tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, metod));
            }
            else
            {
                tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, metod));
            }
            //  ћожно вручную выписать потоки дл€ ускорени€ времени
            /*double b2R = kv(bxC) + kv(byC) + kv(bzC);
            double ptR = pC + b2R / 2.0;
            double upt2 = (kv(uC) + kv(vC) + kv(wC)) / 2.0;
            double sbv2 = uC * bxC + vC * byC + wC * bzC;
            double e2 = pC / g1 + roC * upt2 + b2R / 2.0;

            P[0] = roC * uC;
            P[1] = roC * uC * uC + ptR - kv(bxC);
            P[2] = roC * uC * vC - bxC * byC;
            P[3] = roC * uC * wC - bxC * bzC;
            P[7] = (e2 + ptR) * uC - bxC * sbv2;
            P[4] = 0.0;
            P[5] = uC * byC - vC * bxC;
            P[6] = uC * bzC - wC * bxC;*/

            for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
            {
                Potok[k] = Potok[k] + P[k] * S;
            }
            Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -2)
            {
            double S = dy * dz * 4.0;
            n1 = -1.0;
            n2 = 0.0;
            n3 = 0.0;
            dist = dx;
            if (diver == true)
            {
                sks = n1 * bx + n2 * by + n3 * bz;
            }
            else
            {
                sks = 0.0;
            }
            Potok[8] = Potok[8] + sks * S;
            double uu = u;
            if (uu > -M_inf_ && step_ < 0)
            {
                uu = -M_inf_;
            }
            else if (uu > -0.01)
            {
                uu = -0.01;
            }

            if (!kor_Sol || metod == 1)
            {
                tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, uu, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
            }
            else
            {
                tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, uu, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
            }

            /*double t1, t2, t3, m1, m2, m3;
            double bx_L = bx / spi4;
            double by_L = by / spi4;
            double bz_L = bz / spi4;
            t1 = 0.0;
            t2 = 0.0;
            t3 = 1.0;
            m1 = 0.0;
            m2 = 1.0;
            m3 = 0.0;
            double u1 = uu * n1 + v * n2 + w * n3;
            double v1 = uu * t1 + v * t2 + w * t3;
            double w1 = uu * m1 + v * m2 + w * m3;
            double bn1, bt1, bm1;
            bn1 = bx_L * n1 + by_L * n2 + bz_L * n3;
            bt1 = bx_L * t1 + by_L * t2 + bz_L * t3;
            bm1 = bx_L * m1 + by_L * m2 + bz_L * m3;
            double uu_L = (kv(uu) + kv(v) + kv(w)) / 2.0;
            double bb_L = kv(bx_L) + kv(by_L) + kv(bz_L);
            double e1 = p / g1 + ro * uu_L + bb_L / 2.0;
            double pTL = p + bb_L / 2.0;

            double PO[9];

            PO[0] = ro * u1;
            PO[1] = ro * u1 * u1 + pTL - kv(bn1);
            PO[2] = ro * u1 * v1 - bn1 * bt1;
            PO[3] = ro * u1 * w1 - bn1 * bm1;
            PO[4] = (e1 + pTL) * u1 - bn1 * (u1 * bn1 + v1 * bt1 + w1 * bm1);
            PO[5] = 0.0;
            PO[6] = u1 * bt1 - v1 * bn1;
            PO[7] = u1 * bm1 - w1 * bn1;
            PO[8] = Q * u1;


            P[1] = n1 * PO[1] + t1 * PO[2] + m1 * PO[3];
            P[2] = n2 * PO[1] + t2 * PO[2] + m2 * PO[3];
            P[3] = n3 * PO[1] + t3 * PO[2] + m3 * PO[3];
            P[5] = spi4 * (n1 * PO[5] + t1 * PO[6] + m1 * PO[7]);
            P[6] = spi4 * (n2 * PO[5] + t2 * PO[6] + m2 * PO[7]);
            P[7] = spi4 * (n3 * PO[5] + t3 * PO[6] + m3 * PO[7]);
            P[0] = PO[0];
            P[4] = PO[4];
            PQ = PO[8];

            double SWAP = P[4];
            P[4] = P[5];
            P[5] = P[6];
            P[6] = P[7];
            P[7] = SWAP;*/

            for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
            {
                Potok[k] = Potok[k] + P[k] * S;
            }
            Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -3)
            {
            double S = dx * dz * 4.0;
            n1 = 0.0;
            n2 = 1.0;
            n3 = 0.0;
            dist = dy;
            double uu = v;
            if (uu < 0.0)
            {
                uu = 0.0;
            }
            if (diver == true)
            {
                sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;
            }
            else
            {
                sks = 0.0;
            }
            Potok[8] = Potok[8] + sks * S;
            if (!kor_Sol || metod == 1)
            {
                tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, pC, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
            }
            else
            {
                tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, pC, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
            }
            for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
            {
                Potok[k] = Potok[k] + P[k] * S;
            }
            Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -4)
            {
            double S = dx * dz * 4.0;
            n1 = 0.0;
            n2 = -1.0;
            n3 = 0.0;
            dist = dy;
            if (diver == true)
            {
                //sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;
                sks = n2 * (by + by) / 2.0;
            }
            else
            {
                sks = 0.0;
            }

            Potok[8] = Potok[8] + sks * S;
            if (!kor_Sol || metod == 1)
            {
                tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, -v, w, -bx, by, -bz, P, PQ, n1, n2, n3, dist, metod));
                //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, pC, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
            }
            else
            {
                tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, -v, w, -bx, by, -bz, P, PQ, n1, n2, n3, dist, metod));
            }

            for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
            {
                Potok[k] = Potok[k] + P[k] * S;
            }
            Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -5)
            {
            double S = dy * dx * 4.0;
            n1 = 0.0;
            n2 = 0.0;
            n3 = 1.0;
            dist = dz;
            if (diver == true)
            {
                sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;
            }
            else
            {
                sks = 0.0;
            }


            Potok[8] = Potok[8] + sks * S;
            if (!kor_Sol || metod == 1)
            {
                tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, pC, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
            }
            else
            {
                tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, pC, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
            }

            for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
            {
                Potok[k] = Potok[k] + P[k] * S;
            }
            Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -6)
            {
            double S = dy * dx * 4.0;
            n1 = 0.0;
            n2 = 0.0;
            n3 = -1.0;
            dist = dz;
            if (diver == true)
            {
                //sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;
                sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0;
            }
            else
            {
                sks = 0.0;
            }


            Potok[8] = Potok[8] + sks * S;
            if (!kor_Sol || metod == 1)
            {
                tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, -w, bx, by, -bz, P, PQ, n1, n2, n3, dist, metod));
                //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, pC, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, metod));
            }
            else
            {
                tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, -w, bx, by, -bz, P, PQ, n1, n2, n3, dist, metod));
            }
            for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
            {
                Potok[k] = Potok[k] + P[k] * S;
            }
            Potok[9] = Potok[9] + PQ * S;
            }
            else
            {
            printf("Error 12438jdyu. Ne doljni suda popadat = %d \n", ii);
            }
        }

        double ro33, p33, u33, v33, w33, bx33, by33, bz33, Q33;


        Q33 = Q - *T_do * Potok[9] / Volume;
        ro33 = ro - *T_do * Potok[0] / Volume;
        if (ro33 <= 0.0)
        {
            printf("ERROR -  dssdbfhfshjskfutytqqazz\n");
            printf("%lf, %lf, %lf, %lf\n", x, y, z, ro, ro3);
            ro33 = 0.0001;
        }
        u33 = (ro * u - *T_do * (Potok[1] + (bx / cpi4) * Potok[8]) / Volume) / ro33;
        v33 = (ro * v - *T_do * (Potok[2] + (by / cpi4) * Potok[8]) / Volume) / ro33;
        w33 = (ro * w - *T_do * (Potok[3] + (bz / cpi4) * Potok[8]) / Volume) / ro33;
        bx33 = (bx - *T_do * (Potok[4] + u * Potok[8]) / Volume);
        by33 = (by - *T_do * (Potok[5] + v * Potok[8]) / Volume);
        bz33 = (bz - *T_do * (Potok[6] + w * Potok[8]) / Volume);
        p33 = ((U8(ro, p, u, v, w, bx, by, bz) - *T_do * (Potok[7] + (skk(u, v, w, bx, by, bz) / cpi4) * Potok[8])//
            / Volume) - 0.5 * ro33 * kvv(u33, v33, w33) - kvv(bx33, by33, bz33) / cpi8) * (ggg - 1.0);
        //u3 = (ro * u - *T_do * (Potok[1] + (bx) * Potok[8]) / Volume) / ro3;
        //v3 = (ro * v - *T_do * (Potok[2] + (by) * Potok[8]) / Volume) / ro3;
        //w3 = (ro * w - *T_do * (Potok[3] + (bz) * Potok[8]) / Volume) / ro3;
        //bx3 = (bx - *T_do * (Potok[4] + u * Potok[8]) / Volume);
        //by3 = (by - *T_do * (Potok[5] + v * Potok[8]) / Volume);
        //bz3 = (bz - *T_do * (Potok[6] + w * Potok[8]) / Volume);
        //p3 = ((U8(ro, p, u, v, w, bx, by, bz) - *T_do * (Potok[7] + (skk(u, v, w, bx, by, bz)) * Potok[8])//
        //    / Volume) - 0.5 * ro3 * kvv(u3, v3, w3) - kvv(bx3, by3, bz3)) * (ggg - 1.0);
        if (p33 <= 0)
        {
            p33 = 0.000001;
        }

        Q2[index] = Q33;
        RO2[index] = ro33;
        P2[index] = p33;
        U2[index] = u33;
        V2[index] = v33;
        W2[index] = w33;
        BX2[index] = bx33;
        BY2[index] = by33;
        BZ2[index] = bz33;

        if (*T > tmin)
        {
            *T = tmin;
            __threadfence();
        }
    }

}

__global__ void Cuda_main_HLLDQ_TVD3(int* NN, double* X, double* Y, double* Z, double* DX, double* DY, double* DZ,//
    double* RO1, double* RO2, double* Q1, double* Q2, double* P1, double* P2, double* U1, double* U2, double* V1, double* V2,//
    double* W1, double* W2, double* BX1, double* BY1, double* BZ1, double* BX2, double* BY2, double* BZ2,//
    int* SOSED, int* SOSED2, int* SOSED3, int* SOSED22, int* SOSED23, int* SOSED24, int* SOSED32, int* SOSED33, int* SOSED34,
    int* L, int* R, double* T, double* T_do, int step_, double M_inf_, bool mgd = true, bool diver = true, int metod = 0)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; // √лобальный индекс текущего потока
    if (index > *NN - 1)
    {
        return;
    }
    double x, y, z, dx, dy, dz, ro, p, u, v, w, bx, by, bz, Q;
    int l = L[index];
    int r = R[index];
    int my_metod = metod;
    x = X[index];
    y = Y[index];
    z = Z[index];
    dx = DX[index];
    dy = DY[index];
    dz = DZ[index];
    ro = RO1[index];
    p = P1[index];
    u = U1[index];
    v = V1[index];
    w = W1[index];
    Q = Q1[index];
    bx = BX1[index];
    by = BY1[index];
    bz = BZ1[index];




    //double ddd = kv(y) + kv(z);
    double ddd2 = kv(x) + kv(y) + kv(z);

    if (ddd2 <= ddist * ddist) // || (ddd <= 4.0 && x > -5 && x < 0) ) //(ddd < 5.76 || ddd2 <= 2.0) //1.5
    {
        RO2[index] = ro;
        P2[index] = p;
        U2[index] = u;
        V2[index] = v;
        W2[index] = w;
        BX2[index] = bx;
        BY2[index] = by;
        BZ2[index] = bz;
        Q2[index] = Q;
    }
    else
    {
        /*metod = 3;
        if (ddd2 <= 0.73 * 0.73)
        {
            metod = 2;
        }*/

        double n1 = 0.0;
        double n2 = 0.0;
        double n3 = 0.0;
        double dist = 0.0;
        double P[8] = { 0.0 };
        P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;
        double Potok[10] = { 0.0 };
        Potok[0] = Potok[1] = Potok[2] = Potok[3] = Potok[4] = Potok[5] = Potok[6] = Potok[7] = Potok[8] = Potok[9] = 0.0;
        double PQ;
        double tmin = 1000;
        double Volume = dx * dy * dz * 8.0;
        int ii = 0;
        double Q_2, x2, y2, z2, dx2, dy2, dz2, ro2, p2, u2, v2, w2, bx2, by2, bz2, sks;
        double Q_3, x3, y3, z3, dx3, dy3, dz3, ro3, p3, u3, v3, w3, bx3, by3, bz3;
        double Q_4, x4, y4, z4, dx4, dy4, dz4, ro4, p4, u4, v4, w4, bx4, by4, bz4;
        double x12, y12, z12, dx12, dy12, dz12, ro12, p12, u12, v12, w12, bx12, by12, bz12, Q12, Q21;
        double x21, y21, z21, dx21, dy21, dz21, ro21, p21, u21, v21, w21, bx21, by21, bz21;
        double su1, sv1, sw1, su2, sv2, sw2, sro1, sro2, sp1, sp2;
        double ur, up, uz;
        double roC = 1.0; // 8.2598; //  1.0;
        double pC = 1.0 / (ggg); // 1.0 / (ggg * M_inf * M_inf);
        double uC = M_infty; // -1.0;
        double vC = 0.0;
        double wC = 0.0;
        double QC = 100.0;
        double bxC, byC, bzC;
        bxC = 0.0;
        byC = 0.0;
        bzC = B_inf;

        int kk, kk2, l2, r2;
        for (int i = l; i <= r; i++)
        {
            my_metod = metod;
            ii = SOSED[i];
            if (ii >= 0)
            {
                x2 = X[ii];
                y2 = Y[ii];
                z2 = Z[ii];
                dx2 = DX[ii];
                dy2 = DY[ii];
                dz2 = DZ[ii];
                ro2 = RO1[ii];
                p2 = P1[ii];
                u2 = U1[ii];
                v2 = V1[ii];
                w2 = W1[ii];
                Q_2 = Q1[ii];
                bx2 = BX1[ii];
                by2 = BY1[ii];
                bz2 = BZ1[ii];

                double ddd3 = kv((z + z2) / 2.0) + kv((x + x2) / 2.0) + kv((y + y2) / 2.0);

                kk = SOSED2[i];
                if (kk != -1 && ddd3 > (ddist2 * ddist2)) //&& ddd2 > 0.8
                {
                    dx3 = DX[kk];
                    x3 = X[kk];
                    y3 = Y[kk];
                    z3 = Z[kk];
                    ro3 = RO1[kk];
                    p3 = P1[kk];
                    u3 = U1[kk];
                    v3 = V1[kk];
                    w3 = W1[kk];
                    Q_3 = Q1[kk];
                    bx3 = BX1[kk];
                    by3 = BY1[kk];
                    bz3 = BZ1[kk];

                    int kk2 = SOSED22[i];
                    if (kk2 != -1)
                    {
                        // ¬ этом случае у €чейки 4 соседа с этой стороны и нужно сносить средние значени€
                        ro3 = (ro3 + RO1[kk2] + RO1[SOSED23[i]] + RO1[SOSED24[i]]) / 4.0;
                        p3 = (p3 + P1[kk2] + P1[SOSED23[i]] + P1[SOSED24[i]]) / 4.0;
                        u3 = (u3 + U1[kk2] + U1[SOSED23[i]] + U1[SOSED24[i]]) / 4.0;
                        v3 = (v3 + V1[kk2] + V1[SOSED23[i]] + V1[SOSED24[i]]) / 4.0;
                        w3 = (w3 + W1[kk2] + W1[SOSED23[i]] + W1[SOSED24[i]]) / 4.0;
                        Q_3 = (Q_3 + Q1[kk2] + Q1[SOSED23[i]] + Q1[SOSED24[i]]) / 4.0;
                        bx3 = (bx3 + BX1[kk2] + BX1[SOSED23[i]] + BX1[SOSED24[i]]) / 4.0;
                        by3 = (by3 + BY1[kk2] + BY1[SOSED23[i]] + BY1[SOSED24[i]]) / 4.0;
                        bz3 = (bz3 + BZ1[kk2] + BZ1[SOSED23[i]] + BZ1[SOSED24[i]]) / 4.0;
                    }

                    kk2 = SOSED3[i];

                    dx4 = DX[kk2];
                    x4 = X[kk2];
                    y4 = Y[kk2];
                    z4 = Z[kk2];
                    ro4 = RO1[kk2];
                    p4 = P1[kk2];
                    u4 = U1[kk2];
                    v4 = V1[kk2];
                    w4 = W1[kk2];
                    Q_4 = Q1[kk2];
                    bx4 = BX1[kk2];
                    by4 = BY1[kk2];
                    bz4 = BZ1[kk2];

                    kk2 = SOSED32[i];
                    if (kk2 != -1)
                    {
                        // ¬ этом случае у €чейки 4 соседа с этой стороны и нужно сносить средние значени€
                        ro4 = (ro4 + RO1[kk2] + RO1[SOSED33[i]] + RO1[SOSED34[i]]) / 4.0;
                        p4 = (p4 + P1[kk2] + P1[SOSED33[i]] + P1[SOSED34[i]]) / 4.0;
                        u4 = (u4 + U1[kk2] + U1[SOSED33[i]] + U1[SOSED34[i]]) / 4.0;
                        v4 = (v4 + V1[kk2] + V1[SOSED33[i]] + V1[SOSED34[i]]) / 4.0;
                        w4 = (w4 + W1[kk2] + W1[SOSED33[i]] + W1[SOSED34[i]]) / 4.0;
                        Q_4 = (Q_4 + Q1[kk2] + Q1[SOSED33[i]] + Q1[SOSED34[i]]) / 4.0;
                        bx4 = (bx4 + BX1[kk2] + BX1[SOSED33[i]] + BX1[SOSED34[i]]) / 4.0;
                        by4 = (by4 + BY1[kk2] + BY1[SOSED33[i]] + BY1[SOSED34[i]]) / 4.0;
                        bz4 = (bz4 + BZ1[kk2] + BZ1[SOSED33[i]] + BZ1[SOSED34[i]]) / 4.0;
                    }


                    double S = get_square(x, y, z, dx, dy, dz, x2, y2, z2, dx2, dy2, dz2, n1, n2, n3, dist);
                    double dd = 0.0;
                    if (n1 != 0)
                    {
                        dd = dx;
                    }
                    else if (n2 != 0)
                    {
                        dd = dy;
                    }
                    else if (n3 != 0)
                    {
                        dd = dz;
                    }
                    else
                    {
                        printf("Errrrrr 2323132214243\n");
                    }

                    double s1 = __dsqrt_rn(kv(x - x3) + kv(y - y3) + kv(z - z3));
                    double s2 = __dsqrt_rn(kv(x - x2) + kv(y - y2) + kv(z - z2));
                    double s3 = __dsqrt_rn(kv(x4 - x2) + kv(y4 - y2) + kv(z4 - z2));
                    // p3, p, p2, p4
                    f_TVD(dd, Q, Q_2, Q_3, Q_4, Q12, Q21, s1, s2, s3);
                    f_TVD(dd, p, p2, p3, p4, p12, p21, s1, s2, s3);
                    f_TVD(dd, ro, ro2, ro3, ro4, ro12, ro21, s1, s2, s3);
                    f_TVD(dd, u, u2, u3, u4, u12, u21, s1, s2, s3);
                    f_TVD(dd, v, v2, v3, v4, v12, v21, s1, s2, s3);
                    f_TVD(dd, w, w2, w3, w4, w12, w21, s1, s2, s3);
                    f_TVD(dd, bx, bx2, bx3, bx4, bx12, bx21, s1, s2, s3);
                    f_TVD(dd, by, by2, by3, by4, by12, by21, s1, s2, s3);
                    f_TVD(dd, bz, bz2, bz3, bz4, bz12, bz21, s1, s2, s3);
                    if (Q12 <= 0.0)
                    {
                        Q12 = Q;
                    }
                    if (Q21 <= 0.0)
                    {
                        Q21 = Q_2;
                    }
                    if (ro12 <= 0.0)
                    {
                        ro12 = ro;
                    }
                    if (p12 <= 0.0)
                    {
                        p12 = p;
                    }
                    if (ro21 <= 0.0)
                    {
                        ro21 = ro2;
                    }
                    if (p21 <= 0.0)
                    {
                        p21 = p2;
                    }

                    //if (kvv(u, v, w) / (ggg * p / ro) > 100.0 || kvv(u2, v2, w2) / (ggg * p2 / ro2) > 100.0 || 
                    //    kvv(u3, v3, w3) / (ggg * p3 / ro3) > 100.0 || kvv(u4, v4, w4) / (ggg * p4 / ro4) > 100.0)
                    if (false)
                    {
                        my_metod = 1;

                        ro12 = ro;
                        p12 = p;
                        Q12 = Q;
                        u12 = u;
                        v12 = v;
                        w12 = w;
                        bx12 = bx;
                        by12 = by;
                        bz12 = bz;

                        ro21 = ro2;
                        p21 = p2;
                        Q21 = Q_2;
                        u21 = u2;
                        v21 = v2;
                        w21 = w2;
                        bx21 = bx2;
                        by21 = by2;
                        bz21 = bz2;
                    }


                    //if(ddd2 < 0.8) my_metod = 1;


                    sks = n1 * (bx12 + bx21) / 2.0 + n2 * (by12 + by21) / 2.0 + n3 * (bz12 + bz21) / 2.0;

                    Potok[8] = Potok[8] + sks * S;



                    if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                    {
                        tmin = min(tmin, HLLDQ_Alexashov(ro12, Q12, p12, u12, v12, w12, bx12, by12, bz12, ro21, Q21, p21, u21, v21, w21, bx21, by21, bz21, P, PQ, n1, n2, n3, dist, my_metod));
                    }
                    else
                    {
                        tmin = min(tmin, HLLDQ_Korolkov(ro12, Q12, p12, u12, v12, w12, bx12, by12, bz12, ro21, Q21, p21, u21, v21, w21, bx21, by21, bz21, P, PQ, n1, n2, n3, dist, my_metod));
                    }


                    for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                    {
                        Potok[k] = Potok[k] + P[k] * S;
                    }
                    Potok[9] = Potok[9] + PQ * S;
                }
                else
                {
                    double S = get_square(x, y, z, dx, dy, dz, x2, y2, z2, dx2, dy2, dz2, n1, n2, n3, dist);

                    sks = n1 * (bx + bx2) / 2.0 + n2 * (by + by2) / 2.0 + n3 * (bz + bz2) / 2.0;
                    Potok[8] = Potok[8] + sks * S;

                    su1 = u;
                    sv1 = v;
                    sw1 = w;

                    su2 = u2;
                    sv2 = v2;
                    sw2 = w2;

                    sro1 = ro;
                    sro2 = ro2;

                    sp1 = p;
                    sp2 = p2;

                    // ƒелаем перенос в сферической —  
                    if (ddd3 <= (ddist2 * ddist2))
                        //if (kvv(u, v, w) / (ggg * p / ro) > 100.0 && kvv(u2, v2, w2) / (ggg * p2 / ro2) > 100.0)
                    {
                        //my_metod = 1;

                        spherical_skorost(z, x, y, w, u, v, ur, up, uz);
                        dekard_skorost((z + z2) / 2.0, (x + x2) / 2.0, (y + y2) / 2.0, ur, up, uz, sw1, su1, sv1);

                        spherical_skorost(z2, x2, y2, w2, u2, v2, ur, up, uz);
                        dekard_skorost((z + z2) / 2.0, (x + x2) / 2.0, (y + y2) / 2.0, ur, up, uz, sw2, su2, sv2);

                        sro1 = ro * (kv(z) + kv(x) + kv(y)) / ddd3;
                        sro2 = ro2 * (kv(z2) + kv(x2) + kv(y2)) / ddd3;

                        sp1 = p * pow((kv(z) + kv(x) + kv(y)) / ddd3, ggg);
                        sp2 = p2 * pow((kv(z2) + kv(x2) + kv(y2)) / ddd3, ggg);
                    }



                    /*if (!kor_Sol || metod == 1 || metod == 3)
                    {
                        tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro2, Q_2, p2, u2, v2, w2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, metod));
                    }
                    else
                    {
                        tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro2, Q_2, p2, u2, v2, w2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, metod));
                    }*/

                    if (my_metod <= 1 || my_metod == 3)//(y * y + z * z < 225 && y2 * y2 + z2 * z2 < 225 && x > -15 && x2 > -15 && x < 8 && x2 < 8  && step_ > 10000)
                    {
                        tmin = min(tmin, HLLDQ_Alexashov(sro1, Q, sp1, su1, sv1, sw1, bx, by, bz, sro2, Q_2, sp2, su2, sv2, sw2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, my_metod));
                    }
                    else
                    {
                        tmin = min(tmin, HLLDQ_Korolkov(sro1, Q, sp1, su1, sv1, sw1, bx, by, bz, sro2, Q_2, sp2, su2, sv2, sw2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, my_metod));
                    }


                    for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                    {
                        Potok[k] = Potok[k] + P[k] * S;
                    }
                    Potok[9] = Potok[9] + PQ * S;
                }

            }
            else if (ii == -1)
            {
                double S = dy * dz * 4.0;
                n1 = 1.0;
                n2 = 0.0;
                n3 = 0.0;
                dist = dx;
                sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;

                /*double uu = u;
                if (uu < 0.0)
                {
                    uu = 0.0;
                }*/
                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, my_metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, my_metod));
                }

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -2)
            {
                double S = dy * dz * 4.0;
                n1 = -1.0;
                n2 = 0.0;
                n3 = 0.0;
                dist = dx;
                sks = n1 * bx + n2 * by + n3 * bz;

                Potok[8] = Potok[8] + sks * S;
                double uu = u;
                if (uu > M_infty / 3.0)
                {
                    uu = M_infty;
                }
                /*else if (uu > -0.01)
                {
                    uu = -0.01;
                }*/

                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, uu, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, uu, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }

                /*double t1, t2, t3, m1, m2, m3;
                double bx_L = bx / spi4;
                double by_L = by / spi4;
                double bz_L = bz / spi4;
                t1 = 0.0;
                t2 = 0.0;
                t3 = 1.0;
                m1 = 0.0;
                m2 = 1.0;
                m3 = 0.0;
                double u1 = uu * n1 + v * n2 + w * n3;
                double v1 = uu * t1 + v * t2 + w * t3;
                double w1 = uu * m1 + v * m2 + w * m3;
                double bn1, bt1, bm1;
                bn1 = bx_L * n1 + by_L * n2 + bz_L * n3;
                bt1 = bx_L * t1 + by_L * t2 + bz_L * t3;
                bm1 = bx_L * m1 + by_L * m2 + bz_L * m3;
                double uu_L = (kv(uu) + kv(v) + kv(w)) / 2.0;
                double bb_L = kv(bx_L) + kv(by_L) + kv(bz_L);
                double e1 = p / g1 + ro * uu_L + bb_L / 2.0;
                double pTL = p + bb_L / 2.0;

                double PO[9];

                PO[0] = ro * u1;
                PO[1] = ro * u1 * u1 + pTL - kv(bn1);
                PO[2] = ro * u1 * v1 - bn1 * bt1;
                PO[3] = ro * u1 * w1 - bn1 * bm1;
                PO[4] = (e1 + pTL) * u1 - bn1 * (u1 * bn1 + v1 * bt1 + w1 * bm1);
                PO[5] = 0.0;
                PO[6] = u1 * bt1 - v1 * bn1;
                PO[7] = u1 * bm1 - w1 * bn1;
                PO[8] = Q * u1;


                P[1] = n1 * PO[1] + t1 * PO[2] + m1 * PO[3];
                P[2] = n2 * PO[1] + t2 * PO[2] + m2 * PO[3];
                P[3] = n3 * PO[1] + t3 * PO[2] + m3 * PO[3];
                P[5] = spi4 * (n1 * PO[5] + t1 * PO[6] + m1 * PO[7]);
                P[6] = spi4 * (n2 * PO[5] + t2 * PO[6] + m2 * PO[7]);
                P[7] = spi4 * (n3 * PO[5] + t3 * PO[6] + m3 * PO[7]);
                P[0] = PO[0];
                P[4] = PO[4];
                PQ = PO[8];

                double SWAP = P[4];
                P[4] = P[5];
                P[5] = P[6];
                P[6] = P[7];
                P[7] = SWAP;*/

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -3)
            {
                double S = dx * dz * 4.0;
                n1 = 0.0;
                n2 = 1.0;
                n3 = 0.0;
                dist = dy;

                sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;

                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -4)
            {
                double S = dx * dz * 4.0;
                n1 = 0.0;
                n2 = -1.0;
                n3 = 0.0;
                dist = dy;
                sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;

                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, -v, w, -bx, by, -bz, P, PQ, n1, n2, n3, dist, my_metod));
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, pC, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));

                }
                else
                {
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, -v, w, -bx, by, -bz, P, PQ, n1, n2, n3, dist, my_metod));
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));

                }

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -5)
            {
                double S = dy * dx * 4.0;
                n1 = 0.0;
                n2 = 0.0;
                n3 = 1.0;
                dist = dz;
                sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;


                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -6)
            {
                double S = dy * dx * 4.0;
                n1 = 0.0;
                n2 = 0.0;
                n3 = -1.0;
                dist = dz;
                sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;


                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, -w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                else
                {
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, -w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else
            {
                printf("Error 12438jdyu. Ne doljni suda popadat = %d \n", ii);
            }
        }


        double q2_1 = 0.0, q2_2 = 0.0, q2_3 = 0.0, q3 = 0.0;

        if (false)//(istoch == true)
        {
            double u_H4 = M_infty, v_H4 = 0.0, w_H4 = 0.0, ro_H4 = 1.0, p_H4 = 1.0 / (2.0 * ggg);

            double U_M_H4 = sqrt(kv(u - u_H4) + kv(v - v_H4) + kv(w - w_H4) + (64.0 / (9.0 * pi)) //
                * (p / ro + 2.0 * p_H4 / ro_H4));

            double U_H4 = sqrt(kv(u - u_H4) + kv(v - v_H4) + kv(w - w_H4) + (4.0 / pi) //
                * (p / ro + 2.0 * p_H4 / ro_H4));

            double sigma_H4 = kv(1.0 - a_2 * log(U_M_H4));

            double nu_H4 = ro * ro_H4 * U_M_H4 * sigma_H4;

            q2_1 = (1.5 / Kn_) * (nu_H4 * (u_H4 - u));
            q2_2 = (1.5 / Kn_) * (nu_H4 * (v_H4 - v));
            q2_3 = (1.5 / Kn_) * (nu_H4 * (w_H4 - w));


            q3 = (1.5 / Kn_) * (nu_H4 * ((kv(u_H4) + kv(v_H4) + kv(w_H4) - kv(u) - kv(v) - kv(w)) / 2.0 + //
                (U_H4 / U_M_H4) * (2.0 * p_H4 / ro_H4 - p / ro)));
        }


        double ro33, p33, u33, v33, w33, bx33, by33, bz33, Q33;


        Q33 = Q - *T_do * Potok[9] / Volume;
        ro33 = ro - *T_do * Potok[0] / Volume;
        if (ro33 <= 0.0)
        {
            printf("ERROR -  dssdbfhfshjskfutytqqazz\n");
            printf("%lf, %lf, %lf, %lf\n", x, y, z, ro33);
            ro33 = ro;
        }
        u33 = (ro * u - *T_do * (Potok[1] + (bx / cpi4) * Potok[8] - q2_1 * Volume) / Volume) / ro33;
        v33 = (ro * v - *T_do * (Potok[2] + (by / cpi4) * Potok[8] - q2_2 * Volume) / Volume) / ro33;
        w33 = (ro * w - *T_do * (Potok[3] + (bz / cpi4) * Potok[8] - q2_3 * Volume) / Volume) / ro33;
        bx33 = (bx - *T_do * (Potok[4] + u * Potok[8]) / Volume);
        by33 = (by - *T_do * (Potok[5] + v * Potok[8]) / Volume);
        bz33 = (bz - *T_do * (Potok[6] + w * Potok[8]) / Volume);
        p33 = ((U8(ro, p, u, v, w, bx, by, bz) - *T_do * (Potok[7] + (skk(u, v, w, bx, by, bz) / cpi4) * Potok[8] - q3 * Volume)//
            / Volume) - 0.5 * ro33 * kvv(u33, v33, w33) - kvv(bx33, by33, bz33) / cpi8) * (ggg - 1.0);
        //u3 = (ro * u - *T_do * (Potok[1] + (bx) * Potok[8]) / Volume) / ro3;
        //v3 = (ro * v - *T_do * (Potok[2] + (by) * Potok[8]) / Volume) / ro3;
        //w3 = (ro * w - *T_do * (Potok[3] + (bz) * Potok[8]) / Volume) / ro3;
        //bx3 = (bx - *T_do * (Potok[4] + u * Potok[8]) / Volume);
        //by3 = (by - *T_do * (Potok[5] + v * Potok[8]) / Volume);
        //bz3 = (bz - *T_do * (Potok[6] + w * Potok[8]) / Volume);
        //p3 = ((U8(ro, p, u, v, w, bx, by, bz) - *T_do * (Potok[7] + (skk(u, v, w, bx, by, bz)) * Potok[8])//
        //    / Volume) - 0.5 * ro3 * kvv(u3, v3, w3) - kvv(bx3, by3, bz3)) * (ggg - 1.0);
        if (p33 <= 0)
        {
            p33 = 0.000001;
        }

        Q2[index] = Q33;
        RO2[index] = ro33;
        P2[index] = p33;
        U2[index] = u33;
        V2[index] = v33;
        W2[index] = w33;
        BX2[index] = bx33;
        BY2[index] = by33;
        BZ2[index] = bz33;

        if (*T > tmin)
        {
            *T = tmin;
            __threadfence();
        }
    }

}

__global__ void Cuda_main_HLLDQ_TVD3_symmetry(int* NN, double* X, double* Y, double* Z, double* DX, double* DY, double* DZ,//
    double* RO1, double* RO2, double* Q1, double* Q2, double* P1, double* P2, double* U1, double* U2, double* V1, double* V2,//
    double* W1, double* W2, double* BX1, double* BY1, double* BZ1, double* BX2, double* BY2, double* BZ2,//
    int* SOSED, int* SOSED2, int* SOSED3, int* SOSED22, int* SOSED23, int* SOSED24, int* SOSED32, int* SOSED33, int* SOSED34,
    int* L, int* R, double* T, double* T_do, int step_, double M_inf_, bool mgd = true, bool diver = true, int metod = 0)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; // √лобальный индекс текущего потока
    if (index > *NN - 1)
    {
        return;
    }
    double x, y, z, dx, dy, dz, ro, p, u, v, w, bx, by, bz, Q;
    int l = L[index];
    int r = R[index];
    int my_metod = metod;
    x = X[index];
    y = Y[index];
    z = Z[index];
    dx = DX[index];
    dy = DY[index];
    dz = DZ[index];
    ro = RO1[index];
    p = P1[index];
    u = U1[index];
    v = V1[index];
    w = W1[index];
    Q = Q1[index];
    bx = BX1[index];
    by = BY1[index];
    bz = BZ1[index];




    //double ddd = kv(y) + kv(z);
    double ddd2 = kv(x) + kv(y) + kv(z);

    if (ddd2 <= ddist * ddist) // || (ddd <= 4.0 && x > -5 && x < 0) ) //(ddd < 5.76 || ddd2 <= 2.0) //1.5
    {
        RO2[index] = ro;
        P2[index] = p;
        U2[index] = u;
        V2[index] = v;
        W2[index] = w;
        BX2[index] = bx;
        BY2[index] = by;
        BZ2[index] = bz;
        Q2[index] = Q;
    }
    else
    {
        /*metod = 3;
        if (ddd2 <= 0.73 * 0.73)
        {
            metod = 2;
        }*/

        double n1 = 0.0;
        double n2 = 0.0;
        double n3 = 0.0;
        double dist = 0.0;
        double P[8] = { 0.0 };
        P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;
        double Potok[10] = { 0.0 };
        Potok[0] = Potok[1] = Potok[2] = Potok[3] = Potok[4] = Potok[5] = Potok[6] = Potok[7] = Potok[8] = Potok[9] = 0.0;
        double PQ;
        double tmin = 1000;
        double Volume = dx * dy * dz * 8.0;
        int ii = 0;
        double Q_2, x2, y2, z2, dx2, dy2, dz2, ro2, p2, u2, v2, w2, bx2, by2, bz2, sks;
        double Q_3, x3, y3, z3, dx3, dy3, dz3, ro3, p3, u3, v3, w3, bx3, by3, bz3;
        double Q_4, x4, y4, z4, dx4, dy4, dz4, ro4, p4, u4, v4, w4, bx4, by4, bz4;
        double x12, y12, z12, dx12, dy12, dz12, ro12, p12, u12, v12, w12, bx12, by12, bz12, Q12, Q21;
        double x21, y21, z21, dx21, dy21, dz21, ro21, p21, u21, v21, w21, bx21, by21, bz21;
        double su1, sv1, sw1, su2, sv2, sw2, sro1, sro2, sp1, sp2;
        double ur, up, uz;
        double roC = 1.0; // 8.2598; //  1.0;
        double pC = 1.0 / (ggg); // 1.0 / (ggg * M_inf * M_inf);
        double uC = M_infty; // -1.0;
        double vC = 0.0;
        double wC = 0.0;
        double QC = 100.0;
        double bxC, byC, bzC;
        bxC = 0.0;
        byC = 0.0;
        bzC = B_inf;

        int kk, kk2, l2, r2;
        for (int i = l; i <= r; i++)
        {
            my_metod = metod;
            ii = SOSED[i];
            if (ii >= 0)
            {
                x2 = X[ii];
                y2 = Y[ii];
                z2 = Z[ii];
                dx2 = DX[ii];
                dy2 = DY[ii];
                dz2 = DZ[ii];
                ro2 = RO1[ii];
                p2 = P1[ii];
                u2 = U1[ii];
                v2 = V1[ii];
                w2 = W1[ii];
                Q_2 = Q1[ii];
                bx2 = BX1[ii];
                by2 = BY1[ii];
                bz2 = BZ1[ii];

                double ddd3 = kv((z + z2) / 2.0) + kv((x + x2) / 2.0) + kv((y + y2) / 2.0);

                kk = SOSED2[i];
                if (kk != -1 && ddd3 > (ddist2 * ddist2)) //&& ddd2 > 0.8
                {
                    if (kk != -6)
                    {
                        //dx3 = DX[kk];
                        x3 = X[kk];
                        y3 = Y[kk];
                        z3 = Z[kk];
                        ro3 = RO1[kk];
                        p3 = P1[kk];
                        u3 = U1[kk];
                        v3 = V1[kk];
                        w3 = W1[kk];
                        Q_3 = Q1[kk];
                        bx3 = BX1[kk];
                        by3 = BY1[kk];
                        bz3 = BZ1[kk];
                    }
                    else if(kk == -6)
                    {
                        //dx3 = dx;
                        x3 = x;
                        y3 = y;
                        z3 = -z;
                        ro3 = ro;
                        p3 = p;
                        u3 = u;
                        v3 = v;
                        w3 = -w;
                        Q_3 = Q;
                        bx3 = bx;
                        by3 = by;
                        bz3 = bz;
                    }
                    else
                    {
                        printf("Errrrrr drgedrgertfertge4rfewrfw: %d\n", kk);
                    }

                    int kk2 = SOSED22[i];
                    if (kk2 != -1)
                    {
                        // ¬ этом случае у €чейки 4 соседа с этой стороны и нужно сносить средние значени€
                        ro3 = (ro3 + RO1[kk2] + RO1[SOSED23[i]] + RO1[SOSED24[i]]) / 4.0;
                        p3 = (p3 + P1[kk2] + P1[SOSED23[i]] + P1[SOSED24[i]]) / 4.0;
                        u3 = (u3 + U1[kk2] + U1[SOSED23[i]] + U1[SOSED24[i]]) / 4.0;
                        v3 = (v3 + V1[kk2] + V1[SOSED23[i]] + V1[SOSED24[i]]) / 4.0;
                        w3 = (w3 + W1[kk2] + W1[SOSED23[i]] + W1[SOSED24[i]]) / 4.0;
                        Q_3 = (Q_3 + Q1[kk2] + Q1[SOSED23[i]] + Q1[SOSED24[i]]) / 4.0;
                        bx3 = (bx3 + BX1[kk2] + BX1[SOSED23[i]] + BX1[SOSED24[i]]) / 4.0;
                        by3 = (by3 + BY1[kk2] + BY1[SOSED23[i]] + BY1[SOSED24[i]]) / 4.0;
                        bz3 = (bz3 + BZ1[kk2] + BZ1[SOSED23[i]] + BZ1[SOSED24[i]]) / 4.0;
                    }

                    kk2 = SOSED3[i];

                    if (kk2 != -6)
                    {
                        //dx4 = DX[kk2];
                        x4 = X[kk2];
                        y4 = Y[kk2];
                        z4 = Z[kk2];
                        ro4 = RO1[kk2];
                        p4 = P1[kk2];
                        u4 = U1[kk2];
                        v4 = V1[kk2];
                        w4 = W1[kk2];
                        Q_4 = Q1[kk2];
                        bx4 = BX1[kk2];
                        by4 = BY1[kk2];
                        bz4 = BZ1[kk2];
                    }
                    else
                    {
                        //dx4 = DX[kk2];
                        x4 = x2;
                        y4 = y2;
                        z4 = -z2;
                        ro4 = ro2;
                        p4 = p2;
                        u4 = u2;
                        v4 = v2;
                        w4 = -w2;
                        Q_4 = Q_2;
                        bx4 = bx2;
                        by4 = by2;
                        bz4 = bz2;
                        //printf("TUT: %d\n", kk2);
                    }

                    kk2 = SOSED32[i];
                    if (kk2 != -1)
                    {
                        // ¬ этом случае у €чейки 4 соседа с этой стороны и нужно сносить средние значени€
                        ro4 = (ro4 + RO1[kk2] + RO1[SOSED33[i]] + RO1[SOSED34[i]]) / 4.0;
                        p4 = (p4 + P1[kk2] + P1[SOSED33[i]] + P1[SOSED34[i]]) / 4.0;
                        u4 = (u4 + U1[kk2] + U1[SOSED33[i]] + U1[SOSED34[i]]) / 4.0;
                        v4 = (v4 + V1[kk2] + V1[SOSED33[i]] + V1[SOSED34[i]]) / 4.0;
                        w4 = (w4 + W1[kk2] + W1[SOSED33[i]] + W1[SOSED34[i]]) / 4.0;
                        Q_4 = (Q_4 + Q1[kk2] + Q1[SOSED33[i]] + Q1[SOSED34[i]]) / 4.0;
                        bx4 = (bx4 + BX1[kk2] + BX1[SOSED33[i]] + BX1[SOSED34[i]]) / 4.0;
                        by4 = (by4 + BY1[kk2] + BY1[SOSED33[i]] + BY1[SOSED34[i]]) / 4.0;
                        bz4 = (bz4 + BZ1[kk2] + BZ1[SOSED33[i]] + BZ1[SOSED34[i]]) / 4.0;
                    }


                    double S = get_square(x, y, z, dx, dy, dz, x2, y2, z2, dx2, dy2, dz2, n1, n2, n3, dist);
                    double dd = 0.0;
                    if (n1 != 0)
                    {
                        dd = dx;
                    }
                    else if (n2 != 0)
                    {
                        dd = dy;
                    }
                    else if (n3 != 0)
                    {
                        dd = dz;
                    }
                    else
                    {
                        printf("Errrrrr 2323132214243\n");
                    }

                    double s1 = __dsqrt_rn(kv(x - x3) + kv(y - y3) + kv(z - z3));
                    double s2 = __dsqrt_rn(kv(x - x2) + kv(y - y2) + kv(z - z2));
                    double s3 = __dsqrt_rn(kv(x4 - x2) + kv(y4 - y2) + kv(z4 - z2));
                    // p3, p, p2, p4
                    f_TVD(dd, Q, Q_2, Q_3, Q_4, Q12, Q21, s1, s2, s3);
                    f_TVD(dd, p, p2, p3, p4, p12, p21, s1, s2, s3);
                    f_TVD(dd, ro, ro2, ro3, ro4, ro12, ro21, s1, s2, s3);
                    f_TVD(dd, u, u2, u3, u4, u12, u21, s1, s2, s3);
                    f_TVD(dd, v, v2, v3, v4, v12, v21, s1, s2, s3);
                    f_TVD(dd, w, w2, w3, w4, w12, w21, s1, s2, s3);
                    f_TVD(dd, bx, bx2, bx3, bx4, bx12, bx21, s1, s2, s3);
                    f_TVD(dd, by, by2, by3, by4, by12, by21, s1, s2, s3);
                    f_TVD(dd, bz, bz2, bz3, bz4, bz12, bz21, s1, s2, s3);
                    if (Q12 <= 0.0)
                    {
                        Q12 = Q;
                    }
                    if (Q21 <= 0.0)
                    {
                        Q21 = Q_2;
                    }
                    if (ro12 <= 0.0)
                    {
                        ro12 = ro;
                    }
                    if (p12 <= 0.0)
                    {
                        p12 = p;
                    }
                    if (ro21 <= 0.0)
                    {
                        ro21 = ro2;
                    }
                    if (p21 <= 0.0)
                    {
                        p21 = p2;
                    }

                    //if (kvv(u, v, w) / (ggg * p / ro) > 100.0 || kvv(u2, v2, w2) / (ggg * p2 / ro2) > 100.0 || 
                    //    kvv(u3, v3, w3) / (ggg * p3 / ro3) > 100.0 || kvv(u4, v4, w4) / (ggg * p4 / ro4) > 100.0)
                    if (false)
                    {
                        my_metod = 1;

                        ro12 = ro;
                        p12 = p;
                        Q12 = Q;
                        u12 = u;
                        v12 = v;
                        w12 = w;
                        bx12 = bx;
                        by12 = by;
                        bz12 = bz;

                        ro21 = ro2;
                        p21 = p2;
                        Q21 = Q_2;
                        u21 = u2;
                        v21 = v2;
                        w21 = w2;
                        bx21 = bx2;
                        by21 = by2;
                        bz21 = bz2;
                    }


                    //if(ddd2 < 0.8) my_metod = 1;


                    sks = n1 * (bx12 + bx21) / 2.0 + n2 * (by12 + by21) / 2.0 + n3 * (bz12 + bz21) / 2.0;

                    Potok[8] = Potok[8] + sks * S;



                    if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                    {
                        tmin = min(tmin, HLLDQ_Alexashov(ro12, Q12, p12, u12, v12, w12, bx12, by12, bz12, ro21, Q21, p21, u21, v21, w21, bx21, by21, bz21, P, PQ, n1, n2, n3, dist, my_metod));
                    }
                    else
                    {
                        tmin = min(tmin, HLLDQ_Korolkov(ro12, Q12, p12, u12, v12, w12, bx12, by12, bz12, ro21, Q21, p21, u21, v21, w21, bx21, by21, bz21, P, PQ, n1, n2, n3, dist, my_metod));
                    }


                    for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                    {
                        Potok[k] = Potok[k] + P[k] * S;
                    }
                    Potok[9] = Potok[9] + PQ * S;
                }
                else
                {
                    double S = get_square(x, y, z, dx, dy, dz, x2, y2, z2, dx2, dy2, dz2, n1, n2, n3, dist);

                    sks = n1 * (bx + bx2) / 2.0 + n2 * (by + by2) / 2.0 + n3 * (bz + bz2) / 2.0;
                    Potok[8] = Potok[8] + sks * S;

                    su1 = u;
                    sv1 = v;
                    sw1 = w;

                    su2 = u2;
                    sv2 = v2;
                    sw2 = w2;

                    sro1 = ro;
                    sro2 = ro2;

                    sp1 = p;
                    sp2 = p2;

                    // ƒелаем перенос в сферической —  
                    if (ddd3 <= (ddist2 * ddist2))
                        //if (kvv(u, v, w) / (ggg * p / ro) > 100.0 && kvv(u2, v2, w2) / (ggg * p2 / ro2) > 100.0)
                    {
                        //my_metod = 1;

                        spherical_skorost(z, x, y, w, u, v, ur, up, uz);
                        dekard_skorost((z + z2) / 2.0, (x + x2) / 2.0, (y + y2) / 2.0, ur, up, uz, sw1, su1, sv1);

                        spherical_skorost(z2, x2, y2, w2, u2, v2, ur, up, uz);
                        dekard_skorost((z + z2) / 2.0, (x + x2) / 2.0, (y + y2) / 2.0, ur, up, uz, sw2, su2, sv2);

                        sro1 = ro * (kv(z) + kv(x) + kv(y)) / ddd3;
                        sro2 = ro2 * (kv(z2) + kv(x2) + kv(y2)) / ddd3;

                        sp1 = p * pow((kv(z) + kv(x) + kv(y)) / ddd3, ggg);
                        sp2 = p2 * pow((kv(z2) + kv(x2) + kv(y2)) / ddd3, ggg);
                    }



                    /*if (!kor_Sol || metod == 1 || metod == 3)
                    {
                        tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro2, Q_2, p2, u2, v2, w2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, metod));
                    }
                    else
                    {
                        tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro2, Q_2, p2, u2, v2, w2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, metod));
                    }*/

                    if (my_metod <= 1 || my_metod == 3)//(y * y + z * z < 225 && y2 * y2 + z2 * z2 < 225 && x > -15 && x2 > -15 && x < 8 && x2 < 8  && step_ > 10000)
                    {
                        tmin = min(tmin, HLLDQ_Alexashov(sro1, Q, sp1, su1, sv1, sw1, bx, by, bz, sro2, Q_2, sp2, su2, sv2, sw2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, my_metod));
                    }
                    else
                    {
                        tmin = min(tmin, HLLDQ_Korolkov(sro1, Q, sp1, su1, sv1, sw1, bx, by, bz, sro2, Q_2, sp2, su2, sv2, sw2, bx2, by2, bz2, P, PQ, n1, n2, n3, dist, my_metod));
                    }


                    for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                    {
                        Potok[k] = Potok[k] + P[k] * S;
                    }
                    Potok[9] = Potok[9] + PQ * S;
                }

            }
            else if (ii == -1)
            {
                double S = dy * dz * 4.0;
                n1 = 1.0;
                n2 = 0.0;
                n3 = 0.0;
                dist = dx;
                sks = n1 * (bx + bxC) / 2.0 + n2 * (by + byC) / 2.0 + n3 * (bz + bzC) / 2.0;

                /*double uu = u;
                if (uu < 0.0)
                {
                    uu = 0.0;
                }*/
                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, my_metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, roC, QC, pC, uC, vC, wC, bxC, byC, bzC, P, PQ, n1, n2, n3, dist, my_metod));
                }

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -2)
            {
                double S = dy * dz * 4.0;
                n1 = -1.0;
                n2 = 0.0;
                n3 = 0.0;
                dist = dx;
                sks = n1 * bx + n2 * by + n3 * bz;

                Potok[8] = Potok[8] + sks * S;
                double uu = u;
                if (uu > M_infty / 3.0)
                {
                    uu = M_infty;
                }
                /*else if (uu > -0.01)
                {
                    uu = -0.01;
                }*/

                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, uu, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, uu, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }

                /*double t1, t2, t3, m1, m2, m3;
                double bx_L = bx / spi4;
                double by_L = by / spi4;
                double bz_L = bz / spi4;
                t1 = 0.0;
                t2 = 0.0;
                t3 = 1.0;
                m1 = 0.0;
                m2 = 1.0;
                m3 = 0.0;
                double u1 = uu * n1 + v * n2 + w * n3;
                double v1 = uu * t1 + v * t2 + w * t3;
                double w1 = uu * m1 + v * m2 + w * m3;
                double bn1, bt1, bm1;
                bn1 = bx_L * n1 + by_L * n2 + bz_L * n3;
                bt1 = bx_L * t1 + by_L * t2 + bz_L * t3;
                bm1 = bx_L * m1 + by_L * m2 + bz_L * m3;
                double uu_L = (kv(uu) + kv(v) + kv(w)) / 2.0;
                double bb_L = kv(bx_L) + kv(by_L) + kv(bz_L);
                double e1 = p / g1 + ro * uu_L + bb_L / 2.0;
                double pTL = p + bb_L / 2.0;

                double PO[9];

                PO[0] = ro * u1;
                PO[1] = ro * u1 * u1 + pTL - kv(bn1);
                PO[2] = ro * u1 * v1 - bn1 * bt1;
                PO[3] = ro * u1 * w1 - bn1 * bm1;
                PO[4] = (e1 + pTL) * u1 - bn1 * (u1 * bn1 + v1 * bt1 + w1 * bm1);
                PO[5] = 0.0;
                PO[6] = u1 * bt1 - v1 * bn1;
                PO[7] = u1 * bm1 - w1 * bn1;
                PO[8] = Q * u1;


                P[1] = n1 * PO[1] + t1 * PO[2] + m1 * PO[3];
                P[2] = n2 * PO[1] + t2 * PO[2] + m2 * PO[3];
                P[3] = n3 * PO[1] + t3 * PO[2] + m3 * PO[3];
                P[5] = spi4 * (n1 * PO[5] + t1 * PO[6] + m1 * PO[7]);
                P[6] = spi4 * (n2 * PO[5] + t2 * PO[6] + m2 * PO[7]);
                P[7] = spi4 * (n3 * PO[5] + t3 * PO[6] + m3 * PO[7]);
                P[0] = PO[0];
                P[4] = PO[4];
                PQ = PO[8];

                double SWAP = P[4];
                P[4] = P[5];
                P[5] = P[6];
                P[6] = P[7];
                P[7] = SWAP;*/

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -3)
            {
                double S = dx * dz * 4.0;
                n1 = 0.0;
                n2 = 1.0;
                n3 = 0.0;
                dist = dy;

                sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;

                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -4)
            {
                double S = dx * dz * 4.0;
                n1 = 0.0;
                n2 = -1.0;
                n3 = 0.0;
                dist = dy;
                sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;

                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, -v, w, -bx, by, -bz, P, PQ, n1, n2, n3, dist, my_metod));
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, pC, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));

                }
                else
                {
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, -v, w, -bx, by, -bz, P, PQ, n1, n2, n3, dist, my_metod));
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));

                }

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -5)
            {
                double S = dy * dx * 4.0;
                n1 = 0.0;
                n2 = 0.0;
                n3 = 1.0;
                dist = dz;
                sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;


                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                else
                {
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }

                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else if (ii == -6)
            {
                double S = dy * dx * 4.0;
                n1 = 0.0;
                n2 = 0.0;
                n3 = -1.0;
                dist = dz;
                sks = n1 * (bx + bx) / 2.0 + n2 * (by + by) / 2.0 + n3 * (bz + bz) / 2.0;

                kk = SOSED2[i];

                if (kk != -1) //&& ddd2 > 0.8
                {
                    w3 = W1[kk];

                    double dd = 0.0;
                    if (n1 != 0)
                    {
                        dd = dx;
                    }
                    else if (n2 != 0)
                    {
                        dd = dy;
                    }
                    else if (n3 != 0)
                    {
                        dd = dz;
                    }
                    else
                    {
                        printf("Errrrrr 2323132214243\n");
                    }


                    f_TVD(dd, w, -w, w3, -w3, w12, w21, 2 * dz, 2 * dz, 2 * dz);
                }
                else
                {
                    w12 = w;
                    w21 = -w;
                }

                if (abs(w12 + w21) > 0.0000001)
                {
                    printf("Errrrrr tyhgrtgertf345rt3r43:  %lf, %lf\n", w12, w21);
                }


                Potok[8] = Potok[8] + sks * S;
                if (!kor_Sol || my_metod <= 1 || my_metod == 3)
                {
                    //tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                    tmin = min(tmin, HLLDQ_Alexashov(ro, Q, p, u, v, w12, bx, by, bz, ro, Q, p, u, v, w21, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                else
                {
                    //tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w, bx, by, bz, ro, Q, p, u, v, w, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                    tmin = min(tmin, HLLDQ_Korolkov(ro, Q, p, u, v, w12, bx, by, bz, ro, Q, p, u, v, w21, bx, by, bz, P, PQ, n1, n2, n3, dist, my_metod));
                }
                for (int k = 0; k < 8; k++)  // —уммируем все потоки в €чейке
                {
                    Potok[k] = Potok[k] + P[k] * S;
                }
                Potok[9] = Potok[9] + PQ * S;
            }
            else
            {
                printf("Error 12438jdyu. Ne doljni suda popadat = %d \n", ii);
            }
        }


        double q2_1 = 0.0, q2_2 = 0.0, q2_3 = 0.0, q3 = 0.0;

        if (true)//(istoch == true)
        {
            double u_H4 = M_infty, v_H4 = 0.0, w_H4 = 0.0, ro_H4 = 1.0, p_H4 = 1.0 / (2.0 * ggg);

            double U_M_H4 = sqrt(kv(u - u_H4) + kv(v - v_H4) + kv(w - w_H4) + (64.0 / (9.0 * pi)) //
                * (p / ro + 2.0 * p_H4 / ro_H4));

            double U_H4 = sqrt(kv(u - u_H4) + kv(v - v_H4) + kv(w - w_H4) + (4.0 / pi) //
                * (p / ro + 2.0 * p_H4 / ro_H4));

            double sigma_H4 = kv(1.0 - a_2 * log(U_M_H4));

            double nu_H4 = ro * ro_H4 * U_M_H4 * sigma_H4;

            q2_1 = (1.5 / Kn_) * (nu_H4 * (u_H4 - u));
            q2_2 = (1.5 / Kn_) * (nu_H4 * (v_H4 - v));
            q2_3 = (1.5 / Kn_) * (nu_H4 * (w_H4 - w));


            q3 = (1.5 / Kn_) * (nu_H4 * ((kv(u_H4) + kv(v_H4) + kv(w_H4) - kv(u) - kv(v) - kv(w)) / 2.0 + //
                (U_H4 / U_M_H4) * (2.0 * p_H4 / ro_H4 - p / ro)));
        }


        double ro33, p33, u33, v33, w33, bx33, by33, bz33, Q33;


        Q33 = Q - *T_do * Potok[9] / Volume;
        ro33 = ro - *T_do * Potok[0] / Volume;
        if (ro33 <= 0.0)
        {
            printf("ERROR -  dssdbfhfshjskfutytqqazz\n");
            printf("%lf, %lf, %lf, %lf\n", x, y, z, ro33);
            ro33 = ro;
        }
        u33 = (ro * u - *T_do * (Potok[1] + (bx / cpi4) * Potok[8] - q2_1 * Volume) / Volume) / ro33;
        v33 = (ro * v - *T_do * (Potok[2] + (by / cpi4) * Potok[8] - q2_2 * Volume) / Volume) / ro33;
        w33 = (ro * w - *T_do * (Potok[3] + (bz / cpi4) * Potok[8] - q2_3 * Volume) / Volume) / ro33;
        bx33 = (bx - *T_do * (Potok[4] + u * Potok[8]) / Volume);
        by33 = (by - *T_do * (Potok[5] + v * Potok[8]) / Volume);
        bz33 = (bz - *T_do * (Potok[6] + w * Potok[8]) / Volume);
        p33 = ((U8(ro, p, u, v, w, bx, by, bz) - *T_do * (Potok[7] + (skk(u, v, w, bx, by, bz) / cpi4) * Potok[8] - q3 * Volume)//
            / Volume) - 0.5 * ro33 * kvv(u33, v33, w33) - kvv(bx33, by33, bz33) / cpi8) * (ggg - 1.0);
        //u3 = (ro * u - *T_do * (Potok[1] + (bx) * Potok[8]) / Volume) / ro3;
        //v3 = (ro * v - *T_do * (Potok[2] + (by) * Potok[8]) / Volume) / ro3;
        //w3 = (ro * w - *T_do * (Potok[3] + (bz) * Potok[8]) / Volume) / ro3;
        //bx3 = (bx - *T_do * (Potok[4] + u * Potok[8]) / Volume);
        //by3 = (by - *T_do * (Potok[5] + v * Potok[8]) / Volume);
        //bz3 = (bz - *T_do * (Potok[6] + w * Potok[8]) / Volume);
        //p3 = ((U8(ro, p, u, v, w, bx, by, bz) - *T_do * (Potok[7] + (skk(u, v, w, bx, by, bz)) * Potok[8])//
        //    / Volume) - 0.5 * ro3 * kvv(u3, v3, w3) - kvv(bx3, by3, bz3)) * (ggg - 1.0);
        if (p33 <= 0)
        {
            p33 = 0.000001;
        }

        Q2[index] = Q33;
        RO2[index] = ro33;
        P2[index] = p33;
        U2[index] = u33;
        V2[index] = v33;
        W2[index] = w33;
        BX2[index] = bx33;
        BY2[index] = by33;
        BZ2[index] = bz33;

        if (*T > tmin)
        {
            *T = tmin;
            __threadfence();
        }
    }

}



int main()
{
    
    
    //Konstruktor K("Golikov_Setka_file_HLLC_3.0Max_4Alf.txt");
    //K.count_j();
    //K.print_Tecplot_z_20(0.01, 0.0, "3.0");
    //K.print_Tecplot_y_20(0.01, 0.0, "3.0");
    ///*K.print_Tecplot_z_j(0.01, 0.0, "3.0_j");
    //K.print_Tecplot_y_j(0.01, 0.0, "3.0_j");*/
    ////K.print_3D_20();
    //return 0;

    if (true)
    {
        // Add vectors in parallel.
        cudaError_t cudaStatus = addWithCuda();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }

        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces
        // Ёта штука должна быть вызвана
        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!\n");
            return 1;
        }
    }


    return 0;
}


cudaError_t addWithCuda()
{
    cudaError_t cudaStatus;


    // ќснова дл€ полной геометрии
    //Konstruktor K(32, 32, 32, -3000.0 * ae1, 3000.0 * ae1, -3000.0 * ae1, 3000.0 * ae1, -3000.0 * ae1, 3000.0 * ae1);

    //cout << "(1) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(-1800.0 * ae1, 1200.0 * ae1, -1800.0 * ae1, 1800.0 * ae1, -1800.0 * ae1, 1800.0 * ae1, 2);
    //cout << "(2) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(-900.0 * ae1, 700.0 * ae1, -1200.0 * ae1, 1200.0 * ae1, -1200.0 * ae1, 1200.0 * ae1, 2);
    //cout << "(3) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(-400.0 * ae1, 350.0 * ae1, -400.0 * ae1, 400.0 * ae1, -400.0 * ae1, 400.0 * ae1, 2);
    //cout << "(4) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(-300.0 * ae1, 250.0 * ae1, -300.0 * ae1, 300.0 * ae1, -300.0 * ae1, 300.0 * ae1, 2);
    //cout << "(5) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(0.0, 0.0, 0.0, 30.0 * ae1, 160.0 * ae1, 2, false);
    //cout << "(6) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(0.0, 0.0, 0.0, 20.0 * ae1, 100.0 * ae1, 2, false);
    //cout << "All size = " << K.all_Kyb.size() << endl;

    // ќснова дл€ половинной геометрии

    //Konstruktor K(32, 32, 16, -3000.0 * ae1, 3000.0 * ae1, -3000.0 * ae1, 3000.0 * ae1, 0.0 * ae1, 3000.0 * ae1);

    //cout << "(1) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(-1800.0 * ae1, 1200.0 * ae1, -1800.0 * ae1, 1800.0 * ae1, -1800.0 * ae1, 1800.0 * ae1, 2);
    //cout << "(2) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(-1000.0 * ae1, 700.0 * ae1, -1200.0 * ae1, 1200.0 * ae1, -1200.0 * ae1, 1200.0 * ae1, 2);
    //cout << "(3) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(-600.0 * ae1, 350.0 * ae1, -600.0 * ae1, 600.0 * ae1, -600.0 * ae1, 600.0 * ae1, 2);
    //cout << "(4) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(-500.0 * ae1, 250.0 * ae1, -500.0 * ae1, 500.0 * ae1, -500.0 * ae1, 500.0 * ae1, 2);
    //cout << "(5) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(0.0, 0.0, 0.0, 30.0 * ae1, 160.0 * ae1, 2, false);
    //cout << "(6) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(0.0, 0.0, 0.0, 20.0 * ae1, 100.0 * ae1, 2, false);
    //cout << "All size = " << K.all_Kyb.size() << endl;


    // ƒобавл€ем два мельчени€: -450  230      370
    // ƒобавл€ем два мельчени€: -350  200      330

    //Konstruktor K(100, 100, 160,   -3.06553, 3.06553, -3.06553, 3.06553,   0.0, 4.9048102);   // !!!!!!!!!!!!!!!!!!!!!!!
    //Konstruktor K("binary_Moscow_Boston_3-HLLD_TVD_2025.dat", true);

    Konstruktor K("binary_Maat-HLLC-TVD-atoms-1.3.dat", true);

    //cout << "(1) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(-500.0 * ae1, 450.0 * ae1, -500.0 * ae1, 500.0 * ae1, -500.0 * ae1, 500.0 * ae1, 2);
    //cout << "(2) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(-280.0 * ae1, 230.0 * ae1, -280.0 * ae1, 280.0 * ae1, -280.0 * ae1, 280.0 * ae1, 2);
    //cout << "(3) All size = " << K.all_Kyb.size() << endl;


    //cout << "(1) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(-800.0 * ae1, 400.0 * ae1, -700.0 * ae1, 700.0 * ae1, -700.0 * ae1, 700.0 * ae1, 2);
    //cout << "(2) All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(-420.0 * ae1, 190.0 * ae1, -240.0 * ae1, 240.0 * ae1, -370.0 * ae1, 370.0 * ae1, 2);
    //cout << "(3) All size = " << K.all_Kyb.size() << endl;

    // —только уже не вывозим:
    //K.Drobim(-400.0 * ae1, 170.0 * ae1, -220.0 * ae1, 220.0 * ae1, -350.0 * ae1, 350.0 * ae1, 2);
    //cout << "(4) All size = " << K.all_Kyb.size() << endl;


    // "Maat-Laks-1"
    // "Maat-HLL-1"
    // "Maat-HLLC-atoms-1"
    // "Maat-HLLC-TVD-atoms-1"    "Maat-HLLC-TVD-atoms-1.2" - улучшил “¬ƒ на стыках
    // "Maat-HLLC-TVD-atoms-2"   добавил €чеек два набора
    // 


    string nam = "Maat-HLLC-TVD-atoms-1.3";  // »м€ дл€ вывода файлов
    //string nam = "inst_N_16_MA_4_2025";  // »м€ дл€ вывода файлов
    //string nam = "inst_N_31movi_2024";  // »м€ дл€ вывода файлов

    if (false)
    {
        cout << "All size 2 = " << K.all_Kyb.size() << endl;
        K.Drobim_z(-20.0, 3.5, 2.6, 2);

        cout << "All size 1 = " << K.all_Kyb.size() << endl;
        K.Drobim_x(-3.0, 2.0, 2.0, 2);

        cout << "All size 1 = " << K.all_Kyb.size() << endl;
        K.Drobim_x(-4.0, 1.0, 1.5, 2);
    }

    //cout << "All size 1 = " << K.all_Kyb.size() << endl;
    //K.Drobim_z_2(3.5, 6.5, 2.5, -1.5, 0.0, 2);
    //cout << "All size 1 = " << K.all_Kyb.size() << endl;
    //K.Drobim_x(-5.0, -1.0, 3.0, 2);

    


    cout << "All size 2 = " << K.all_Kyb.size() << endl;
    //K.Drobim(-1.5, 1.5, -1.5, 1.5, 0.0, 2.5, 2);


    // ¬от это дробление работает! ƒл€ сравнени€ Ѕостона и ћосквы моделей
    //K.Drobim_z(-20.0, 3.06553, 1.8393, 2);
    //K.Drobim_z(-20.0, 1.8393, 1.686045, 2);


    //K.Drobim(0.0, 0.0, 0.0, 0.4, 1.2, 2, false);
    //K.Drobim(0.0, 0.0, 0.0, 0.4, 1.0, 2, false);
    //K.Drobim(0.0, 0.0, 0.0, 0.1, 0.75, 2, false);

    /*cout << "All size 2 = " << K.all_Kyb.size() << endl;
    K.Drobim_z(-20.0, 2.5, 2.0, 2);

    cout << "All size 3 = " << K.all_Kyb.size() << endl;
    K.Drobim(0.0, 0.0, 0.0, 0.2, 1.0, 2, false);

    cout << "All size 3 = " << K.all_Kyb.size() << endl;
    K.Drobim(0.0, 0.0, 0.0, 0.2, 1.5, 2, false);

    cout << "All size 3 = " << K.all_Kyb.size() << endl;
    K.Drobim(0.0, 0.0, 0.0, 0.2, 0.7, 2, false);*/



    //cout << "All size 3 = " << K.all_Kyb.size() << endl;
    //K.Drobim(0.0, 0.0, 0.0, 0.2, 3.0, 2, false);

    //cout << "All size 4 = " << K.all_Kyb.size() << endl;
    //K.Drobim(0.0, 0.0, 0.0, 0.0, 0.73, 2, false);

    //cout << "All size 4 = " << K.all_Kyb.size() << endl;
    //K.Drobim(0.0, 0.0, 0.0, 0.0, 0.73, 2, false);

    //cout << "All size 5 = " << K.all_Kyb.size() << endl;
    //K.Drobim(0.0, 0.0, 0.0, 0.0, 0.35, 2, false);


    //cout << "All size = " << K.all_Kyb.size() << endl;
    //K.Drobim(-1.08, 0.0, 0.0, 1.4, 6.3, 2, false);

    /*cout << "All size = " << K.all_Kyb.size() << endl;
    K.Drobim(-1.08, 0.0, 0.0, 1.4, 5.5, 2, false);*/



    cout << "All size = " << K.all_Kyb.size() << endl;


    cout << "proverca start" << endl;
    //for (Kyb* & i : K.all_Kyb)  // ѕровер€ем, чтобы у соседа текущей €чейки в сосед€х тоже была текуща€ €чейка
    //{
    //    for (Kyb* & k : i->sosed)
    //    {
    //        if (k->number >= 0)
    //        {
    //            auto I = find_if(k->sosed.begin(), k->sosed.end(), [&](Kyb*& A)
    //                {
    //                    if (A->number == i->number)
    //                    {
    //                        return true;
    //                    }
    //                    else return false;
    //                });
    //            if (I == k->sosed.end())
    //            {
    //                cout << "Eror edwwdwedweew 12313234354364323 " << i->number << " " << k->number << endl;
    //            }
    //        }
    //    }
    //}

    cout << "proverca end" << endl;

    cout << "proverca2 start" << endl;
    // Ќужно добавить проверку функции определени€ соседей.
    for (Kyb*& i : K.all_Kyb)  // ѕровер€ем, чтобы у соседа текущей €чейки в сосед€х тоже была текуща€ €чейка
    {
        for (Kyb*& k : i->sosed)
        {
            if (k->number >= 0)
            {
                if (K.get_square(i, k) == false)
                {
                    cout << "EROR mesh 3845kdjdgef" << endl;
                    exit(-1);
                }
            }
        }
    }
    cout << "proverca2 end" << endl;

    //Konstruktor K(30, 36, 36, -1000, 550, -2100, 2100, -2100, 2100);
    //Konstruktor K(4, 4, 4, -1500, 420, -1800, 1800, -1800, 1800);
    //Konstruktor K("Setka_file_test.txt");
    /*for (auto& i : K.all_Kyb)
    {
        cout << "n = " << i->sosed.size() << " " << i->x << " " << i->y << " " << i->z  << endl;
    }*/
    int N = K.all_Kyb.size();          // „исло €чеек
    cout << "All size = " << N << endl;
    int nn = K.get_size_conektiv();    // „исло св€зей (размер массива св€зей)
    cout << "connect = " << nn << endl;
    cout << "Sozdal" << endl;

    

    //K.get_inner();   // ѕопытка считать граничные услови€ из 2ƒ задачи

    //K.filling();
    //K.filling_mini();



    K.filling_mini();


    cout << "Zapolnil" << endl;
    int* host_sosed;
    int* host_sosed2;
    int* host_sosed22;
    int* host_sosed23;
    int* host_sosed24;
    int* dev_sosed;
    int* dev_sosed2;
    int* dev_sosed22;
    int* dev_sosed23;
    int* dev_sosed24;
    int* host_sosed3;
    int* host_sosed32;
    int* host_sosed33;
    int* host_sosed34;
    int* dev_sosed3;
    int* dev_sosed32;
    int* dev_sosed33;
    int* dev_sosed34;
    double* host_T, * host_T_do, * host_TT;
    double* dev_T, * dev_T_do, * dev_TT;
    int* host_i;
    int* dev_i;
    int My_n1 = 0;                                         // Ёту €чейку выводим на каждом шаге дл€ р€дов ‘урье
    bool istoch = false;

    /*double v_do = 108;
    double v_posle = 13.3987;
    double r_do = 474.47;
    double r_posle = 99.4899;
    double rho_do = 0.00017651;
    double rho_posle = 0.0669;
    double p_do = rho_do * v_do * v_do;
    double p_posle = rho_posle * v_posle * v_posle;
    double b_do = v_do * sqrt(rho_do);
    double b_posle = v_posle * sqrt(rho_posle);*/

    // —оздаЄм массивы переменных\€чеек
    double MMM = 0.0;                             //------------------------------------------------------------------------------------------------------------------------------
    int* dev_l, * dev_r, * Nu, * dev_N;
    double* dev_x, * dev_y, * dev_z;
    double* dev_dx, * dev_dy, * dev_dz;
    double* dev_Q1, * host_Q1, * dev_Q2;
    double* dev_ro1, * dev_p1, * dev_u1, * dev_v1, * dev_w1, * dev_ro2, * dev_p2, * dev_u2, * dev_v2, * dev_w2;
    double* dev_bx1, * dev_by1, * dev_bz1;
    double* dev_bx2, * dev_by2, * dev_bz2;
    double* host_x, * host_y, * host_z;
    double* host_dx, * host_dy, * host_dz;
    double* host_ro1, * host_p1, * host_u1, * host_v1, * host_w1, * host_bx1, * host_by1, * host_bz1;
    int* host_l, * host_r;
    double time_null = -1.0;
    time_t start_time, end_time;
    double seconds;

    int deviceCount;
    cudaDeviceProp deviceProp;

    ofstream fout_fur;
    int kjk;

    host_T = (double*)malloc(sizeof(double));
    host_T_do = (double*)malloc(sizeof(double));
    host_TT = (double*)malloc(sizeof(double));
    host_i = (int*)malloc(sizeof(int));
    Nu = (int*)malloc(sizeof(int));

    int NNN = (int)(N / 256) + 1;
    host_x = new double[N];
    host_y = new double[N];
    host_z = new double[N];
    host_dx = new double[N];
    host_dy = new double[N];
    host_dz = new double[N];
    host_ro1 = new double[N];
    if (TVQ_)
    {
        host_Q1 = new double[N];
    }
    host_p1 = new double[N];
    host_u1 = new double[N];
    host_v1 = new double[N];
    host_w1 = new double[N];
    host_bx1 = new double[N];
    host_by1 = new double[N];
    host_bz1 = new double[N];
    /*host_ro2 = new double[N];
    host_p2 = new double[N];
    host_u2 = new double[N];
    host_v2 = new double[N];
    host_w2 = new double[N];*/
    host_l = new int[N];
    host_r = new int[N];

    *host_T = 10000000.0;
    *host_T_do = 0.00000001;
    *host_TT = 0.0;
    *host_i = 0;
    *Nu = N;

    //// ¬ыбор на каком GPU работаем (дл€ систем с несколькими GPU актуально)
    //cudaStatus = cudaSetDevice(0);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    //    goto Error;
    //}

    host_sosed = new int[nn];
    if (TVD_ == true)
    {
        host_sosed2 = new int[nn];
        host_sosed3 = new int[nn];
        host_sosed22 = new int[nn];
        host_sosed23 = new int[nn];
        host_sosed24 = new int[nn];
        host_sosed32 = new int[nn];
        host_sosed33 = new int[nn];
        host_sosed34 = new int[nn];
        cout << "Generiruem TVD massivi - start" << endl;
        K.Generate_sosed_for_TVD(host_sosed2, host_sosed3, host_sosed22, host_sosed23, host_sosed24, host_sosed32, host_sosed33, host_sosed34);
        //K.Generate_sosed_for_TVD_symmetry(host_sosed2, host_sosed3, host_sosed22, host_sosed23, host_sosed24, host_sosed32, host_sosed33, host_sosed34);
        cout << "Generiruem TVD massivi - end" << endl;
    }

    cout << "Uspex" << endl;

    // «аполнение массивов
    int c = 0;
    for (Kyb*& i : K.all_Kyb)
    {
        for (Kyb*& j : i->sosed)
        {
            host_sosed[c] = j->number;
            c++;
        }
    }
    cout << "11111" << endl;
    int ll = 0;
    for (int i = 0; i < K.all_Kyb.size(); i++)
    {
        host_x[i] = K.all_Kyb[i]->x;
        host_y[i] = K.all_Kyb[i]->y;
        host_z[i] = K.all_Kyb[i]->z;
        host_dx[i] = K.all_Kyb[i]->dx;
        host_dy[i] = K.all_Kyb[i]->dy;
        host_dz[i] = K.all_Kyb[i]->dz;
        host_ro1[i] = K.all_Kyb[i]->ro;
        host_Q1[i] = K.all_Kyb[i]->Q;
        host_p1[i] = K.all_Kyb[i]->p;
        host_u1[i] = K.all_Kyb[i]->u;
        host_v1[i] = K.all_Kyb[i]->v;
        host_w1[i] = K.all_Kyb[i]->w;
        host_bx1[i] = K.all_Kyb[i]->Bx;
        host_by1[i] = K.all_Kyb[i]->By;
        host_bz1[i] = K.all_Kyb[i]->Bz;
        host_l[i] = ll;
        host_r[i] = ll + K.all_Kyb[i]->sosed.size() - 1;
        ll = ll + K.all_Kyb[i]->sosed.size();
    }
    cout << "2222222" << endl;
    if (false)//(TVD_ == true) // ћожно проверить правильность работы “¬ƒ
    {
        cout << "proverca TVD " << endl;;
        int km = -1;
        for (Kyb*& i : K.all_Kyb)
        {
            for (Kyb*& j : i->sosed)
            {
                km++;
                if (host_sosed2[km] == -1)
                {
                    continue;
                }
                double m1 = i->x - K.all_Kyb[host_sosed[km]]->x;
                double m2 = i->y - K.all_Kyb[host_sosed[km]]->y;
                double m3 = i->z - K.all_Kyb[host_sosed[km]]->z;
                double k1 = K.all_Kyb[host_sosed2[km]]->x - K.all_Kyb[host_sosed3[km]]->x;
                double k2 = K.all_Kyb[host_sosed2[km]]->y - K.all_Kyb[host_sosed3[km]]->y;
                double k3 = K.all_Kyb[host_sosed2[km]]->z - K.all_Kyb[host_sosed3[km]]->z;
                if (skk(m1, m2, m3, k1, k2, k3) / (sqrt(kv(m1) + kv(m2) + kv(m3)) * sqrt(kv(k1) + kv(k2) + kv(k3))) < 0.9)
                {
                    cout << "NE PROIDEN!  " << skk(m1, m2, m3, k1, k2, k3) / (sqrt(kv(m1) + kv(m2) + kv(m3)) * sqrt(kv(k1) + kv(k2) + kv(k3))) << endl;
                    cout << i->x << " " << i->y << " " << i->z << endl;
                    cout << K.all_Kyb[host_sosed[km]]->x << " " << K.all_Kyb[host_sosed[km]]->y << " " << K.all_Kyb[host_sosed[km]]->z << endl;
                    cout << K.all_Kyb[host_sosed2[km]]->x << " " << K.all_Kyb[host_sosed2[km]]->y << " " << K.all_Kyb[host_sosed2[km]]->z << endl;
                    cout << K.all_Kyb[host_sosed3[km]]->x << " " << K.all_Kyb[host_sosed3[km]]->y << " " << K.all_Kyb[host_sosed3[km]]->z << endl;
                    exit(-1);
                }
            }
        }

        cout << "proshla yspeshno" << endl;
    }
    else
    {
        cout << "proverka TVD ne provodilas" << endl;
    }


    cout << "Sozdal massivi Cuda" << endl;

    if (TVD_)
    {
        cout << "Pamyat = " << (N * 15 * sizeof(double) + 4 * nn * sizeof(int)) / 1000000000.0 << endl;
    }
    else
    {
        cout << "Pamyat = " << (N * 15 * sizeof(double) + nn * sizeof(int)) / 1000000000.0 << endl;
    }



    // ¬ыделение пам€ти на девайсе
    if (true)
    {
        cudaStatus = cudaMalloc((void**)&dev_x, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 1!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_y, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 2!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_z, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 3!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_dx, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 4!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_dy, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 5!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_dz, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 6!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_ro1, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 7!");
            goto Error;
        }

        if (TVQ_)
        {
            cudaStatus = cudaMalloc((void**)&dev_Q1, N * sizeof(double));
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed 8!");
                goto Error;
            }

            cudaStatus = cudaMalloc((void**)&dev_Q2, N * sizeof(double));
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed 9!");
                goto Error;
            }
        }

        cudaStatus = cudaMalloc((void**)&dev_ro2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 10!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_p1, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 11!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_p2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 12!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_u1, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 13!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_u2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 14!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_v1, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 15!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_v2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 16!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_w1, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 17!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_w2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 18!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_l, N * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 19!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_r, N * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 20!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_sosed, nn * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 21!");
            goto Error;
        }

        if (TVD_)
        {
            cudaStatus = cudaMalloc((void**)&dev_sosed2, nn * sizeof(int));
            cudaStatus = cudaMalloc((void**)&dev_sosed22, nn * sizeof(int));
            cudaStatus = cudaMalloc((void**)&dev_sosed23, nn * sizeof(int));
            cudaStatus = cudaMalloc((void**)&dev_sosed24, nn * sizeof(int));
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed 21!");
                goto Error;
            }

            cudaStatus = cudaMalloc((void**)&dev_sosed3, nn * sizeof(int));
            cudaStatus = cudaMalloc((void**)&dev_sosed32, nn * sizeof(int));
            cudaStatus = cudaMalloc((void**)&dev_sosed33, nn * sizeof(int));
            cudaStatus = cudaMalloc((void**)&dev_sosed34, nn * sizeof(int));
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed 21!");
                goto Error;
            }

        }

        cudaStatus = cudaMalloc((void**)&dev_T, sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 22!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_T_do, sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 23!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_TT, sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 24!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_i, sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 25!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_N, sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 26!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_bx1, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 27!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_bx2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 28!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_by1, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 29!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_by2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 30!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_bz1, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 31!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_bz2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 32!");
            goto Error;
        }
    }


    //  опируем массивы с хоста на девайс
    if (true)
    {
        cudaStatus = cudaMemcpy(dev_sosed, host_sosed, nn * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed -1 !");
            goto Error;
        }

        if (TVD_)
        {
            cudaStatus = cudaMemcpy(dev_sosed2, host_sosed2, nn * sizeof(int), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(dev_sosed22, host_sosed22, nn * sizeof(int), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(dev_sosed23, host_sosed23, nn * sizeof(int), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(dev_sosed24, host_sosed24, nn * sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed 11 !");
                goto Error;
            }

            cudaStatus = cudaMemcpy(dev_sosed3, host_sosed3, nn * sizeof(int), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(dev_sosed32, host_sosed32, nn * sizeof(int), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(dev_sosed33, host_sosed33, nn * sizeof(int), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(dev_sosed34, host_sosed34, nn * sizeof(int), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed 11 !");
                goto Error;
            }


        }

        cudaStatus = cudaMemcpy(dev_x, host_x, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_x, host_x, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_y, host_y, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_z, host_z, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_dx, host_dx, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_dy, host_dy, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_dz, host_dz, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_ro1, host_ro1, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }
        if (TVQ_)
        {
            cudaStatus = cudaMemcpy(dev_Q1, host_Q1, N * sizeof(double), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed 012!");
                goto Error;
            }
        }
        cudaStatus = cudaMemcpy(dev_p1, host_p1, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_u1, host_u1, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_v1, host_v1, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_w1, host_w1, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_bx1, host_bx1, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_by1, host_by1, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_bz1, host_bz1, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_l, host_l, N * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_r, host_r, N * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_T, host_T, sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 1!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_TT, host_TT, sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 2!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_T_do, host_T_do, sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 3!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_i, host_i, sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 4!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_N, Nu, sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 4!");
            goto Error;
        }
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // «апуск €дра
    //Cuda_main << <N, 1 >> > (dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
    //    dev_ro1, dev_ro2, dev_p1, dev_p2, dev_u1, dev_u2, dev_v1, dev_v2,//
    //    dev_w1, dev_w2, dev_sosed, dev_l, dev_r, dev_T, dev_T_do);
    //Cuda_main_HLLD << <N / 256, 256 >> > (dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
    //    dev_ro1, dev_ro2, dev_p1, dev_p2, dev_u1, dev_u2, dev_v1, dev_v2,//
    //    dev_w1, dev_w2, dev_bx1, dev_by1, dev_bz1, dev_bx2, dev_by2, dev_bz2,//
    //    dev_sosed, dev_l, dev_r, dev_T, dev_T_do);




    // ЌайдЄм номер €чейки, котора€ нужна 
    My_n1 = 0;
    kjk = 0;
    for (Kyb*& i : K.all_Kyb)
    {
        kjk++;
        if (fabs(i->x + 0.001) <= 2.0 * i->dx)
        {
            if (fabs(i->y) <= 2.0 * i->dy)
            {
                if (fabs(i->z - 1.4) <= 2.0 * i->dz)
                {
                    My_n1 = kjk;
                    break;
                }
            }
        }
    }

    cout << "Vivodim  " << K.all_Kyb[My_n1]->x << " " << K.all_Kyb[My_n1]->y << " " << K.all_Kyb[My_n1]->z << endl;


    fout_fur.open(nam + "_rho_time.txt");

    time(&start_time);
    //nam = "1.97";
    MMM = 0.0;
    for (int i = 0; i < 0; i = i + 2)  // —колько шагов по времени делаем?
    {
        if (i % 10000 == 0)
        {
            cout << "from HOST HLLDQ " << i << endl;
        }
        // запускаем add() kernel на GPU, передава€ параметры
        Cuda_main_HLLDQ << <(int)(N / 256) + 1, 256 >> > (dev_N, dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
            dev_ro1, dev_ro2, dev_Q1, dev_Q2, dev_p1, dev_p2, dev_u1, dev_u2, dev_v1, dev_v2,//
            dev_w1, dev_w2, dev_bx1, dev_by1, dev_bz1, dev_bx2, dev_by2, dev_bz2,//
            dev_sosed, dev_l, dev_r, dev_T, dev_T_do, i, MMM, true, true, 2);

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 11111\n", cudaStatus);
            goto Error;
        }

        funk_time << <1, 1 >> > (dev_T, dev_T_do, dev_TT, dev_i);
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 222222\n", cudaStatus);
            goto Error;
        }

        Cuda_main_HLLDQ << <(int)(N / 256) + 1, 256 >> > (dev_N, dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
            dev_ro2, dev_ro1, dev_Q2, dev_Q1, dev_p2, dev_p1, dev_u2, dev_u1, dev_v2, dev_v1,//
            dev_w2, dev_w1, dev_bx2, dev_by2, dev_bz2, dev_bx1, dev_by1, dev_bz1,//
            dev_sosed, dev_l, dev_r, dev_T, dev_T_do, i, MMM, true, true, 2);

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 333333\n", cudaStatus);
            goto Error;
        }

        funk_time << <1, 1 >> > (dev_T, dev_T_do, dev_TT, dev_i);
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 4444444\n", cudaStatus);
            goto Error;
        }

        if ((i % 1000000000 == 0))
        {
            cout << "HLLC + D " + nam << endl;
            if (true)
            {
                cudaStatus = cudaMemcpy(host_ro1, dev_ro1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_p1, dev_p1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_u1, dev_u1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_v1, dev_v1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_w1, dev_w1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_bx1, dev_bx1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_by1, dev_by1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_bz1, dev_bz1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_Q1, dev_Q1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_TT, dev_TT, sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  34534\n");
                    goto Error;
                }
            }
            string fgf = nam + to_string(i);
            K.read_Cuda_massiv(host_ro1, host_p1, host_u1, host_v1, host_w1, host_bx1, host_by1, host_bz1, host_Q1);
            if (time_null < 0.0)
            {
                time_null = *host_TT;
            }
            K.print_Tecplot_y_20(0.0001, i, nam, *host_TT - time_null);
            K.print_Tecplot_x_20(0.0001, i, nam, *host_TT - time_null);
            K.print_Tecplot_z_20(0.0001, i, nam, *host_TT - time_null);
        }

        if ((i % 10000000000 == 0 && i > 1) || (i == 30000000000000))
        {
            if (true)
            {
                cudaStatus = cudaMemcpy(host_ro1, dev_ro1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_Q1, dev_Q1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452edw\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_p1, dev_p1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_u1, dev_u1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_v1, dev_v1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_w1, dev_w1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_bx1, dev_bx1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_by1, dev_by1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_bz1, dev_bz1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
            }

            K.binary_save_Setka("Moscow_Boston_1_promeg_HLLD");
        }

    }

    istoch = false;
    for (int i = 0; i < 0; i = i + 2)  // —колько шагов по времени делаем?
    {
        if (i % 500 == 0)
        {
            cout << "from HOST HLLC  " << i << endl;
        }

        /*if (i > 12000)
        {
            istoch = true;
        }*/


        // запускаем add() kernel на GPU, передава€ параметры
        Cuda_main_HLLDQ << <(int)(N / 256) + 1, 256 >> > (dev_N, dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
            dev_ro1, dev_ro2, dev_Q1, dev_Q2, dev_p1, dev_p2, dev_u1, dev_u2, dev_v1, dev_v2,//
            dev_w1, dev_w2, dev_bx1, dev_by1, dev_bz1, dev_bx2, dev_by2, dev_bz2,//
            dev_sosed, dev_l, dev_r, dev_T, dev_T_do, i, MMM, true, true, 3, istoch); //3
        //Cuda_main_HLLDQ_TVD << <(int)(N / 256) + 1, 256 >> > (dev_N, dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
        //    dev_ro1, dev_ro2, dev_Q1, dev_Q2, dev_p1, dev_p2, dev_u1, dev_u2, dev_v1, dev_v2,//
        //    dev_w1, dev_w2, dev_bx1, dev_by1, dev_bz1, dev_bx2, dev_by2, dev_bz2,//
        //    dev_sosed, dev_sosed2, dev_l, dev_r, dev_T, dev_T_do, i, MMM, true, true, 3);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 1weweqwewed11\n", cudaStatus);
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 11111\n", cudaStatus);
            goto Error;
        }

        funk_time << <1, 1 >> > (dev_T, dev_T_do, dev_TT, dev_i);
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 222222\n", cudaStatus);
            goto Error;
        }

        Cuda_main_HLLDQ << <(int)(N / 256) + 1, 256 >> > (dev_N, dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
            dev_ro2, dev_ro1, dev_Q2, dev_Q1, dev_p2, dev_p1, dev_u2, dev_u1, dev_v2, dev_v1,//
            dev_w2, dev_w1, dev_bx2, dev_by2, dev_bz2, dev_bx1, dev_by1, dev_bz1,//
            dev_sosed, dev_l, dev_r, dev_T, dev_T_do, i, MMM, true, true, 3, istoch);
        //Cuda_main_HLLDQ_TVD << <(int)(N / 256) + 1, 256 >> > (dev_N, dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
        //    dev_ro2, dev_ro1, dev_Q2, dev_Q1, dev_p2, dev_p1, dev_u2, dev_u1, dev_v2, dev_v1,//
        //    dev_w2, dev_w1, dev_bx2, dev_by2, dev_bz2, dev_bx1, dev_by1, dev_bz1,//
        //    dev_sosed, dev_sosed2, dev_l, dev_r, dev_T, dev_T_do, i, MMM,  true, true, 3);
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 333333\n", cudaStatus);
            goto Error;
        }

        funk_time << <1, 1 >> > (dev_T, dev_T_do, dev_TT, dev_i);
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 4444444\n", cudaStatus);
            goto Error;
        }

        if (i % 10 == 0)
        {
            cudaStatus = cudaMemcpy(&host_ro1[My_n1], &dev_ro1[My_n1], sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!  3452\n");
                goto Error;
            }
            cudaStatus = cudaMemcpy(host_TT, dev_TT, sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!  34534\n");
                goto Error;
            }

            fout_fur << *host_TT << " " << host_ro1[My_n1] << " " << i << endl;
        }

        if ((i % 25000 == 0 && i >= 0) || (i % 25000 == 0 && i >= 1) || i==1000 || i == 2000 || i == 3000 || i == 4000)
        {
            cout << "HLLC + D " + nam << endl;
            if (true)
            {
                cudaStatus = cudaMemcpy(host_ro1, dev_ro1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_p1, dev_p1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_u1, dev_u1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_v1, dev_v1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_w1, dev_w1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_bx1, dev_bx1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_by1, dev_by1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_bz1, dev_bz1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_Q1, dev_Q1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_TT, dev_TT, sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  34534\n");
                    goto Error;
                }
            }
            string fgf = nam + to_string(i);
            K.read_Cuda_massiv(host_ro1, host_p1, host_u1, host_v1, host_w1, host_bx1, host_by1, host_bz1, host_Q1);
            if (time_null < 0.0)
            {
                time_null = *host_TT;
            }

            K.count_j();
            K.print_Tecplot_y_20(0.0001, i, nam, *host_TT - time_null);
            //K.print_Tecplot_x_20(0.0001, i, nam + "000", *host_TT - time_null);
            K.print_Tecplot_z_20(0.800001, i, nam + "0.8", *host_TT - time_null);
            K.print_Tecplot_z_20(1.000001, i, nam + "1.0", *host_TT - time_null);
            K.print_Tecplot_z_20(1.200001, i, nam + "1.2", *host_TT - time_null);
            K.print_Tecplot_z_20(1.400001, i, nam + "1.4", *host_TT - time_null);
            K.print_Tecplot_z_20(1.600001, i, nam + "1.6", *host_TT - time_null);
            K.print_Tecplot_z_20(1.800001, i, nam + "1.8", *host_TT - time_null);
            K.print_Tecplot_z_20(2.000001, i, nam + "2.0", *host_TT - time_null);
            K.print_Tecplot_z_20(2.500001, i, nam + "2.5", *host_TT - time_null);
            K.print_Tecplot_z_20(3.000001, i, nam + "3.0", *host_TT - time_null);

        }

        if ((i % 300000 == 0 && i > 10))
        {
            if (true)
            {
                cudaStatus = cudaMemcpy(host_ro1, dev_ro1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_Q1, dev_Q1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452edw\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_p1, dev_p1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_u1, dev_u1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_v1, dev_v1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_w1, dev_w1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_bx1, dev_bx1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_by1, dev_by1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_bz1, dev_bz1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
            }

            K.binary_save_Setka("Golikov_Setka_file_moscow_27_2024_vremenniy");
        }
    }


    // 65000 - 10 минут дл€ HLLC + TVD

    // 19000 - это час дл€ HLLC + TVD с т€жЄлой сеткой на Nuclon
    // 13900 - это час дл€ HLLC + TVD с т€жЄлой сеткой на HSE
    for (int i = 0; i < 65000 * 6 * 5; i = i + 2)  // —колько шагов по времени делаем?
    {

        if (i % 10000 == 0)
        {
            cout << "from HOST HLLC + TVD  " << i << endl;
        }


        // запускаем add() kernel на GPU, передава€ параметры
        //Cuda_main_HLLDQ_TVD2 << <(int)(N / 256) + 1, 256 >> > (dev_N, dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
        //    dev_ro1, dev_ro2, dev_Q1, dev_Q2, dev_p1, dev_p2, dev_u1, dev_u2, dev_v1, dev_v2,//
        //    dev_w1, dev_w2, dev_bx1, dev_by1, dev_bz1, dev_bx2, dev_by2, dev_bz2,//
        //    dev_sosed, dev_sosed2, dev_sosed3, dev_l, dev_r, dev_T, dev_T_do, i, MMM, true, true, 2);

        //Cuda_main_HLLDQ_TVD3 
        //Cuda_main_HLLDQ_TVD3_symmetry
        Cuda_main_HLLDQ_TVD3 << <(int)(N / 256) + 1, 256 >> > (dev_N, dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
            dev_ro1, dev_ro2, dev_Q1, dev_Q2, dev_p1, dev_p2, dev_u1, dev_u2, dev_v1, dev_v2,//
            dev_w1, dev_w2, dev_bx1, dev_by1, dev_bz1, dev_bx2, dev_by2, dev_bz2,//
            dev_sosed, dev_sosed2, dev_sosed3, dev_sosed22, dev_sosed23, dev_sosed24, dev_sosed32, dev_sosed33, dev_sosed34,
            dev_l, dev_r, dev_T, dev_T_do, i, MMM, true, true, 2);
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 11111\n", cudaStatus);
            goto Error;
        }

        funk_time << <1, 1 >> > (dev_T, dev_T_do, dev_TT, dev_i);
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 222222\n", cudaStatus);
            goto Error;
        }

        //Cuda_main_HLLDQ_TVD2 << <(int)(N / 256) + 1, 256 >> > (dev_N, dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
        //    dev_ro2, dev_ro1, dev_Q2, dev_Q1, dev_p2, dev_p1, dev_u2, dev_u1, dev_v2, dev_v1,//
        //    dev_w2, dev_w1, dev_bx2, dev_by2, dev_bz2, dev_bx1, dev_by1, dev_bz1,//
        //    dev_sosed, dev_sosed2, dev_sosed3, dev_l, dev_r, dev_T, dev_T_do, i, MMM, true, true, 2);
        //Cuda_main_HLLDQ_TVD3 
        //Cuda_main_HLLDQ_TVD3_symmetry
        Cuda_main_HLLDQ_TVD3 << <(int)(N / 256) + 1, 256 >> > (dev_N, dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
            dev_ro2, dev_ro1, dev_Q2, dev_Q1, dev_p2, dev_p1, dev_u2, dev_u1, dev_v2, dev_v1,//
            dev_w2, dev_w1, dev_bx2, dev_by2, dev_bz2, dev_bx1, dev_by1, dev_bz1,//
            dev_sosed, dev_sosed2, dev_sosed3, dev_sosed22, dev_sosed23, dev_sosed24, dev_sosed32, dev_sosed33, dev_sosed34,
            dev_l, dev_r, dev_T, dev_T_do, i, MMM, true, true, 2);
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 333333\n", cudaStatus);
            goto Error;
        }

        funk_time << <1, 1 >> > (dev_T, dev_T_do, dev_TT, dev_i);
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 4444444\n", cudaStatus);
            goto Error;
        }

        if (i % 50000000 == 0)
        {
            cudaStatus = cudaMemcpy(&host_ro1[My_n1], &dev_ro1[My_n1], sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!  3452\n");
                goto Error;
            }
            cudaStatus = cudaMemcpy(host_TT, dev_TT, sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed!  34534\n");
                goto Error;
            }

            fout_fur << *host_TT << " " << host_ro1[My_n1] << " " << i << endl;
        }

        if ((i % 50000000 == 0 && i > 0) || (i % 50000000 == 0 && i > 7))
        {
            cout << "HLLD + TVD " + nam << endl;
            if (true)
            {
                cudaStatus = cudaMemcpy(host_ro1, dev_ro1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_p1, dev_p1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_u1, dev_u1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_v1, dev_v1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_w1, dev_w1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_bx1, dev_bx1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_by1, dev_by1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_bz1, dev_bz1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_Q1, dev_Q1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_TT, dev_TT, sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  34534\n");
                    goto Error;
                }
            }
            string fgf = nam + to_string(i);
            K.read_Cuda_massiv(host_ro1, host_p1, host_u1, host_v1, host_w1, host_bx1, host_by1, host_bz1, host_Q1);
            if (time_null < 0.0)
            {
                time_null = *host_TT;
            }
            K.print_Tecplot_y_20(0.0001, i, nam, *host_TT - time_null);
            //K.print_Tecplot_x_20(0.0001, i, nam, *host_TT - time_null);
            //K.print_Tecplot_z_20(0.0001, i, nam, *host_TT - time_null);

        }

        if ((i % 50000000 == 0 && i > 10))
        {
            if (true)
            {
                cudaStatus = cudaMemcpy(host_ro1, dev_ro1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_Q1, dev_Q1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452edw\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_p1, dev_p1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_u1, dev_u1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_v1, dev_v1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_w1, dev_w1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_bx1, dev_bx1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_by1, dev_by1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
                cudaStatus = cudaMemcpy(host_bz1, dev_bz1, N * sizeof(double), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed!  3452\n");
                    goto Error;
                }
            }

            K.binary_save_Setka("Golikov_Setka_file_moscow_26_2024_vremenniy");
        }


    }

    time(&end_time);
    seconds = difftime(end_time, start_time);
    printf("The time: %f seconds or %lf minutes or %lf hours\n", seconds, seconds / 60.0, seconds / 60.0 / 60.0);

    if (true)
    {
        cudaStatus = cudaMemcpy(host_ro1, dev_ro1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_p1, dev_p1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_u1, dev_u1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_v1, dev_v1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_w1, dev_w1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_bx1, dev_bx1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_by1, dev_by1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_bz1, dev_bz1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_Q1, dev_Q1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
    }
    K.read_Cuda_massiv(host_ro1, host_p1, host_u1, host_v1, host_w1, host_bx1, host_by1, host_bz1, host_Q1);
    //K.save_Setka("HLLC_" + nam + "GD");
    //K.print_Tecplot_z_20(0.001, 0.0, nam );
    K.count_j();
    K.print_Tecplot_y_20(0.001, 0.0, nam );
    K.print_Tecplot_z_20(0.001, 0.0, nam );
    //K.print_Tecplot_x_20(-20.001, 0.0, nam );
    //K.print_3D(nam);



    fout_fur.close();




    if (false)
    {
        cudaStatus = cudaMemcpy(host_ro1, dev_ro1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_p1, dev_p1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_u1, dev_u1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_v1, dev_v1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_w1, dev_w1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_bx1, dev_bx1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_by1, dev_by1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_bz1, dev_bz1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_Q1, dev_Q1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
    }
   /* K.read_Cuda_massiv(host_ro1, host_p1, host_u1, host_v1, host_w1, host_bx1, host_by1, host_bz1, host_Q1);
    K.save_Setka("HLL_1.1Max_4Alf");
    K.print_Tecplot_z_20(0.01, 0.0, "1.1");
    K.print_Tecplot_y_20(0.01, 0.0, "1.1");*/
   


    // cudaDeviceSynchronize ожидает завершени€ работы €дра и возвращает
    // любые ошибки, полученные в процессе выполнени€
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! dffddff\n", cudaStatus);
        goto Error;
    }

    //  опируем обратно на ’ост
    if (true)
    {
        cudaStatus = cudaMemcpy(host_ro1, dev_ro1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_Q1, dev_Q1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452edw\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_p1, dev_p1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_u1, dev_u1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_v1, dev_v1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_w1, dev_w1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_bx1, dev_bx1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_by1, dev_by1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
        cudaStatus = cudaMemcpy(host_bz1, dev_bz1, N * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
    }



Error:
    cudaFree(dev_sosed);
    cudaFree(dev_ro1);
    cudaFree(dev_ro2);
    cudaFree(dev_p1);
    cudaFree(dev_p2);
    cudaFree(dev_u1);
    cudaFree(dev_u2);
    cudaFree(dev_v1);
    cudaFree(dev_v2);
    cudaFree(dev_w1);
    cudaFree(dev_w2);
    cudaFree(dev_bx1);
    cudaFree(dev_bx2);
    cudaFree(dev_by1);
    cudaFree(dev_by2);
    cudaFree(dev_bz1);
    cudaFree(dev_bz2);
    cudaFree(dev_T);
    cudaFree(dev_i);
    cudaFree(dev_T_do);
    cudaFree(dev_TT);
    cudaFree(dev_Q1);
    cudaFree(dev_Q2);

    K.read_Cuda_massiv(host_ro1, host_p1, host_u1, host_v1, host_w1, host_bx1, host_by1, host_bz1, host_Q1);
    //for (auto& i : K.all_Kyb)   // ѕеренормировака
    //{
    //    i->Bx = i->Bx * b_do / b_posle;
    //    i->By = i->By * b_do / b_posle;
    //    i->Bz = i->Bz * b_do / b_posle;
    //    i->u = i->u * v_do / v_posle;
    //    i->v = i->v * v_do / v_posle;
    //    i->w = i->w * v_do / v_posle;
    //    i->ro = i->ro * rho_do / rho_posle;
    //    i->p = i->p * p_do / p_posle;
    //    i->x = i->x * r_do / r_posle;
    //    i->dx = i->dx * r_do / r_posle;
    //    i->y = i->y * r_do / r_posle;
    //    i->dy = i->dy * r_do / r_posle;
    //    i->z = i->z * r_do / r_posle;
    //    i->dz = i->dz * r_do / r_posle;
    //}
    //K.print_Tecplot_z_20(0.01, 0.0);
    //K.print_Tecplot_y_20(0.01, 0.0);
    //K.print_Tecplot_z(0.001, 0.0);
    //K.print_Tecplot_y(0.001, 0.0);
    /*K.print_Tecplot_x(0.001, 0.0);
    K.print_some_point();*/
    /*K.print_3D_20();*/
    /*K.print_3D();*/

    //K.save_Setka("inst_6+_MA_4.txt");

    K.binary_save_Setka(nam);

    return cudaStatus;
}

cudaError_t addWithCuda_G_D() // √азова€ динамика
{
    Konstruktor K(400, 400, 400, -2.4, 2.4, -2.4, 2.4, -2.4, 2.4);
    /*K.Drobim(-0.9, 0.9, -0.9, 0.9, -0.9, 0.9, 2);
    cout << "0" << endl;
    K.Drobim(-0.8, -0.1, -0.8, 0.8, -0.8, 0.8, 3);
    cout << "1" << endl;
    K.Drobim(0.1, 0.8, -0.8, 0.8, -0.8, 0.8, 3);
    cout << "2" << endl;
    K.Drobim(-0.1, 0.1, -0.8, -0.1, -0.8, 0.8, 3);
    cout << "3" << endl;
    K.Drobim(-0.1, 0.1, 0.1, 0.8, -0.8, 0.8, 3);
    cout << "4" << endl;
    K.Drobim(-0.1, 0.1, -0.1, 0.1, -0.8, -0.1, 3);
    cout << "5" << endl;
    K.Drobim(-0.1, 0.1, -0.1, 0.1, 0.1, 0.8, 3);
    cout << "6" << endl;*/
    /*cout << "0" << endl;
    K.Drobim(0.05, 1.6, 3);
    cout << "1" << endl;
    K.Drobim(0.05, 1.2, 2);
    cout << "2" << endl;
    K.Drobim(0.6, 0.9, 2);
    cout << "3" << endl;
    K.Drobim(0.7, 0.8, 2);
    cout << "4" << endl;*/

    int N = K.all_Kyb.size();          // „исло €чеек
    int th = 256;
    for (th = 256; th > 0; th--)
    {
        if (N % th == 0)
        {
            break;
        }
    }
    cout << "All size = " << N << endl;
    cout << "th = " << th << endl;
    int nn = K.get_size_conektiv();    // „исло св€зей (размер массива св€зей)
    cout << "connect = " << nn << endl;
    cout << "Sozdal" << endl;
    K.filling_G_D();
    cout << "Zapolnil" << endl;
    cudaError_t cudaStatus;
    int* host_sosed;
    int* dev_sosed;
    double* host_T, * host_T_do, * host_TT;
    double* dev_T, * dev_T_do, * dev_TT;
    int* host_i;
    int* dev_i;

    // —оздаЄм массивы переменных\€чеек
    int* dev_l, * dev_r;
    double* dev_x, * dev_y, * dev_z;
    double* dev_dx, * dev_dy, * dev_dz;
    double* dev_ro1, * dev_p1, * dev_u1, * dev_v1, * dev_w1, * dev_ro2, * dev_p2, * dev_u2, * dev_v2, * dev_w2;
    double* host_x, * host_y, * host_z;
    double* host_dx, * host_dy, * host_dz;
    double* host_ro1, * host_p1, * host_u1, * host_v1, * host_w1, * host_bx1, * host_by1, * host_bz1;
    int* host_l, * host_r;

    host_T = (double*)malloc(sizeof(double));
    host_T_do = (double*)malloc(sizeof(double));
    host_TT = (double*)malloc(sizeof(double));
    host_i = (int*)malloc(sizeof(int));

    host_x = new double[N];
    host_y = new double[N];
    host_z = new double[N];
    host_dx = new double[N];
    host_dy = new double[N];
    host_dz = new double[N];
    host_ro1 = new double[N];
    host_p1 = new double[N];
    host_u1 = new double[N];
    host_v1 = new double[N];
    host_w1 = new double[N];
    host_bx1 = new double[N];
    host_by1 = new double[N];
    host_bz1 = new double[N];
    /*host_ro2 = new double[N];
    host_p2 = new double[N];
    host_u2 = new double[N];
    host_v2 = new double[N];
    host_w2 = new double[N];*/
    host_l = new int[N];
    host_r = new int[N];

    *host_T = 10000000.0;
    *host_T_do = 0.00000001;
    *host_TT = 0.0;
    *host_i = 0;

    //// ¬ыбор на каком GPU работаем (дл€ систем с несколькими GPU актуально)
    //cudaStatus = cudaSetDevice(0);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    //    goto Error;
    //}

    host_sosed = new int[nn];

    // «аполнение массивов
    int c = 0;
    for (Kyb* & i : K.all_Kyb)
    {
        for (Kyb* & j : i->sosed)
        {
            host_sosed[c] = j->number;
            c++;
        }
    }

    int ll = 0;
    int kkk = 0;
    int gg = 1;
    for (int i = 0; i < K.all_Kyb.size(); i++)
    {
        host_x[i] = K.all_Kyb[i]->x;
        host_y[i] = K.all_Kyb[i]->y;
        host_z[i] = K.all_Kyb[i]->z;
        host_dx[i] = K.all_Kyb[i]->dx;
        host_dy[i] = K.all_Kyb[i]->dy;
        host_dz[i] = K.all_Kyb[i]->dz;
        host_ro1[i] = K.all_Kyb[i]->ro;
        host_p1[i] = K.all_Kyb[i]->p;
        host_u1[i] = K.all_Kyb[i]->u;
        host_v1[i] = K.all_Kyb[i]->v;
        host_w1[i] = K.all_Kyb[i]->w;
        host_bx1[i] = K.all_Kyb[i]->Bx;
        host_by1[i] = K.all_Kyb[i]->By;
        host_bz1[i] = K.all_Kyb[i]->Bz;
        host_l[i] = ll;
        host_r[i] = ll + K.all_Kyb[i]->sosed.size() - 1;
        ll = ll + K.all_Kyb[i]->sosed.size();
    }

    cout << "Sozdal massivi Cuda" << endl;

    cout << "Pamyat = " << (N * 11 * sizeof(double) + nn * sizeof(int)) / 1000000000.0 << endl;

    if (true)
    {

        cudaStatus = cudaMalloc((void**)&dev_x, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 1!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_y, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 2!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_z, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 3!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_dx, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 4!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_dy, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 5!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_dz, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 6!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_ro1, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 7!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_ro2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 8!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_p1, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 9!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_p2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 10!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_u1, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 11!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_u2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 12!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_v1, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 13!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_v2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 14!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_w1, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 15!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_w2, N * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 16!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_l, N * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 17!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_r, N * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 18!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_sosed, nn * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 19!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_T, sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 20!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_T_do, sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 21!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_TT, sizeof(double));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 22!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_i, sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed 23!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_sosed, host_sosed, nn * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed -1 !");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_x, host_x, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 0!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_x, host_x, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 1!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_y, host_y, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 2!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_z, host_z, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 3!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_dx, host_dx, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 4!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_dy, host_dy, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 5!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_dz, host_dz, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 6!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_ro1, host_ro1, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 7!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_p1, host_p1, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 8!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_u1, host_u1, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 9!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_v1, host_v1, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 10!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_w1, host_w1, N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 11!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_l, host_l, N * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 12!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_r, host_r, N * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 13!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_T, host_T, sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 14!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_TT, host_TT, sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 15!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_T_do, host_T_do, sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 16!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(dev_i, host_i, sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed 17!");
            goto Error;
        }

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
    }

    //int kkk = 0;

     while (*host_TT < 1.97)  // —колько шагов по времени делаем?
    //for (int ii = 0; ii < 1000; ii++)
    {
        // запускаем add() kernel на GPU, передава€ параметры
        Cuda_main_HLL << <N / th, th >> > (dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
            dev_ro1, dev_ro2, dev_p1, dev_p2, dev_u1, dev_u2, dev_v1, dev_v2,//
            dev_w1, dev_w2, dev_sosed, dev_l, dev_r, dev_T, dev_T_do);
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 11111\n", cudaStatus);
            goto Error;
        }

        funk_time << <1, 1 >> > (dev_T, dev_T_do, dev_TT, dev_i);
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 222222\n", cudaStatus);
            goto Error;
        }

        Cuda_main_HLL << <N / th, th >> > (dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
            dev_ro2, dev_ro1, dev_p2, dev_p1, dev_u2, dev_u1, dev_v2, dev_v1,//
            dev_w2, dev_w1, dev_sosed, dev_l, dev_r, dev_T, dev_T_do);
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 333333\n", cudaStatus);
            goto Error;
        }

        funk_time << <1, 1 >> > (dev_T, dev_T_do, dev_TT, dev_i);
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 4444444\n", cudaStatus);
            goto Error;
        }

        cudaStatus = cudaMemcpy(host_TT, dev_TT, sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!  3452\n");
            goto Error;
        }
    }

     perekluch << <N / th, th >> > (dev_x, dev_y, dev_z, //
         dev_ro1, dev_p1, dev_u1, dev_v1,//
         dev_w1, dev_TT);

     *host_TT = 0.0;

     while (*host_TT < 0.394)  // —колько шагов по времени делаем?
    //for (int ii = 0; ii < 1000; ii++)
     {
         // запускаем add() kernel на GPU, передава€ параметры
         Cuda_main_HLL << <N / th, th >> > (dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
             dev_ro1, dev_ro2, dev_p1, dev_p2, dev_u1, dev_u2, dev_v1, dev_v2,//
             dev_w1, dev_w2, dev_sosed, dev_l, dev_r, dev_T, dev_T_do);
         cudaStatus = cudaDeviceSynchronize();
         if (cudaStatus != cudaSuccess) {
             fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 11111\n", cudaStatus);
             goto Error;
         }

         funk_time << <1, 1 >> > (dev_T, dev_T_do, dev_TT, dev_i);
         cudaStatus = cudaDeviceSynchronize();
         if (cudaStatus != cudaSuccess) {
             fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 222222\n", cudaStatus);
             goto Error;
         }

         Cuda_main_HLL << <N / th, th >> > (dev_x, dev_y, dev_z, dev_dx, dev_dy, dev_dz,//
             dev_ro2, dev_ro1, dev_p2, dev_p1, dev_u2, dev_u1, dev_v2, dev_v1,//
             dev_w2, dev_w1, dev_sosed, dev_l, dev_r, dev_T, dev_T_do);
         cudaStatus = cudaDeviceSynchronize();
         if (cudaStatus != cudaSuccess) {
             fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 333333\n", cudaStatus);
             goto Error;
         }

         funk_time << <1, 1 >> > (dev_T, dev_T_do, dev_TT, dev_i);
         cudaStatus = cudaDeviceSynchronize();
         if (cudaStatus != cudaSuccess) {
             fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! 4444444\n", cudaStatus);
             goto Error;
         }

         cudaStatus = cudaMemcpy(host_TT, dev_TT, sizeof(double), cudaMemcpyDeviceToHost);
         if (cudaStatus != cudaSuccess) {
             fprintf(stderr, "cudaMemcpy failed!  3452\n");
             goto Error;
         }

         if (*host_TT > 0.03 * gg)
         {
             gg++;
             cudaStatus = cudaMemcpy(host_ro1, dev_ro1, N * sizeof(double), cudaMemcpyDeviceToHost);
             if (cudaStatus != cudaSuccess) {
                 fprintf(stderr, "cudaMemcpy failed!  3452\n");
                 goto Error;
             }
             cudaStatus = cudaMemcpy(host_p1, dev_p1, N * sizeof(double), cudaMemcpyDeviceToHost);
             if (cudaStatus != cudaSuccess) {
                 fprintf(stderr, "cudaMemcpy failed!  3452\n");
                 goto Error;
             }
             cudaStatus = cudaMemcpy(host_u1, dev_u1, N * sizeof(double), cudaMemcpyDeviceToHost);
             if (cudaStatus != cudaSuccess) {
                 fprintf(stderr, "cudaMemcpy failed!  3452\n");
                 goto Error;
             }
             cudaStatus = cudaMemcpy(host_v1, dev_v1, N * sizeof(double), cudaMemcpyDeviceToHost);
             if (cudaStatus != cudaSuccess) {
                 fprintf(stderr, "cudaMemcpy failed!  3452\n");
                 goto Error;
             }
             cudaStatus = cudaMemcpy(host_w1, dev_w1, N * sizeof(double), cudaMemcpyDeviceToHost);
             if (cudaStatus != cudaSuccess) {
                 fprintf(stderr, "cudaMemcpy failed!  3452\n");
                 goto Error;
             }
             K.read_Cuda_massiv(host_ro1, host_p1, host_u1, host_v1, host_w1, host_bx1, host_by1, host_bz1, host_bz1);
             K.print_Tecplot_z(0.0, *host_TT);
             K.print_Tecplot_x(0.0, *host_TT);
             K.print_Tecplot_y(0.0, *host_TT);
         }
     }

    // cudaDeviceSynchronize ожидает завершени€ работы €дра и возвращает
    // любые ошибки, полученные в процессе выполнени€
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! dffddff\n", cudaStatus);
        goto Error;
    }

    //  опируем обратно на ’ост
    cudaStatus = cudaMemcpy(host_ro1, dev_ro1, N * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!  3452\n");
        goto Error;
    }
    cudaStatus = cudaMemcpy(host_p1, dev_p1, N * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!  3452\n");
        goto Error;
    }
    cudaStatus = cudaMemcpy(host_u1, dev_u1, N * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!  3452\n");
        goto Error;
    }
    cudaStatus = cudaMemcpy(host_v1, dev_v1, N * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!  3452\n");
        goto Error;
    }
    cudaStatus = cudaMemcpy(host_w1, dev_w1, N * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!  3452\n");
        goto Error;
    }

Error:
    cudaFree(dev_sosed);
    cudaFree(dev_ro1);
    cudaFree(dev_ro2);
    cudaFree(dev_p1);
    cudaFree(dev_p2);
    cudaFree(dev_u1);
    cudaFree(dev_u2);
    cudaFree(dev_v1);
    cudaFree(dev_v2);
    cudaFree(dev_w1);
    cudaFree(dev_w2);
    cudaFree(dev_T);
    cudaFree(dev_i);
    cudaFree(dev_T_do);
    cudaFree(dev_TT);

    K.read_Cuda_massiv(host_ro1, host_p1, host_u1, host_v1, host_w1, host_bx1, host_by1, host_bz1, host_bz1);
    K.print_Tecplot_z(0.0, *host_TT);
    K.print_Tecplot_x(0.0, *host_TT);
    K.print_Tecplot_y(0.0, *host_TT);
    /*K.print_Tecplot(150.0);
    K.print_Tecplot(240.0);
    K.print_Tecplot(320.0);*/
    K.save_Setka();

    return cudaStatus;
}

