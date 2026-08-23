#pragma once

#define krit 0.25
#define ggg (5.0/3.0)          // Показатель адиабаты
#define ga (5.0/3.0)          // Показатель адиабаты
#define g1 (ga - 1.0)
#define gg1 (ga - 1.0)
#define g2 (ga + 1.0)
#define gg2 (ga + 1.0)
#define gp ((g2/ga)/2.0)
#define gm ((g1/ga)/2.0)
#define gga ga
#define kv(x) ((x)*(x))
#define skk(u,v,w,bx,by,bz) ( (u)*(bx) + (v)*(by) + (w)*(bz) )
#define kvv(x,y,z)  (kv(x) + kv(y) + kv(z))
#define U8(ro, p, u, v, w, bx, by, bz)  ( (p) / (ggg - 1.0) + 0.5 * (ro) * kvv(u,v,w) + kvv(bx,by,bz) / 25.13274122871834590768)

#define pi 3.14159265358979323846
#define PI 3.14159265358979323846
#define cpi4 12.56637061435917295384
#define cpi8 25.13274122871834590768
#define spi4 ( 3.544907701811032 )

#define M_inf  0.0 // 0.7 // 0.4 // 0.8
#define M_infty  M_inf
#define phi_0  4.0 // 2.0 // 17.0 // 4.878 //1.627
#define alpha  45.0
#define M_alf  8.0 // 12.0
#define epsilon_ (1.0/M_alf)
#define step  70000
#define omega 0.0 //6 //1600
#define M_0  10.0

#define ddist 0.3 // 0.45 // 0.65 // 0.85

#define kk_ 196.0
//#define chi 17.0
#define rr_0  0.07    // Внутренняя сфера (на которой ставятся граничные условия - реально она дальше)


extern __device__ int sign_(const double& x);
extern __device__ double minmod_(double x, double y);
extern __device__ double linear_(double x1, double t1, double x2, double t2, double x3, double t3, double y);
extern __device__ void linear2_(double x1, double t1, double x2, double t2, double x3, double t3, double y1, double y2,//
    double& A, double& B);
extern __device__ void TVD(const double2& s_1, const double2& s_2, const double2& s_3, const double2& s_4, const double2& s_5,//
    const double2& s_6, const double2& s_7, const double2& s_8, const double2& s_9, double2& s12,//
    double2& s13, double2& s14, double2& s15, double2& s21, double2& s31, double2& s41, double2& s51, double dx, double dy, bool zero);
extern __device__ double HLLC_Korolkov_2D(const double2& Ls, const double2& Lu, const double2& Rs, const double2& Ru,//
    const double n1, const double n2, double2& Ps, double2& Pu, const double rad);
extern __device__ double HLLCQ_Korolkov_2D(const double2& Ls, const double2& Lu, const double2& Rs, const double2& Ru,//
    const double& LQ, const double& RQ, double n1, double n2, double2& Ps, double2& Pu, double& PQ, double rad);
extern __device__ double HLLCQ_Aleksashov(const double2& Ls, const double2& Lu, const double2& Rs, const double2& Ru,//
    const double& LQ, const double& RQ, double n1, double n2, double2& Ps, double2& Pu, double& PQ, double rad);
extern __device__ double HLLC_Aleksashov_2D(double2& Ls, double2& Lu, double2& Rs, double2& Ru,//
    double n1, double n2, double2& Ps, double2& Pu, double rad);
__device__ double HLLDQ_Korolkov(const double& ro_L, const double& Q_L, const double& p_L, const double& v1_L, const double& v2_L, const double& v3_L,//
    const double& Bx_L, const double& By_L, const double& Bz_L, const double& ro_R, const double& Q_R, const double& p_R, const double& v1_R, const double& v2_R, const double& v3_R,//
    const double& Bx_R, const double& By_R, const double& Bz_R, double* P, double& PQ, const double& n1, const double& n2, const double& n3, const double& rad, int metod, double x = 0.0, double y = 0.0);
__device__ double HLLDQ_Korolkov2(const double& ro_L, const double& Q_L, const double& p_L, const double& v1_L, const double& v2_L, const double& v3_L,//
    const double& Bx_L, const double& By_L, const double& Bz_L, const double& ro_R, const double& Q_R, const double& p_R, const double& v1_R, const double& v2_R, const double& v3_R,//
    const double& Bx_R, const double& By_R, const double& Bz_R, double* P, double& PQ, const double& n1, const double& n2, const double& n3, const double& rad, int metod, double x = 0.0, double y = 0.0);

extern __device__ double POTOK_Korolkov(const double& ro_L, const double& Q_L, const double& p_L, const double& v1_L, const double& v2_L, const double& v3_L,//
    const double& Bx_L, const double& By_L, const double& Bz_L, double* P, const double& n1, const double& n2, const double& n3);