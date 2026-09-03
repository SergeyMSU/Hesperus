#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>
#include <vector>
#include <string>
#include "Header.h"

#define Omega 0.0
#define N 350 // 7167 //1792 //1792                 // Количество ячеек по x
#define M 256 // 256 // //1280 //1280                 // Количество ячеек по y
#define K (N*M)                // Количество ячеек в сетке
#define Rb (6.0)             // Внешний радиус сетки
#define qb (1.02)            // Сгущение сетки каждая следующая ширина на (qb - 1)% больше предыдущей
#define dphi (pi/M)           
#define qphi (1.02)              // можно настроить; при большом M брать близким к 1
#define M_HALF (M / 2)           // предполагаем, что M чётное
#define BX 16   // Размеры блока потоков 
#define BY 16   // Размеры блока потоков 
#define HALO 2
#define SHARED_I (BX + 2*HALO)   // = 20
#define SHARED_J (BY + 2*HALO)   // = 20

// Предполагается, что индексы ячеек i (по радиусу) и j (по углу) отсчитываются от 0
// 
// ----------------- Радиальное разбиение -----------------

// Ширина первой радиальной ячейки (аналитически из суммы геометрической прогрессии)
#define DR1 ((Rb - 1.0) * (qb - 1.0) / (pow(qb, N) - 1.0))

// Радиус i-й границы (i = 0..N)
#define R_EDGE(i) (1.0 + DR1 * (pow(qb, (i)) - 1.0) / (qb - 1.0))

// Ширина i-й радиальной ячейки (i = 0..N-1)
#define DR(i) (DR1 * pow(qb, (i)))

// Радиус центра i-й ячейки (i = 0..N-1)
//#define R_CENTER(i) ( (2.0/3.0) * \
//                        (pow(R_EDGE((i)+1), 3) - pow(R_EDGE(i), 3)) / \
//                        (pow(R_EDGE((i)+1), 2) - pow(R_EDGE(i), 2)) * \
//                        (sin(0.5 * dphi) / (0.5 * dphi)) )  // равномерный угол
#define R_CENTER(i,j) ( (2.0/3.0) * \
                        (pow(R_EDGE((i)+1), 3) - pow(R_EDGE(i), 3)) / \
                        (pow(R_EDGE((i)+1), 2) - pow(R_EDGE(i), 2)) * \
                        (sin(0.5 * DPHI(j)) / (0.5 * DPHI(j))) )
//#define R_CENTER(i) (0.5 * (R_EDGE(i) + R_EDGE(i + 1)))

// ----------------- Угловое разбиение равномерное -----------------

//// Угловая граница с индексом k (k = 0..M): phi = -pi/2 + k * dphi
//#define PHI_EDGE(k) (-(pi) / 2.0 + (k) * dphi)
//
//// Нижняя граница j-й ячейки (j = 0..M-1)
//#define PHI_LEFT(j) (PHI_EDGE(j))
//
//// Верхняя граница j-й ячейки (j = 0..M-1)
//#define PHI_RIGHT(j) (PHI_EDGE((j) + 1))
//
//// Центр j-й ячейки (как было ранее, можно оставить)
//#define PHI_CENTER(j) (0.5 * (PHI_LEFT(j) + PHI_RIGHT(j)))

//// Внутренняя дуга (при r = R_EDGE(i))
//#define L_INNER(i) (R_EDGE(i) * dphi)

//// Внешняя дуга (при r = R_EDGE(i+1))
//#define L_OUTER(i) (R_EDGE(i + 1) * dphi)

// ----------------- Угловое разбиение неравномерное ----------------------------------------

// Минимальный угловой шаг (у phi = 0)
#define DPHI_MIN ((pi / 2.0) * (qphi - 1.0) / (pow(qphi, M_HALF) - 1.0))

// Ширина j-й угловой ячейки (j = 0..M-1)
#define DPHI(j) ((j) < M_HALF ? \
                 DPHI_MIN * pow(qphi, M_HALF - 1 - (j)) : \
                 DPHI_MIN * pow(qphi, (j) - M_HALF))

// Угловая граница с индексом k (k = 0..M)
#define PHI_EDGE(k) ((k) <= M_HALF ? \
                     (-(pi) / 2.0 + DPHI_MIN * (pow(qphi, M_HALF) - pow(qphi, M_HALF - (k))) / (qphi - 1.0)) : \
                     (DPHI_MIN * (pow(qphi, (k) - M_HALF) - 1.0) / (qphi - 1.0)))

// Нижняя и верхняя границы j-й ячейки
#define PHI_LEFT(j)  (PHI_EDGE(j))
#define PHI_RIGHT(j) (PHI_EDGE((j) + 1))

// Центр j-й ячейки по углу (средний угол)
#define PHI_CENTER(j) (0.5 * (PHI_LEFT(j) + PHI_RIGHT(j)))

// Внутренняя дуга (при r = R_EDGE(i))
#define L_INNER(i,j) (R_EDGE(i) * DPHI(j))

// Внешняя дуга (при r = R_EDGE(i+1))
#define L_OUTER(i,j) (R_EDGE(i + 1) * DPHI(j))

// ---------------------------------------------------------------------------------------------------------------------------

// Левый и правый радиальные отрезки (равны DR(i))
#define L_LEFT(i) (DR(i))
#define L_RIGHT(i) (DR(i))

// ----------------- Площадь (объём) ячейки -----------------

//#define CELL_AREA(i) (0.5 * (R_EDGE(i + 1) * R_EDGE(i + 1) - R_EDGE(i) * R_EDGE(i)) * dphi)   // равномерный угол
#define CELL_AREA(i,j) (0.5 * (R_EDGE(i + 1) * R_EDGE(i + 1) - R_EDGE(i) * R_EDGE(i)) * DPHI(j))   // неравномерный угол


#define const_p 0.000186401  // (0.000447362)     // p = const_p * rho
//#define rho_in 0.8 // (0.220637)     // p = const_p * rho
#define rho_in 0.45 // (0.220637)     // p = const_p * rho

#define F_grav (-0.187168)           // Коэффициент перед силой гравитации
#define F_continuum (0.0129046)     // Коэффициент перед силой радиационного давления (континуума)
#define F_line (0.067358)     // Коэффициент внутри line-driven силы
//#define alpha_line (0.752342)      // Коэффициент внутри line-driven силы
//#define k_line (0.00587879)      // Коэффициент внутри line-driven силы

#define alpha_line (0.44) //(0.752342) //(0.5)      // Коэффициент внутри line-driven силы

#define Bo_init 0.45542// 1.53551  // (15.0 * 0.00314065) //(15.0 * 0.00314065) // 0.06 (0.00587879) // (0.108238)    
#define phi_init (0.785409) // 0.582751 // (pi/2.0) // 0.797285  // смена гран условий по углу

#define V_phi_init 0.0  // (0.266667)   //   Скорость вращения звезды


#define Bx_dipole(r, phi) ( (3.0/2.0) * Bo_init * sin(phi) * cos(phi) / ((r)*(r)*(r)) )
#define By_dipole(r, phi) ( Bo_init * ( sin(phi)*sin(phi) - 0.5*cos(phi)*cos(phi) ) / ((r)*(r)*(r)) )

// Освободить хост-память (если она больше не нужна после копирования)
#define FREE_HOST(ptr) delete[] ptr;

// Освободить девайс-память
#define FREE_DEVICE(ptr) cudaFree(ptr);

#define DERIVATIVE(f_left, f_center, f_right, hL, hR) \
    ( - (hR) / ((hL) * ((hL) + (hR))) * (f_left) \
      + ((hR) - (hL)) / ((hL) * (hR)) * (f_center) \
      + (hL) / ((hR) * ((hL) + (hR))) * (f_right) )

#define x_max 6.0 //450.0
#define x_min (x_max/(2.0 * N)) // -2760.0 // -2500.0 // -1300  //-2000                // -1500.0
#define y_max 6.0 // 2250.0 // 1600.0 //1840.0
#define y_min -6.0 // (y_max/(2.0 * M))  // -30.0 // (y_max/(2.0 * M)) 
#define dx ((x_max)/(N))  // ((x_max - x_min)/(N - 1))     // Величина грани по dx
#define dy ((y_max - y_min)/(M)) //  ((y_max - y_min)/(M - 1))     // Величина грани по dy

#define ER_S std::cout << "\n---------------------\nStandart error in file: Solvers.cpp\n" << endl
#define watch(x) cout << (#x) << " is " << (x) << endl
#define eps (1e-10)
#define eps8 (1e-8)
#define hy 00.0
#define hx -3288.0
#define grad_p true
#define Nmin 3              // Каждую какую точку выводим?
#define THREADS_PER_BLOCK 256    // Количество нитей в одном потоке // Необходимо, чтобы количество ячеек в сетке делилось на число нитей (лучше N делилось на число нитей)

__host__ __device__ int sign(double& x);
__device__ double minmod(double x, double y);
__device__ double linear(double x1, double t1, double x2, double t2, double x3, double t3, double y);
__device__ void linear2(double x1, double t1, double x2, double t2, double x3, double t3, double y1, double y2,//
    double& A, double& B);

using namespace std;

// Переменные в центрах ячеек
struct CellVars {
    double* rho, * Vx, * Vy, * Vz, * Bx, * By, * Bz;
};

// Переменные на гранях (потоки или другие величины)
struct FaceVars {
    double* Prho, * Pvx, * Pvy, * Pvz, * Pbx, * Pby, * Pbz, * Bn;
};

// Переменные в узлах (например, электрическое поле)
struct NodeVars {
    double* Ez;
};

template<typename T>
T* allocateHost(int count) {
    T* ptr = new T[count];
    for (int i = 0; i < count; ++i) ptr[i] = 0.0;
    return ptr;
}

template<typename T>
T* allocateDevice(int count) {
    T* ptr = nullptr;
    cudaMalloc(&ptr, count * sizeof(T));
    return ptr;
}

template<typename T>
void copyToDevice(T* d_ptr, const T* h_ptr, int count) {
    cudaMemcpy(d_ptr, h_ptr, count * sizeof(T), cudaMemcpyHostToDevice);
}

__host__ __device__ double minmod(double x, double y)
{
    if (sign(x) + sign(y) == 0)
    {
        return 0.0;
    }
    else
    {
        return   ((sign(x) + sign(y)) / 2.0) * min(fabs(x), fabs(y));  ///minmod
        //return (2*x*y)/(x + y);   /// vanleer
    }
}

__device__ double VanLier(double x, double y)
{
    return minmod((x + y)/2.0, 2.0 * minmod(x, y));
}

__host__ __device__ double linear(double x1, double t1, double x2, double t2, double x3, double t3, double y)
{
    double d = minmod((t1 - t2) / (x1 - x2), (t2 - t3) / (x2 - x3));
    return  (d * (y - x2) + t2);
}

__device__ void linear2(double x1, double t1, double x2, double t2, double x3, double t3, double y1, double y2,//
    double& A, double& B)
{
    // ГЛАВНОЕ ЗНАЧЕНИЕ - ЦЕНТРАЛЬНОЕ - НЕ ЗАБЫВАЙ ОБ ЭТОМ
    double d = minmod((t1 - t2) / (x1 - x2), (t2 - t3) / (x2 - x3));
    A = (d * (y1 - x2) + t2);
    B = (d * (y2 - x2) + t2);
    //printf("%lf | %lf | %lf | %lf | %lf | %lf | %lf | %lf | %lf | %lf \n", x1, t1, x2, t2, x3, t3, y1, y2, A, B);
    return;
}

__device__ int sign(double& x)
{
    if (x > 0)
    {
        return 1;
    }
    else if (x < 0)
    {
        return  -1;
    }
    else
    {
        return 0;
    }
}

__device__ double  my_min(double a, double b)
{
    if (a <= b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

__device__ double  my_max(double a, double b)
{
    if (a >= b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

double polar_angle(double x, double y)
{
    if (x * x + y * y < 0.0000001)
    {
        return 0.0;
    }
    else if (x < 0)
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

void dekard_skorost(double x, double y, double z, double Vr, double Vphi, double Vtheta, double& Vx, double& Vy, double& Vz)
{
    double r_2 = sqrt(x * x + y * y + z * z);
    double the_2 = acos(z / r_2);
    double phi_2 = polar_angle(x, y);

    if (sqrt(x * x + y * y) < 0.000001)
    {
        Vx = 0.0;
        Vy = 0.0;
        Vz = 0.0;
    }
    else
    {
        Vx = Vr * sin(the_2) * cos(phi_2) + Vtheta * cos(the_2) * cos(phi_2) - Vphi * sin(phi_2);
        Vy = Vr * sin(the_2) * sin(phi_2) + Vtheta * cos(the_2) * sin(phi_2) + Vphi * cos(phi_2);
        Vz = Vr * cos(the_2) - Vtheta * sin(the_2);
    }
}

__device__ void predictor(const double *Q, const double* DX, const double* DY, double* QQ, double step_time, double x, double y)
{
    double P[7];
    double2 PS = { 0.0, 0.0 };
    double2 PU = { 0.0, 0.0 };
    double3 PB = { 0.0, 0.0, 0.0 };
    double Pdiv = 0.0;
    double ro_, p_;

    ro_ = Q[0] - DX[0] / 2.0;
    p_ = Q[1] - DX[1] / 2.0;
    if (ro_ <= 0.0)
    {
        ro_ = Q[0];
    }
    if (p_ <= 0.0)
    {
        p_ = Q[1];
    }
    POTOK_Korolkov(ro_, 0.0, p_, Q[2] - DX[2] / 2.0, Q[3] - DX[3] / 2.0, 0.0,//
        Q[4] - DX[4] / 2.0, Q[5] - DX[5] / 2.0, Q[6] - DX[6] / 2.0, P, -1.0, 0.0, 0.0);
    PS.x = PS.x + P[0] * dy;
    PS.y = PS.y + P[7] * dy;
    PU.x = PU.x + P[1] * dy;
    PU.y = PU.y + P[2] * dy;
    PB.x = PB.x + P[4] * dy;
    PB.y = PB.y + P[5] * dy;
    PB.z = PB.z + P[6] * dy;

    ro_ = Q[0] + DX[0] / 2.0;
    p_ = Q[1] + DX[1] / 2.0;
    if (ro_ <= 0.0)
    {
        ro_ = Q[0];
    }
    if (p_ <= 0.0)
    {
        p_ = Q[1];
    }
    POTOK_Korolkov(ro_, 0.0, p_, Q[2] + DX[2] / 2.0, Q[3] + DX[3] / 2.0, 0.0,//
        Q[4] + DX[4] / 2.0, Q[5] + DX[5] / 2.0, Q[6] + DX[6] / 2.0, P, 1.0, 0.0, 0.0);
    PS.x = PS.x + P[0] * dy;
    PS.y = PS.y + P[7] * dy;
    PU.x = PU.x + P[1] * dy;
    PU.y = PU.y + P[2] * dy;
    PB.x = PB.x + P[4] * dy;
    PB.y = PB.y + P[5] * dy;
    PB.z = PB.z + P[6] * dy;

    ro_ = Q[0] + DY[0] / 2.0;
    p_ = Q[1] + DY[1] / 2.0;
    if (ro_ <= 0.0)
    {
        ro_ = Q[0];
    }
    if (p_ <= 0.0)
    {
        p_ = Q[1];
    }
    POTOK_Korolkov(ro_, 0.0, p_, Q[2] + DY[2] / 2.0, Q[3] + DY[3] / 2.0, 0.0,//
        Q[4] + DY[4] / 2.0, Q[5] + DY[5] / 2.0, Q[6] + DY[6] / 2.0, P, 0.0, 1.0, 0.0);
    PS.x = PS.x + P[0] * dx;
    PS.y = PS.y + P[7] * dx;
    PU.x = PU.x + P[1] * dx;
    PU.y = PU.y + P[2] * dx;
    PB.x = PB.x + P[4] * dx;
    PB.y = PB.y + P[5] * dx;
    PB.z = PB.z + P[6] * dx;

    ro_ = Q[0] - DY[0] / 2.0;
    p_ = Q[1] - DY[1] / 2.0;
    if (ro_ <= 0.0)
    {
        ro_ = Q[0];
    }
    if (p_ <= 0.0)
    {
        p_ = Q[1];
    }
    POTOK_Korolkov(ro_, 0.0, p_, Q[2] - DY[2] / 2.0, Q[3] - DY[3] / 2.0, 0.0,//
        Q[4] - DY[4] / 2.0, Q[5] - DY[5] / 2.0, Q[6] - DY[6] / 2.0, P, 0.0, -1.0, 0.0);
    PS.x = PS.x + P[0] * dx;
    PS.y = PS.y + P[7] * dx;
    PU.x = PU.x + P[1] * dx;
    PU.y = PU.y + P[2] * dx;
    PB.x = PB.x + P[4] * dx;
    PB.y = PB.y + P[5] * dx;
    PB.z = PB.z + P[6] * dx;

    double dV = dx * dy;

    QQ[0] = Q[0] - step_time * PS.x / dV - step_time * Q[0] * Q[3] / y;
    if (QQ[0] <= 0)
    {
        printf("Problemsssss 84745377! x = %lf, y = %lf, ro = %lf, T = %lf, ro = %lf \n", x, y, QQ[0], step_time, Q[0]);
        QQ[0] = Q[0];
    }
    QQ[2] = (Q[0] * Q[2] - step_time * (PU.x) / dV - step_time * (Q[0] * Q[2] * Q[3] - Q[4] * Q[5] / cpi4) / y) / QQ[0];
    QQ[3] = (Q[0] * Q[3] - step_time * (PU.y) / dV - step_time * (Q[0] * Q[3] * Q[3] + (kv(Q[6]) - kv(Q[5])) / cpi4) / y) / QQ[0];
    QQ[4] = (Q[4] - step_time * (PB.x) / dV - step_time * (Q[3] * Q[4] - Q[5] * Q[2]) / y);
    QQ[5] = (Q[5] - step_time * (PB.y) / dV);
    QQ[6] = (Q[6] - step_time * (PB.z) / dV);
    QQ[1] = (U8(Q[0], Q[1], Q[2], Q[3], 0.0, Q[4], Q[5], Q[6]) - step_time * (PS.y)//
        / dV - step_time * (((U8(Q[0], Q[1], Q[2], Q[3], 0.0, Q[4], Q[5], Q[6]) + Q[1] + kvv(Q[4], Q[5], Q[6]) / cpi8) * Q[3] - Q[5] * skk(Q[2], Q[3], 0.0, Q[4], Q[5], Q[6]) / cpi4) / y) //
        - 0.5 * QQ[0] * kvv(QQ[2], QQ[3], 0.0) - kvv(QQ[4], QQ[5], QQ[6]) / cpi8) * (ggg - 1.0);
    if (QQ[1] <= 0)
    {
        QQ[1] = 0.000001;
    }
}

__global__ void funk_time(double* T, double* T_do, double* TT, int* i, double* ch, double* ch_posle)
{
    *T_do = *T;
    *ch = *ch_posle;
    *TT = *TT + *T_do;
    *T = 10000000;
    *ch_posle = 0.0;
    *i = *i + 1;
    if (*i % 10000 == 0)
    {
        printf("i = %d,  TT all = %lf, hours = %lf; dT hours = %E, %E \n", *i, *TT, *TT * 1.09556, *T_do * 1.09556, *ch);
    }
    return;
}

__device__ double atomicMinDouble(double* address, double val) 
{
    unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *addr_as_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double new_val = (val < old_val) ? val : old_val;
        unsigned long long new_val_ull = __double_as_longlong(new_val);
        old = atomicCAS(addr_as_ull, assumed, new_val_ull);
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double atomicMaxDouble(double* address, double val)
{
    // Приводим указатель к типу unsigned long long для работы с atomicCAS
    unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(address);
    // Читаем текущее значение как битовый образ
    unsigned long long old = *addr_as_ull;
    unsigned long long assumed;

    do {
        assumed = old;
        // Преобразуем биты обратно в double
        double old_val = __longlong_as_double(assumed);
        // Выбираем большее из двух значений
        double new_val = (val > old_val) ? val : old_val;
        // Конвертируем новое значение в битовое представление
        unsigned long long new_val_ull = __double_as_longlong(new_val);
        // Пытаемся атомарно заменить, если значение не изменилось
        old = atomicCAS(addr_as_ull, assumed, new_val_ull);
    } while (assumed != old); // Повторяем, если за время чтения значение было изменено

    // Возвращаем старое значение (в double)
    return __longlong_as_double(old);
}

__global__ void add2_TVD(double3* s, double3* u, double3* b, double3* s2, double3* u2, double3* b2, double* T, double* T_do, double* ch, double* ch_posle, int i, int method)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;   // Глобальный индекс текущей ячейки (текущего потока)
    int n = index % N;                                   // номер ячейки по x (от 0)
    int m = (index - n) / N;                             // номер ячейки по y (от 0)

    double r = R_CENTER(n, m);
    //double r = R_CENTER(n);
    double phi = PHI_CENTER(m);

    double y = r * sin(phi);
    double x = r * cos(phi);


    double3 s_1, s_2, s_3, s_4, s_5;      // Переменные всех соседей и самой ячейки
    double3 u_1, u_2, u_3, u_4, u_5;      
    double3 b_1, b_2, b_3, b_4, b_5;
    double3 s_21, s_31, s_41, s_51;      
    double3 u_21, u_31, u_41, u_51;      
    double3 b_21, b_31, b_41, b_51;
    double2 Ps12 = { 0,0 }, Pu12 = { 0,0 }, Ps13 = { 0,0 }, Pu13 = { 0,0 }, //
        Ps14 = { 0,0 }, Pu14 = { 0,0 }, Ps15 = { 0,0 }, Pu15 = { 0,0 }; // Вектора потоков
    double3 Pb12 = { 0.0, 0.0, 0.0 }, Pb13 = { 0.0, 0.0, 0.0 }, Pb14 = { 0.0, 0.0, 0.0 }, Pb15 = { 0.0, 0.0, 0.0 };
    double tmin = 1000;
    double P[8];
    P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;

    if (index < 0 || index > N * M - 1)
    {
        printf("Error index = %d \n", index);
    }

    s_1 = s[index];
    u_1 = u[index];
    b_1 = b[index];

    double r2, r3, r4, r5, phi2, phi3, phi4, phi5;
    double r21, r31, r41, r51, phi21, phi31, phi41, phi51;


    // Берём параметры соседей и задаём граничные условия
    if ((m == M - 1))
    {
        r5 = r;
        phi5 = pi / 2.0;
        // Левая граница (сверху над сферой)

        // симметрия
        s_5 = s_1;
        u_5 = u_1;
        b_5 = b_1;
        u_5.x = 0.0;
        u_5.z = 0.0;
        b_5.x = 0.0;
        b_5.z = 0.0;
    }
    else
    {
        s_5 = s[(m + 1) * N + n];
        u_5 = u[(m + 1) * N + n];
        b_5 = b[(m + 1) * N + n];
        r5 = R_CENTER(n, m + 1);
        phi5 = PHI_CENTER(m + 1);
    }

    if ((m <= M - 3))
    {
        s_51 = s[(m + 2) * N + n];
        u_51 = u[(m + 2) * N + n];
        b_51 = b[(m + 2) * N + n];
        r51 = R_CENTER(n, m + 2);
        phi51 = PHI_CENTER(m + 2);
    }

    if ((n == N - 1))
    {
        r2 = Rb;
        phi2 = phi;
        // крайняя ячейка справа области

        // мягкие условия
        s_2 = s_1;
        u_2 = u_1;
        b_2 = b_1;

        // Fildmeier
        //double Vphi = u_1.x * sin(phi) + u_1.y * cos(phi);
        //double Vr = 1.0;

        //u_2.x = (Vr * cos(phi) - Vphi * sin(phi));
        //u_2.y = (Vr * sin(phi) + Vphi * cos(phi));
    }
    else
    {
        s_2 = s[(m)*N + n + 1];
        u_2 = u[(m)*N + n + 1];
        b_2 = b[(m)*N + n + 1];
        r2 = R_CENTER(n + 1, m);
        //r2 = R_CENTER(n + 1);
        phi2 = phi;
    }

    if ((n <= N - 3))
    {
        s_21 = s[(m)*N + n + 2];
        u_21 = u[(m)*N + n + 2];
        b_21 = b[(m)*N + n + 2];
        r21 = R_CENTER(n + 2, m);
        //r21 = R_CENTER(n + 2);
        phi21 = phi;
    }

    if (n == 0)
    {
        r4 = 1.0;
        phi4 = phi;

        double x4, y4;
        x4 = 1.0 * cos(phi);
        y4 = 1.0 * sin(phi);
        // крайняя ячейка слева области

        double Vr;
        // double Vr = (u_1.x * x + u_1.y * y) / r;   // Попробуем снести скорость мягко (можно потом попробовать вторым порядком даже)
        // 
        // 
        // линейный снос Vr
        double Vr1 = (u_1.x * x + u_1.y * y) / r;
        double Vr2 = u_2.x * cos(phi2) + u_2.y * sin(phi2);
        Vr = Vr1 + (Vr2 - Vr1) / (r2 - r) * (r4 - r);


        if (Vr < 0.000001) Vr = Vr1;
        if (Vr <= 0.0) Vr = 0.0;

        double Vphi = 0.0;

        //Vthe = u_1.x * sin(phi) + u_1.y * cos(phi);

        // Если поле сильное, нужно сносить Vthe
        //if (kvv(b_1.x, b_1.y, b_1.z) / (8.0 * pi) / (0.5 * s_1.x * kvv(u_1.x, u_1.y, u_1.z)) > 1.0)
        //{
        //    Vthe = u_1.x * sin(phi) + u_1.y * cos(phi);
        //}

        double vphi_ = V_phi_init * sin(pi / 2.0 - phi);

        double rho_0 = rho_in; // / 5.0;// v_in / Vr;

        s_4 = { rho_0, const_p * rho_0 };
        s_4.z = s_1.z;

        //u_4 = { 0.0, 0.0, 0.0 };
        //s_4 = { 1.0, 1.0};

        b_4.z = b_1.z;

        double Br1 = b_1.x * cos(phi) + b_1.y * sin(phi);
        double Br = kv(r) * (Br1 + Bo_init * cos(pi / 2.0 - phi) * pow(1.0 / r, 2.0)) - Bo_init * cos(pi / 2.0 - phi);
        //double Br = Br1 + Bo_init * cos(pi / 2.0 - phi) * (-1.0 + kv(1.0/r));
        //double Br = 0.0;


        double Bphi1 = -b_1.x * sin(phi) + b_1.y * cos(phi);
        double Bphi2 = -b_2.x * sin(phi) + b_2.y * cos(phi);
        double Bphi = Bphi1 + (Bphi2 - Bphi1) / (r2 - r) * (r4 - r);
        //Bphi = 0.0;

        //double Bphi = -Bo_init/2.0 * sin(pi / 2.0 - phi);

        b_4.x = (Br * cos(phi) - Bphi * sin(phi));
        b_4.y = (Br * sin(phi) + Bphi * cos(phi));

        double Br_dipole = Bo_init * cos(pi / 2.0 - phi);
        double Bphi_dipole = -Bo_init / 2.0 * sin(pi / 2.0 - phi);


        if (fabs(phi) > phi_init && fabs(Br + Br_dipole) > 0.0000001)
        {
            Vphi = Vr * (Bphi + Bphi_dipole) / (Br + Br_dipole);
        }

        u_4.x = (Vr * cos(phi) - Vphi * sin(phi));
        u_4.y = (Vr * sin(phi) + Vphi * cos(phi));
        u_4.z = vphi_;
    }
    else
    {
        s_4 = s[(m)*N + n - 1];
        u_4 = u[(m)*N + n - 1];
        b_4 = b[(m)*N + n - 1];
        r4 = R_CENTER(n - 1, m);
        //r4 = R_CENTER(n - 1);
        phi4 = phi;
    }

    if (n >= 2)
    {
        s_41 = s[(m)*N + n - 2];
        u_41 = u[(m)*N + n - 2];
        b_41 = b[(m)*N + n - 2];
        r41 = R_CENTER(n - 2, m);
        //r41 = R_CENTER(n - 2);
        phi41 = phi;
    }

    if ((m == 0))
    {
        r3 = r;
        phi3 = -pi / 2.0;
        // симметрия
        s_3 = s_1;
        u_3 = u_1;
        b_3 = b_1;
        u_3.x = 0.0;
        u_3.z = 0.0;
        b_3.x = 0.0;
        b_3.z = 0.0;
    }
    else
    {
        s_3 = s[(m - 1) * N + (n)];
        u_3 = u[(m - 1) * N + (n)];
        b_3 = b[(m - 1) * N + (n)];
        r3 = R_CENTER(n, m - 1);
        phi3 = PHI_CENTER(m - 1);
    }

    if (m >= 2)
    {
        s_31 = s[(m - 2) * N + (n)];
        u_31 = u[(m - 2) * N + (n)];
        b_31 = b[(m - 2) * N + (n)];
        r31 = R_CENTER(n, m - 2);
        phi31 = PHI_CENTER(m - 2);
    }


    double Q = 1.0;
    double PQ = 0.0;
    double3 PS = { 0.0, 0.0, 0.0 };
    double3 PU = { 0.0, 0.0, 0.0 };
    double3 PB = { 0.0, 0.0, 0.0 };
    double Pdiv = 0.0;

    double r_g, phi_g, x_g, y_g;   // r и phi грани
    double n1, n2;
    double rho_L, rho_R;
    double psi_L, psi_R;
    double p_L, p_R;
    double Vr, Vphi;
    double Br, Bphi;
    double u_L, v_L, u_R, v_R, bx_L, by_L, bx_R, by_R;
    double Bx_dipole_, By_dipole_;

    double Vr1, Vr2, Vr4, dr2, dr4;
    dr2 = r2 - r;
    dr4 = r - r4;


    double ch_now = 2.0 * *ch;
    double ch_max = 0.0;
    

    // Перед распадом надо определить нормаль к грани и разложить все вектора (скорости и магнитного поля) по этой нормали
    // Также предлагаю снести на грань с двух сторон все значения в полярной системе координат
    if (true)
    {
        double Vr_L, Vphi_L, Vr_R, Vphi_R, w_L, w_R;
        double Br_L, Bphi_L, Br_R, Bphi_R, bz_L, bz_R;
        double Vr3, Vr5, Vr31, Vr51, Vr21, Vr41;
        double Br1, Br2, Br4, Br3, Br5, Br31, Br51, Br21, Br41;
        double Vphi1, Vphi2, Vphi4, Vphi3, Vphi5, Vphi31, Vphi51, Vphi21, Vphi41;
        double Bphi1, Bphi2, Bphi4, Bphi3, Bphi5, Bphi31, Bphi51, Bphi21, Bphi41;

        double ch_max_ = 0.0;

        Vr1 = u_1.x * cos(phi) + u_1.y * sin(phi);
        Vr2 = u_2.x * cos(phi2) + u_2.y * sin(phi2);
        Vr3 = u_3.x * cos(phi3) + u_3.y * sin(phi3);
        Vr4 = u_4.x * cos(phi4) + u_4.y * sin(phi4);
        Vr5 = u_5.x * cos(phi5) + u_5.y * sin(phi5);

        Vphi1 = -u_1.x * sin(phi) + u_1.y * cos(phi);
        Vphi2 = -u_2.x * sin(phi2) + u_2.y * cos(phi2);
        Vphi3 = -u_3.x * sin(phi3) + u_3.y * cos(phi3);
        Vphi4 = -u_4.x * sin(phi4) + u_4.y * cos(phi4);
        Vphi5 = -u_5.x * sin(phi5) + u_5.y * cos(phi5);

        Br1 = b_1.x * cos(phi) + b_1.y * sin(phi);
        Br2 = b_2.x * cos(phi2) + b_2.y * sin(phi2);
        Br3 = b_3.x * cos(phi3) + b_3.y * sin(phi3);
        Br4 = b_4.x * cos(phi4) + b_4.y * sin(phi4);
        Br5 = b_5.x * cos(phi5) + b_5.y * sin(phi5);

        Bphi1 = -b_1.x * sin(phi) + b_1.y * cos(phi);
        Bphi2 = -b_2.x * sin(phi2) + b_2.y * cos(phi2);
        Bphi3 = -b_3.x * sin(phi3) + b_3.y * cos(phi3);
        Bphi4 = -b_4.x * sin(phi4) + b_4.y * cos(phi4);
        Bphi5 = -b_5.x * sin(phi5) + b_5.y * cos(phi5);


        
        // r+ грань
        if (true)
        {
            n1 = x / r;
            n2 = y / r;
            r_g = R_EDGE(n + 1);
            phi_g = phi;
            x_g = r_g * cos(phi_g);
            y_g = r_g * sin(phi_g);

            if (r_g > r2 || r_g < r4)
            {
                printf("Problems rr:  %lf, %lf, %lf, %d, %d \n", r_g, r2, r4, n, m);
            }


            rho_L = linear(r4, s_4.x * kv(r4), r, s_1.x * kv(r), r2, s_2.x * kv(r2), r_g) / kv(r_g);
            if (rho_L <= 0.0) rho_L = s_1.x;
            p_L = const_p * rho_L;

            psi_L = linear(r4, s_4.z, r, s_1.z, r2, s_2.z, r_g);

            w_L = linear(r4, u_4.z, r, u_1.z, r2, u_2.z, r_g);
            bz_L = linear(r4, b_4.z, r, b_1.z, r2, b_2.z, r_g);

            Vr_L = linear(r4, Vr4, r, Vr1, r2, Vr2, r_g);
            Vphi_L = linear(r4, Vphi4, r, Vphi1, r2, Vphi2, r_g);
            u_L = Vr_L * cos(phi_g) - Vphi_L * sin(phi_g);
            v_L = Vr_L * sin(phi_g) + Vphi_L * cos(phi_g);

            Br_L = linear(r4, Br4, r, Br1, r2, Br2, r_g);
            Bphi_L = linear(r4, Bphi4, r, Bphi1, r2, Bphi2, r_g);
            bx_L = Br_L * cos(phi_g) - Bphi_L * sin(phi_g);
            by_L = Br_L * sin(phi_g) + Bphi_L * cos(phi_g);

            if ((n <= N - 3))
            {
                rho_R = linear(r, s_1.x * kv(r), r2, s_2.x * kv(r2), r21, s_21.x * kv(r21), r_g) / kv(r_g);
                if (rho_R <= 0.0) rho_R = s_2.x;

                Vr21 = u_21.x * cos(phi21) + u_21.y * sin(phi21);
                Vphi21 = -u_21.x * sin(phi21) + u_21.y * cos(phi21);
                Vr_R = linear(r, Vr1, r2, Vr2, r21, Vr21, r_g);
                Vphi_R = linear(r, Vphi1, r2, Vphi2, r21, Vphi21, r_g);

                Br21 = b_21.x * cos(phi21) + b_21.y * sin(phi21);
                Bphi21 = -b_21.x * sin(phi21) + b_21.y * cos(phi21);
                Br_R = linear(r, Br1, r2, Br2, r21, Br21, r_g);
                Bphi_R = linear(r, Bphi1, r2, Bphi2, r21, Bphi21, r_g);

                psi_R = linear(r, s_1.z, r2, s_2.z, r21, s_21.z, r_g);
                w_R = linear(r, u_1.z, r2, u_2.z, r21, u_21.z, r_g);
                bz_R = linear(r, b_1.z, r2, b_2.z, r21, b_21.z, r_g);
            }
            else
            {
                rho_R = s_2.x * kv(r2 / r_g);
                Vr_R = Vr2;
                Vphi_R = Vphi2;
                Br_R = Br2;
                Bphi_R = Bphi2;
                w_R = u_2.z;
                psi_R = s_2.z;
                bz_R = b_2.z;
            }

            p_R = const_p * rho_R;
            u_R = Vr_R * cos(phi_g) - Vphi_R * sin(phi_g);
            v_R = Vr_R * sin(phi_g) + Vphi_R * cos(phi_g);
            bx_R = Br_R * cos(phi_g) - Bphi_R * sin(phi_g);
            by_R = Br_R * sin(phi_g) + Bphi_R * cos(phi_g);


            Bx_dipole_ = Bx_dipole(r_g, phi_g);
            By_dipole_ = By_dipole(r_g, phi_g);


            tmin = my_min(tmin, HLLDQ_Korolkov_psi(rho_L, psi_L, p_L, u_L, v_L, w_L, bx_L + Bx_dipole_, by_L + By_dipole_, bz_L, rho_R, psi_R, p_R, //
                u_R, v_R, w_R, bx_R + Bx_dipole_, by_R + By_dipole_, bz_R, P, PQ, n1, n2, 0.0, DR(n), method, ch_now, ch_max_, x, y));
            ch_max = max(ch_max, ch_max_);

            if (isnan(P[4]) == true || isnan(P[5]) == true || isnan(P[6]) == true)
            {
                printf("Problems P... = %lf, %lf, %lf, %lf, %lf, %lf, %lf, %d, %d \n", P[4], r_g, r2, P[0], P[7], rho_L, rho_R, n, m);
            }

            PS.x = PS.x + P[0] * DPHI(m) * r_g;
            PS.y = PS.y + P[7] * DPHI(m) * r_g;
            PS.z = PS.z + PQ   * DPHI(m) * r_g;
            PU.x = PU.x + P[1] * DPHI(m) * r_g;
            PU.y = PU.y + P[2] * DPHI(m) * r_g;
            PU.z = PU.z + P[3] * DPHI(m) * r_g;
            PB.x = PB.x + P[4] * DPHI(m) * r_g;
            PB.y = PB.y + P[5] * DPHI(m) * r_g;
            PB.z = PB.z + P[6] * DPHI(m) * r_g;
            //Pdiv = Pdiv + dphi * r_g * ( n1 * (bx_L + bx_R) / 2.0 + n2 * (by_L + by_R) / 2.0);
            Pdiv = Pdiv + DPHI(m) * r_g * (n1 * (bx_L + bx_R + 2.0 * Bx_dipole_) / 2.0 + n2 * (by_L + by_R + 2.0 * By_dipole_) / 2.0);
        }

        P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;

        // phi- грань
        if (true)
        {
            r_g = r;
            phi_g = PHI_LEFT(m);
            x_g = r_g * cos(phi_g);
            y_g = r_g * sin(phi_g);
            n1 = y_g / r_g;
            n2 = -x_g / r_g;


            rho_L = linear(phi5, s_5.x, phi, s_1.x, phi3, s_3.x, phi_g);
            if (rho_L <= 0.0) rho_L = s_1.x;
            p_L = const_p * rho_L;

            Vr_L = linear(phi5, Vr5, phi, Vr1, phi3, Vr3, phi_g);
            Vphi_L = linear(phi5, Vphi5, phi, Vphi1, phi3, Vphi3, phi_g);
            u_L = Vr_L * cos(phi_g) - Vphi_L * sin(phi_g);
            v_L = Vr_L * sin(phi_g) + Vphi_L * cos(phi_g);

            Br_L = linear(phi5, Br5, phi, Br1, phi3, Br3, phi_g);
            Bphi_L = linear(phi5, Bphi5, phi, Bphi1, phi3, Bphi3, phi_g);
            bx_L = Br_L * cos(phi_g) - Bphi_L * sin(phi_g);
            by_L = Br_L * sin(phi_g) + Bphi_L * cos(phi_g);

            psi_L = linear(phi5, s_5.z, phi, s_1.z, phi3, s_3.z, phi_g);
            w_L = linear(phi5, u_5.z, phi, u_1.z, phi3, u_3.z, phi_g);
            bz_L = linear(phi5, b_5.z, phi, b_1.z, phi3, b_3.z, phi_g);

            if (m >= 2)
            {
                rho_R = linear(phi, s_1.x, phi3, s_3.x, phi31, s_31.x, phi_g);
                if (rho_R <= 0.0) rho_R = s_3.x;

                Vr31 = u_31.x * cos(phi31) + u_31.y * sin(phi31);
                Vphi31 = -u_31.x * sin(phi31) + u_31.y * cos(phi31);
                Vr_R = linear(phi, Vr1, phi3, Vr3, phi31, Vr31, phi_g);
                Vphi_R = linear(phi, Vphi1, phi3, Vphi3, phi31, Vphi31, phi_g);

                Br31 = b_31.x * cos(phi31) + b_31.y * sin(phi31);
                Bphi31 = -b_31.x * sin(phi31) + b_31.y * cos(phi31);
                Br_R = linear(phi, Br1, phi3, Br3, phi31, Br31, phi_g);
                Bphi_R = linear(phi, Bphi1, phi3, Bphi3, phi31, Bphi31, phi_g);

                psi_R = linear(phi, s_1.z, phi3, s_3.z, phi31, s_31.z, phi_g);
                w_R = linear(phi, u_1.z, phi3, u_3.z, phi31, u_31.z, phi_g);
                bz_R = linear(phi, b_1.z, phi3, b_3.z, phi31, b_31.z, phi_g);
            }
            else
            {
                rho_R = s_3.x * kv(r3 / r_g);
                Vr_R = Vr3;
                Vphi_R = Vphi3;
                Br_R = Br3;
                Bphi_R = Bphi3;
                psi_R = s_3.z;
                w_R = u_3.z;
                bz_R = b_3.z;
            }

            p_R = const_p * rho_R;
            u_R = Vr_R * cos(phi_g) - Vphi_R * sin(phi_g);
            v_R = Vr_R * sin(phi_g) + Vphi_R * cos(phi_g);
            bx_R = Br_R * cos(phi_g) - Bphi_R * sin(phi_g);
            by_R = Br_R * sin(phi_g) + Bphi_R * cos(phi_g);


            if (m == 0)
            {
                u_R = -u_L;
                v_R = v_L;
                w_R = -w_L;
                rho_R = rho_L;
                p_R = p_L;
                bx_R = -bx_L;
                by_R = by_L;
                bz_R = -bz_L;
            }

            Bx_dipole_ = Bx_dipole(r_g, phi_g);
            By_dipole_ = By_dipole(r_g, phi_g);

            tmin = my_min(tmin, HLLDQ_Korolkov_psi(rho_L, psi_L, p_L, u_L, v_L, w_L, bx_L + Bx_dipole_, by_L + By_dipole_, bz_L, rho_R, psi_R, p_R, //
                u_R, v_R, w_R, bx_R + Bx_dipole_, by_R + By_dipole_, bz_R, P, PQ, n1, n2, 0.0, DPHI(m) * r_g, method, ch_now, ch_max_, x, y));
            ch_max = max(ch_max, ch_max_);

            PS.x = PS.x + P[0] * DR(n);
            PS.y = PS.y + P[7] * DR(n);
            PS.z = PS.z + PQ * DR(n);
            PU.x = PU.x + P[1] * DR(n);
            PU.y = PU.y + P[2] * DR(n);
            PU.z = PU.z + P[3] * DR(n);
            PB.x = PB.x + P[4] * DR(n);
            PB.y = PB.y + P[5] * DR(n);
            PB.z = PB.z + P[6] * DR(n);
            //Pdiv = Pdiv + DR(n) * (n1 * (bx_L + bx_R) / 2.0 + n2 * (by_L + by_R) / 2.0);
            Pdiv = Pdiv + DR(n) * (n1 * (bx_L + bx_R + 2.0 * Bx_dipole_) / 2.0 + n2 * (by_L + by_R + 2.0 * By_dipole_) / 2.0);
        }

        P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;

        // r- грань
        if (true)
        {
            n1 = -x / r;
            n2 = -y / r;
            r_g = R_EDGE(n);
            phi_g = phi;
            x_g = r_g * cos(phi_g);
            y_g = r_g * sin(phi_g);

            rho_L = linear(r2, s_2.x * kv(r2), r, s_1.x * kv(r), r4, s_4.x * kv(r4), r_g) / kv(r_g);
            if (rho_L <= 0.0) rho_L = s_1.x;
            p_L = const_p * rho_L;

            Vr_L = linear(r2, Vr2, r, Vr1, r4, Vr4, r_g);
            Vphi_L = linear(r2, Vphi2, r, Vphi1, r4, Vphi4, r_g);
            u_L = Vr_L * cos(phi_g) - Vphi_L * sin(phi_g);
            v_L = Vr_L * sin(phi_g) + Vphi_L * cos(phi_g);

            Br_L = linear(r2, Br2, r, Br1, r4, Br4, r_g);
            Bphi_L = linear(r2, Bphi2, r, Bphi1, r4, Bphi4, r_g);
            bx_L = Br_L * cos(phi_g) - Bphi_L * sin(phi_g);
            by_L = Br_L * sin(phi_g) + Bphi_L * cos(phi_g);

            psi_L = linear(r2, s_2.z, r, s_1.z, r4, s_4.z, r_g);
            w_L = linear(r2, u_2.z, r, u_1.z, r4, u_4.z, r_g);
            bz_L = linear(r2, b_2.z, r, b_1.z, r4, b_4.z, r_g);

            if (n >= 2)
            {
                rho_R = linear(r, s_1.x * kv(r), r4, s_4.x * kv(r4), r41, s_41.x * kv(r41), r_g) / kv(r_g);
                if (rho_R <= 0.0) rho_R = s_4.x;

                Vr41 = u_41.x * cos(phi41) + u_41.y * sin(phi41);
                Vphi41 = -u_41.x * sin(phi41) + u_41.y * cos(phi41);
                Vr_R = linear(r, Vr1, r4, Vr4, r41, Vr41, r_g);
                Vphi_R = linear(r, Vphi1, r4, Vphi4, r41, Vphi41, r_g);

                Br41 = b_41.x * cos(phi41) + b_41.y * sin(phi41);
                Bphi41 = -b_41.x * sin(phi41) + b_41.y * cos(phi41);
                Br_R = linear(r, Br1, r4, Br4, r41, Br41, r_g);
                Bphi_R = linear(r, Bphi1, r4, Bphi4, r41, Bphi41, r_g);

                psi_R = linear(r, s_1.z, r4, s_4.z, r41, s_41.z, r_g);
                w_R = linear(r, u_1.z, r4, u_4.z, r41, u_41.z, r_g);
                bz_R = linear(r, b_1.z, r4, b_4.z, r41, b_41.z, r_g);
            }
            else
            {
                rho_R = s_4.x * kv(r4 / r_g);
                Vr_R = Vr4;
                Vphi_R = Vphi4;
                Br_R = Br4;
                Bphi_R = Bphi4;
                psi_R = s_4.z;
                w_R = u_4.z;
                bz_R = b_4.z;
            }

            p_R = const_p * rho_R;
            u_R = Vr_R * cos(phi_g) - Vphi_R * sin(phi_g);
            v_R = Vr_R * sin(phi_g) + Vphi_R * cos(phi_g);
            bx_R = Br_R * cos(phi_g) - Bphi_R * sin(phi_g);
            by_R = Br_R * sin(phi_g) + Bphi_R * cos(phi_g);

            if (n == 0)
            {
                u_L = u_R;
                v_L = v_R;
                w_L = w_R;
                rho_L = rho_R;
                p_L = p_R;
                bx_L = bx_R;
                by_L = by_R;
                bz_L = bz_R;
            }

            Bx_dipole_ = Bx_dipole(r_g, phi_g);
            By_dipole_ = By_dipole(r_g, phi_g);

            tmin = my_min(tmin, HLLDQ_Korolkov_psi(rho_L, psi_L, p_L, u_L, v_L, w_L, bx_L + Bx_dipole_, by_L + By_dipole_, bz_L, rho_R, psi_R, p_R, //
                u_R, v_R, w_R, bx_R + Bx_dipole_, by_R + By_dipole_, bz_R, P, PQ, n1, n2, 0.0, DR(n), method, ch_now, ch_max_, x, y));
            ch_max = max(ch_max, ch_max_);

            PS.x = PS.x + P[0] * DPHI(m) * r_g;
            PS.y = PS.y + P[7] * DPHI(m) * r_g;
            PS.z = PS.z + PQ * DPHI(m) * r_g;
            PU.x = PU.x + P[1] * DPHI(m) * r_g;
            PU.y = PU.y + P[2] * DPHI(m) * r_g;
            PU.z = PU.z + P[3] * DPHI(m) * r_g;
            PB.x = PB.x + P[4] * DPHI(m) * r_g;
            PB.y = PB.y + P[5] * DPHI(m) * r_g;
            PB.z = PB.z + P[6] * DPHI(m) * r_g;
            //Pdiv = Pdiv + dphi * r_g * (n1 * (b_1.x + b_4.x) / 2.0 + n2 * (b_1.y + b_4.y) / 2.0);
            Pdiv = Pdiv + DPHI(m) * r_g * (n1 * (bx_L + bx_R + 2.0 * Bx_dipole_) / 2.0 + n2 * (by_L + by_R + 2.0 * By_dipole_) / 2.0);
        }

        P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;

        // phi+ грань
        if (true)
        {
            r_g = r;
            phi_g = PHI_RIGHT(m);
            x_g = r_g * cos(phi_g);
            y_g = r_g * sin(phi_g);
            n1 = -y_g / r_g;
            n2 = x_g / r_g;

            
            rho_L = linear(phi3, s_3.x, phi, s_1.x, phi5, s_5.x, phi_g);
            if (rho_L <= 0.0) rho_L = s_1.x;
            p_L = const_p * rho_L;

            Vr_L = linear(phi3, Vr3, phi, Vr1, phi5, Vr5, phi_g);
            Vphi_L = linear(phi3, Vphi3, phi, Vphi1, phi5, Vphi5, phi_g);
            u_L = Vr_L * cos(phi_g) - Vphi_L * sin(phi_g);
            v_L = Vr_L * sin(phi_g) + Vphi_L * cos(phi_g);

            Br_L = linear(phi3, Br3, phi, Br1, phi5, Br5, phi_g);
            Bphi_L = linear(phi3, Bphi3, phi, Bphi1, phi5, Bphi5, phi_g);
            bx_L = Br_L * cos(phi_g) - Bphi_L * sin(phi_g);
            by_L = Br_L * sin(phi_g) + Bphi_L * cos(phi_g);

            psi_L = linear(phi3, s_3.z, phi, s_1.z, phi5, s_5.z, phi_g);
            w_L = linear(phi3, u_3.z, phi, u_1.z, phi5, u_5.z, phi_g);
            bz_L = linear(phi3, b_3.z, phi, b_1.z, phi5, b_5.z, phi_g);

            if (m <= M - 3)
            {
                rho_R = linear(phi, s_1.x, phi5, s_5.x, phi51, s_51.x, phi_g);
                if (rho_R <= 0.0) rho_R = s_5.x;

                Vr51 = u_51.x * cos(phi51) + u_51.y * sin(phi51);
                Vphi51 = -u_51.x * sin(phi51) + u_51.y * cos(phi51);
                Vr_R = linear(phi, Vr1, phi5, Vr5, phi51, Vr51, phi_g);
                Vphi_R = linear(phi, Vphi1, phi5, Vphi5, phi51, Vphi51, phi_g);

                Br51 = b_51.x * cos(phi51) + b_51.y * sin(phi51);
                Bphi51 = -b_51.x * sin(phi51) + b_51.y * cos(phi51);
                Br_R = linear(phi, Br1, phi5, Br5, phi51, Br51, phi_g);
                Bphi_R = linear(phi, Bphi1, phi5, Bphi5, phi51, Bphi51, phi_g);

                psi_R = linear(phi, s_1.z, phi5, s_5.z, phi51, s_51.z, phi_g);
                w_R = linear(phi, u_1.z, phi5, u_5.z, phi51, u_51.z, phi_g);
                bz_R = linear(phi, b_1.z, phi5, b_5.z, phi51, b_51.z, phi_g);
            }
            else
            {
                rho_R = s_5.x * kv(r5 / r_g);
                Vr_R = Vr5;
                Vphi_R = Vphi5;
                Br_R = Br5;
                Bphi_R = Bphi5;
                psi_R = s_5.z;
                w_R = u_5.z;
                bz_R = b_5.z;
            }

            p_R = const_p * rho_R;
            u_R = Vr_R * cos(phi_g) - Vphi_R * sin(phi_g);
            v_R = Vr_R * sin(phi_g) + Vphi_R * cos(phi_g);
            bx_R = Br_R * cos(phi_g) - Bphi_R * sin(phi_g);
            by_R = Br_R * sin(phi_g) + Bphi_R * cos(phi_g);

            if (m == M - 1)
            {
                u_R = -u_L;
                v_R = v_L;
                w_R = -w_L;
                rho_R = rho_L;
                p_R = p_L;
                bx_R = -bx_L;
                by_R = by_L;
                bz_R = -bz_L;
            }

            Bx_dipole_ = Bx_dipole(r_g, phi_g);
            By_dipole_ = By_dipole(r_g, phi_g);

            tmin = my_min(tmin, HLLDQ_Korolkov_psi(rho_L, psi_L, p_L, u_L, v_L, w_L, bx_L + Bx_dipole_, by_L + By_dipole_, bz_L, rho_R, psi_R, p_R, //
                u_R, v_R, w_R, bx_R + Bx_dipole_, by_R + By_dipole_, bz_R, P, PQ, n1, n2, 0.0, DPHI(m) * r_g, method, ch_now, ch_max_, x, y));
            ch_max = max(ch_max, ch_max_);

            PS.x = PS.x + P[0] * DR(n);
            PS.y = PS.y + P[7] * DR(n);
            PS.z = PS.z + PQ * DR(n);
            PU.x = PU.x + P[1] * DR(n);
            PU.y = PU.y + P[2] * DR(n);
            PU.z = PU.z + P[3] * DR(n);
            PB.x = PB.x + P[4] * DR(n);
            PB.y = PB.y + P[5] * DR(n);
            PB.z = PB.z + P[6] * DR(n);
            //Pdiv = Pdiv + DR(n) * (n1 * (bx_L + bx_R) / 2.0 + n2 * (by_L + by_R) / 2.0);
            Pdiv = Pdiv + DR(n) * (n1 * (bx_L + bx_R + 2.0 * Bx_dipole_) / 2.0 + n2 * (by_L + by_R + 2.0 * By_dipole_) / 2.0);
        }

    }


    if (*T > tmin)
    {
        atomicMinDouble(T, tmin);
    }

    if (*ch_posle < ch_max)
    {
        atomicMaxDouble(ch_posle, ch_max);
    }

    double dV = CELL_AREA(n, m);
    //double dV = CELL_AREA(n);

    double Fx = 0.0, Fy = 0.0;

    // Вычисляем силы
    if (true)
    {
        double fr = 0.0;
        fr = (F_grav + F_continuum) * s_1.x / kv(r);  // Сила притяжения к звезде + радиационное отталкивание от континуума

        if (true)
        {
            // Line-driven force
            //double Vr2 = (u_2.x * (x + dx) + u_2.y * y) / sqrt(kvv(0.0, x + dx, y));
            //double Vr3 = (u_3.x * x + u_3.y * (y - dy)) / sqrt(kvv(0.0, x, y - dy));
            //double Vr5 = (u_5.x * x + u_5.y * (y + dy)) / sqrt(kvv(0.0, x, y + dy));
            //double Vr4 = (u_4.x * (x - dx) + u_4.y * y) / sqrt(kvv(0.0, x - dx, y));
            //double dVrdr = ((Vr2 - Vr4) / (2 * dx) * x + (Vr5 - Vr3) / (2 * dy) * y) / r;


            //double dVrdr = DERIVATIVE(Vr4, Vr1, Vr2, dr4, dr2);
            double dVrdr = (Vr2 - Vr1) / dr2;

            //dVrdr = min(dVrdr, 100.0);

            //if (isnan(dVrdr) == true || dVrdr == 0.0)
            //{
            //    printf("Problems fr = %lf, %lf, %lf, %lf \n", Vr2, Vr1, dr2, pow(dVrdr, alpha_line));
            //}

            //dVrdr = max(dVrdr, 1.0);

            double sigma = fabs(dVrdr * r / Vr1) - 1.0;  // Vr1
            double muc = 1.0 - 1.0 / kv(r);

            double ff = 1.0;

            if (fabs(dVrdr) > 0.00001)
            {
                ff = (pow(1.0 + sigma, 1.0 + alpha_line) - pow(1.0 + sigma * muc, 1.0 + alpha_line)) /
                    ((1.0 + alpha_line) * (1.0 - muc) * sigma * pow(1.0 + sigma, alpha_line));

                if (isnan(ff) == true)
                {
                    //printf("Problems ff = %lf, %lf, %lf \n", ff, sigma, muc);
                }
            }

            //double sigma = dist / ((u_1.x * x + u_1.y * y) / dist) * fabs(dVrdr) - 1.0;
            //double muc = sqrt(1.0 - 1.0 / kv(dist));
            //double fD = (pow(1.0 + sigma, 1.6) - pow(1.0 + sigma * kv(muc), 1.6)) / (1.6 * (1.0 - kv(muc)) * sigma * pow(1.0 + sigma, 0.6));

            //fr += s_1.x * fD * 6.9152 * pow(5.50815E-7 / s_1.x * fabs(dVrdr), 0.6) / kv(dist);

            double fline = F_line * ff * s_1.x * pow(fabs(dVrdr) / s_1.x, alpha_line) / kv(r);
            fr += fline;

            //if (isnan(fr) == true)
            //{
            //    printf("Problems fr = %lf, %lf, %lf, %lf, %lf \n", fr, ff, fabs(dVrdr), Vr2, Vr1);
            //}


            //double A_abbott = Vr1 - alpha_line * fline / fabs(dVrdr);
            tmin = krit * dr2 / (alpha_line * fline / max(fabs(dVrdr), 0.0005));

            if (*T > tmin)
            {
                atomicMinDouble(T, tmin);
            }

            //double vth = sqrt(const_p);
            //double vth = sqrt(const_p * sqrt(1.0 / r));
            //double tt = F_line * s_1.x * vth;
            //double Mt = k_line * pow(fabs(dVrdr)/tt, alpha_line);
            //fr += F_continuum * s_1.x * Mt / kv(r);

            //if (fr > 100.0)
            //{
            //    printf("Problems fr = %lf", fr);
            //}

        }

        //if (fr < 0.0 && Vr1 <= 0.0)
        //{
        //    fr = 0.000001;
        //}

        Fx = fr * x / r;
        Fy = fr * y / r;
    }


    double bx = b_1.x + Bx_dipole(r, phi);
    double by = b_1.y + By_dipole(r, phi);

    //Pdiv = Pdiv + dV * b_1.x / x;

    Pdiv = Pdiv + dV * bx / x;
    //Pdiv = 0.0;


    s2[index].x = s_1.x - *T_do * (PS.x / dV + s_1.x * u_1.x / x);
    //s2[index].x = s_1.x - (*T_do / dV) * PS.x;   // В декартовых координатах
    if (s2[index].x <= 0)
    {
        printf("Problemsssss! x = %lf, y = %lf, ro = %lf, T = %lf, ro = %lf \n", x, y, s2[index].x, *T_do, s_1.x);
        s2[index].x = s_1.x;
    }

    //u2[index].x = (s_1.x * u_1.x - (*T_do / dV) * (PU.x + (b_1.x/cpi4)*Pdiv ) - *T_do * (s_1.x * u_1.y * u_1.x) / y) / s2[index].x;
    //u2[index].y = (s_1.x * u_1.y - (*T_do / dV) * (PU.y + (b_1.y / cpi4) * Pdiv) - *T_do * s_1.x * u_1.y * u_1.y / y) / s2[index].x;
    //b2[index].x = (b_1.x - *T_do * (PB.x + u_1.x * Pdiv) / dV) - *T_do * (u_1.y*b_1.x - u_1.x * b_1.y)/y;
    //b2[index].y = (b_1.y - *T_do * (PB.y + u_1.y * Pdiv) / dV);
    //b2[index].z = (b_1.x - *T_do * (PB.z) / dV) - *T_do * (u_1.y * b_1.z) / y;

    //s2[index].y = ( ((s_1.y / (ggg - 1) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) - (*T_do / dV) * (PS.y + //
    //    (skk(u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z) / cpi4) * Pdiv) - //
    //    *T_do * u_1.y * (ggg * s_1.y / (ggg - 1) + s_1.x * (u_1.x * u_1.x + u_1.y * u_1.y) * 0.5) / y) - //
    //    0.5 * s2[index].x * (u2[index].x * u2[index].x + u2[index].y * u2[index].y) - kvv(b_1.x, b_1.y, b_1.z) / cpi8 ) * (ggg - 1);


    //s2[index].x = s_1.x - *T_do * PS.x / dV - *T_do * s_1.x * u_1.y / y;
    //u2[index].x = (s_1.x * u_1.x - *T_do * (PU.x + (b_1.x / cpi4) * Pdiv) / dV  - *T_do * (s_1.x * u_1.x * u_1.y - b_1.x * b_1.y /cpi4)/y ) / s2[index].x;

    //double Smr = (s_1.x * kv(u_1.z) - kv(b_1.z) / cpi4 - (s_1.x * kv(u_1.x) + s_1.y + kvv(b_1.x, b_1.y, b_1.z) / cpi8 - kv(b_1.x) / cpi4)) / x;
    //u2[index].x = (s_1.x * u_1.x - *T_do * (PU.x + (b_1.x / cpi4) * Pdiv) / dV  + *T_do * Smr + *T_do * Fx) / s2[index].x;

    //u2[index].x = (s_1.x * u_1.x - *T_do * (PU.x + (b_1.x / cpi4) * Pdiv) / dV  - *T_do * (s_1.x * u_1.y * u_1.y + (kv(b_1.x) + kv(b_1.z)) / cpi4) / x + *T_do * Fx) / s2[index].x;



    u2[index].x = (s_1.x * u_1.x - *T_do * (PU.x + (bx / cpi4) * Pdiv) / dV + *T_do * (s_1.x * (kv(u_1.z) - kv(u_1.x)) + (kv(bx) - kv(b_1.z)) / cpi4) / x + *T_do * Fx) / s2[index].x;
    u2[index].y = (s_1.x * u_1.y - *T_do * (PU.y + (by / cpi4) * Pdiv) / dV - *T_do * (s_1.x * u_1.x * u_1.y - bx * by / cpi4) / x + *T_do * Fy) / s2[index].x;
    u2[index].z = (s_1.x * u_1.z - *T_do * (PU.z + (b_1.z / cpi4) * Pdiv) / dV - 2.0 * *T_do * (s_1.x * u_1.x * u_1.z - bx * b_1.z / cpi4) / x) / s2[index].x;



    //u2[index].y = (s_1.x * u_1.y - *T_do * (PU.y + (b_1.y / cpi4) * Pdiv) / dV - *T_do * (s_1.x * u_1.y * u_1.y + (kv(b_1.z) - kv(b_1.y)) / cpi4) / y ) / s2[index].x;
    //b2[index].x = (b_1.x - *T_do * (PB.x + u_1.x * Pdiv) / dV - *T_do*(u_1.y * b_1.x - b_1.y * u_1.x)/y);
    //b2[index].y = (b_1.y - *T_do * (PB.y + u_1.y * Pdiv) / dV);

    b2[index].x = b_1.x - *T_do * (PB.x + u_1.x * Pdiv) / dV;
    b2[index].y = (b_1.y - *T_do * (PB.y + u_1.y * Pdiv) / dV - *T_do * (u_1.x * by - bx * u_1.y) / x);
    b2[index].z = b_1.z - *T_do * (PB.z + u_1.z * Pdiv) / dV;

    //s2[index].y = (U8(s_1.x, s_1.y, u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z) - *T_do * (PS.y + (skk(u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z) / cpi4) * Pdiv)//
    //    / dV - *T_do * ( ( (U8(s_1.x, s_1.y, u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z) + s_1.y + kvv(b_1.x, b_1.y, b_1.z) / cpi8)* u_1.y - b_1.y * skk(u_1.x, u_1.y, 0.0, b_1.x, b_1.y, b_1.z)/cpi4)/ y) //
    //    - 0.5 * s2[index].x * kvv(u2[index].x, u2[index].y, 0.0) - kvv(b2[index].x, b2[index].y, b2[index].z) / cpi8) * (ggg - 1.0);


    //s2[index].y = const_p * s2[index].x * sqrt(1.0 / r);
    s2[index].y = const_p * s2[index].x;
    //s2[index].y = s2[index].x;

    double tau = 0.18 * min(DPHI(m) * r, DR(n)) / ch_now;  // 0.18     18 - норм
    //double tau = 0.18/ ch_now;

    if (tau < 2.0 * *T_do)  tau = 2.0 * *T_do;
    if (tau > 100.0 * *T_do) tau = 100.0 * *T_do;

    //tau = 4.0 * *T_do;

    s2[index].z = (s_1.z - *T_do * PS.z / dV - *T_do * ch_now * ch_now * bx / x) * exp(-*T_do / tau);


    //s2[index].y = 0.000671042 * s2[index].x * sqrt(1.0 / dist);
    //s2[index].y = s2[index].x;
}


__device__ double get_cell(const double* field, int N_, int M_, int i, int j) 
{
    // Экстраполяция: clamp к допустимым индексам
    if (i == -1) i = 0;
    if (i == -2) i = 1;
    if (i == N_) i = N_ - 1;
    if (i == N_ + 1) i = N_ - 2;
    if (j == -1) j = 0;
    if (j == -2) j = 1;
    if (j == M_) j = M_ - 1;
    if (j == M_ + 1) j = M_ - 2;
    return field[i * M_ + j];
}


__global__ void compute_fluxes(
    const double* rho, const double* Vx, const double* Vy, const double* Vz,
    const double* Bx, const double* By, const double* Bz,
    double* h_Prho, double* h_Pvx, double* h_Pvy, double* h_Pvz,
    double* h_Pbx, double* h_Pby, double* h_Pbz, const double* h_Bn,
    double* v_Prho, double* v_Pvx, double* v_Pvy, double* v_Pvz,
    double* v_Pbx, double* v_Pby, double* v_Pbz, const double* v_Bn)
{
    // Shared memory для всех 7 переменных ячеек
    __shared__ double sh_rho[SHARED_I][SHARED_J];
    __shared__ double sh_Vx[SHARED_I][SHARED_J];
    __shared__ double sh_Vy[SHARED_I][SHARED_J];
    __shared__ double sh_Vz[SHARED_I][SHARED_J];
    __shared__ double sh_Bx[SHARED_I][SHARED_J];
    __shared__ double sh_By[SHARED_I][SHARED_J];
    __shared__ double sh_Bz[SHARED_I][SHARED_J];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int i_start = bx * BX;
    int j_start = by * BY;

    // --- Загрузка блока ячеек с запасом ---
    if (true)
    {
        int load_i_start = i_start - HALO;
        int load_j_start = j_start - HALO;
        int total_load = SHARED_I * SHARED_J;
        int tid = tx + ty * BX;  // линейный индекс в блоке

        for (int idx = tid; idx < total_load; idx += BX * BY)
        {
            int i_local = idx / SHARED_J;
            int j_local = idx % SHARED_J;
            int i_glob = load_i_start + i_local;
            int j_glob = load_j_start + j_local;

            sh_rho[i_local][j_local] = get_cell(rho, N, M, i_glob, j_glob);
            sh_Vx[i_local][j_local] = get_cell(Vx, N, M, i_glob, j_glob);
            sh_Vy[i_local][j_local] = get_cell(Vy, N, M, i_glob, j_glob);
            sh_Vz[i_local][j_local] = get_cell(Vz, N, M, i_glob, j_glob);
            sh_Bx[i_local][j_local] = get_cell(Bx, N, M, i_glob, j_glob);
            sh_By[i_local][j_local] = get_cell(By, N, M, i_glob, j_glob);
            sh_Bz[i_local][j_local] = get_cell(Bz, N, M, i_glob, j_glob);
        }
    }
    __syncthreads();

    // --- Обработка текущей ячейки (i, j) --- это глобальный индекс ячейки, которая сейчас обрабатывается этим потоком
    int i = i_start + tx;
    int j = j_start + ty;
    if (i >= N || j >= M) return;

    // Локальные координаты в shared (с учётом запаса)
    // это координаты текущей ячейки в shared памяти (это важно, так как через них можно брать координаты соседей и т.д.)
    int i_l = tx + HALO;
    int j_l = ty + HALO;

    // Считаем потоки
    if (true)
    {
        double tmin = 1.0E30;
        double rho_L, rho_R, Vx_L, Vx_R, Vy_L, Vy_R, Vz_L, Vz_R;
        double Bx_L, Bx_R, By_L, By_R, Bz_L, Bz_R;
        double r = R_CENTER(i, j);
        
        // 1. Верхняя горизонтальная грань (она заполняется для всех ячеек)
        if (true)
        {
            double phi_g = PHI_RIGHT(j);

            if (true)
            {
                double phi1 = PHI_CENTER(j);
                double phi2, phi4;
                double phi3;

                if (j == 0)
                {
                    phi3 = -pi - phi1; // Симметричная ячейка относительно оси
                }
                else
                {
                    phi3 = PHI_CENTER(j - 1);
                }

                if (j == M - 1)
                {
                    phi2 = pi - phi1;
                    phi4 = pi - phi3;
                }
                else if (j == M - 2)
                {
                    phi2 = PHI_CENTER(j + 1);
                    phi4 = pi - phi2;
                }
                else
                {
                    phi2 = PHI_CENTER(j + 1);
                    phi4 = PHI_CENTER(j + 2);
                }

                // Ставим симметрию на оси
                if (j == 0)
                {
                    sh_Vz[i_l][j_l - 1] = -sh_Vz[i_l][j_l];
                    sh_Vx[i_l][j_l - 1] = -sh_Vx[i_l][j_l];
                    sh_Bx[i_l][j_l - 1] = -sh_Bx[i_l][j_l];
                    sh_Bz[i_l][j_l - 1] = -sh_Bz[i_l][j_l];

                    sh_Vz[i_l][j_l - 2] = -sh_Vz[i_l][j_l + 1];
                    sh_Vx[i_l][j_l - 2] = -sh_Vx[i_l][j_l + 1];
                    sh_Bx[i_l][j_l - 2] = -sh_Bx[i_l][j_l + 1];
                    sh_Bz[i_l][j_l - 2] = -sh_Bz[i_l][j_l + 1];

                    sh_Vy[i_l][j_l - 2] = sh_Vy[i_l][j_l + 1];
                    sh_By[i_l][j_l - 2] = sh_By[i_l][j_l + 1];
                    sh_rho[i_l][j_l - 2] = sh_rho[i_l][j_l + 1];
                }

                if (j == M - 1)
                {
                    sh_Vz[i_l][j_l + 1] = -sh_Vz[i_l][j_l];
                    sh_Vx[i_l][j_l + 1] = -sh_Vx[i_l][j_l];
                    sh_Bx[i_l][j_l + 1] = -sh_Bx[i_l][j_l];
                    sh_Bz[i_l][j_l + 1] = -sh_Bz[i_l][j_l];

                    sh_Vz[i_l][j_l + 2] = -sh_Vz[i_l][j_l - 1];
                    sh_Vx[i_l][j_l + 2] = -sh_Vx[i_l][j_l - 1];
                    sh_Bx[i_l][j_l + 2] = -sh_Bx[i_l][j_l - 1];
                    sh_Bz[i_l][j_l + 2] = -sh_Bz[i_l][j_l - 1];
                    sh_Vy[i_l][j_l + 2] = sh_Vy[i_l][j_l - 1];
                    sh_By[i_l][j_l + 2] = sh_By[i_l][j_l - 1];
                    sh_rho[i_l][j_l + 2] = sh_rho[i_l][j_l - 1];
                }
                else if (j == M - 2)
                {
                    sh_Vz[i_l][j_l + 2] = -sh_Vz[i_l][j_l + 1];
                    sh_Vx[i_l][j_l + 2] = -sh_Vx[i_l][j_l + 1];
                    sh_Bx[i_l][j_l + 2] = -sh_Bx[i_l][j_l + 1];
                    sh_Bz[i_l][j_l + 2] = -sh_Bz[i_l][j_l + 1];
                }

                rho_L = linear(phi3, sh_rho[i_l][j_l - 1], phi1, sh_rho[i_l][j_l], phi2, sh_rho[i_l][j_l + 1], phi_g);
                if (rho_L <= 0.0) rho_L = sh_rho[i_l][j_l];
                rho_R = linear(phi4, sh_rho[i_l][j_l + 2], phi2, sh_rho[i_l][j_l + 1], phi1, sh_rho[i_l][j_l], phi_g);
                if (rho_R <= 0.0) rho_R = sh_rho[i_l][j_l + 1];

                Vz_L = linear(phi3, sh_Vz[i_l][j_l - 1], phi1, sh_Vz[i_l][j_l], phi2, sh_Vz[i_l][j_l + 1], phi_g);
                Vz_R = linear(phi4, sh_Vz[i_l][j_l + 2], phi2, sh_Vz[i_l][j_l + 1], phi1, sh_Vz[i_l][j_l], phi_g);

                Bz_L = linear(phi3, sh_Bz[i_l][j_l - 1], phi1, sh_Bz[i_l][j_l], phi2, sh_Bz[i_l][j_l + 1], phi_g);
                Bz_R = linear(phi4, sh_Bz[i_l][j_l + 2], phi2, sh_Bz[i_l][j_l + 1], phi1, sh_Bz[i_l][j_l], phi_g);

                // Скорости Vx, Vy
                if (true)
                {
                    double Vr_L, Vphi_L, Vr_R, Vphi_R;

                    // Vr
                    if (true)
                    {
                        double Vr1 = sh_Vx[i_l][j_l] * cos(phi1) + sh_Vy[i_l][j_l] * sin(phi1);
                        double Vr2 = sh_Vx[i_l][j_l + 1] * cos(phi2) + sh_Vy[i_l][j_l + 1] * sin(phi2);
                        double Vr3 = sh_Vx[i_l][j_l - 1] * cos(phi3) + sh_Vy[i_l][j_l - 1] * sin(phi3);
                        double Vr4 = sh_Vx[i_l][j_l + 2] * cos(phi4) + sh_Vy[i_l][j_l + 2] * sin(phi4);

                        Vr_L = linear(phi3, Vr3, phi1, Vr1, phi2, Vr2, phi_g);
                        Vr_R = linear(phi4, Vr4, phi2, Vr2, phi1, Vr1, phi_g);
                    }

                    // Vphi
                    if (true)
                    {
                        double Vphi1 = -sh_Vx[i_l][j_l] * sin(phi1) + sh_Vy[i_l][j_l] * cos(phi1);
                        double Vphi2 = -sh_Vx[i_l][j_l + 1] * sin(phi2) + sh_Vy[i_l][j_l + 1] * cos(phi2);
                        double Vphi3 = -sh_Vx[i_l][j_l - 1] * sin(phi3) + sh_Vy[i_l][j_l - 1] * cos(phi3);
                        double Vphi4 = -sh_Vx[i_l][j_l + 2] * sin(phi4) + sh_Vy[i_l][j_l + 2] * cos(phi4);

                        Vphi_L = linear(phi3, Vphi3, phi1, Vphi1, phi2, Vphi2, phi_g);
                        Vphi_R = linear(phi4, Vphi4, phi2, Vphi2, phi1, Vphi1, phi_g);
                    }

                    Vx_L = Vr_L * cos(phi_g) - Vphi_L * sin(phi_g);
                    Vy_L = Vr_L * sin(phi_g) + Vphi_L * cos(phi_g);

                    Vx_R = Vr_R * cos(phi_g) - Vphi_R * sin(phi_g);
                    Vy_R = Vr_R * sin(phi_g) + Vphi_R * cos(phi_g);
                }

                // Магнитные поля Bx, By
                if (true)
                {
                    double Br_L, Bphi_L, Br_R, Bphi_R;

                    // Br
                    if (true)
                    {
                        double Br1 = sh_Bx[i_l][j_l] * cos(phi1) + sh_By[i_l][j_l] * sin(phi1);
                        double Br2 = sh_Bx[i_l][j_l + 1] * cos(phi2) + sh_By[i_l][j_l + 1] * sin(phi2);
                        double Br3 = sh_Bx[i_l][j_l - 1] * cos(phi3) + sh_By[i_l][j_l - 1] * sin(phi3);
                        double Br4 = sh_Bx[i_l][j_l + 2] * cos(phi4) + sh_By[i_l][j_l + 2] * sin(phi4);

                        Br_L = linear(phi3, Br3, phi1, Br1, phi2, Br2, phi_g);
                        Br_R = linear(phi4, Br4, phi2, Br2, phi1, Br1, phi_g);
                    }

                    // Bphi
                    if (true)
                    {
                        double Bphi1 = -sh_Bx[i_l][j_l] * sin(phi1) + sh_By[i_l][j_l] * cos(phi1);
                        double Bphi2 = -sh_Bx[i_l][j_l + 1] * sin(phi2) + sh_By[i_l][j_l + 1] * cos(phi2);
                        double Bphi3 = -sh_Bx[i_l][j_l - 1] * sin(phi3) + sh_By[i_l][j_l - 1] * cos(phi3);
                        double Bphi4 = -sh_Bx[i_l][j_l + 2] * sin(phi4) + sh_By[i_l][j_l + 2] * cos(phi4);

                        Bphi_L = linear(phi3, Bphi3, phi1, Bphi1, phi2, Bphi2, phi_g);
                        Bphi_R = linear(phi4, Bphi4, phi2, Bphi2, phi1, Bphi1, phi_g);
                    }

                    Bx_L = Br_L * cos(phi_g) - Bphi_L * sin(phi_g);
                    By_L = Br_L * sin(phi_g) + Bphi_L * cos(phi_g);

                    Bx_R = Br_R * cos(phi_g) - Bphi_R * sin(phi_g);
                    By_R = Br_R * sin(phi_g) + Bphi_R * cos(phi_g);
                }
            }

            // Считаем поток
            if (true)
            {
                // Надо будет ещё подпроавить Bn в ячейке
                double PQ = 0.0;
                double P[8];
                P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;

                tmin = my_min(tmin, HLLDQ_Korolkov(rho_L, 0.0, const_p * rho_L, Vx_L, Vy_L, Vz_L, 0.0, 0.0, 0.0,
                    rho_R, 0.0, const_p * rho_R, Vx_R, Vy_R, Vz_R, 0.0, 0.0, 0.0,
                    P, PQ, -sin(phi_g), cos(phi_g), 0.0, DPHI(j) * r, 1));

                int idx_h = i * (M + 1) + (j + 1);
                h_Prho[idx_h] = P[0];
                h_Pvx[idx_h] = P[1];
                h_Pvy[idx_h] = P[2];
                h_Pvz[idx_h] = P[3];
                h_Pbx[idx_h] = P[4];
                h_Pby[idx_h] = P[5];
                h_Pbz[idx_h] = P[6];
            }
        }

        // 2. Нижняя горизонтальная грань (она есть только у нижнего ряда ячеек)
        if (j == 0)
        {
            // Значения в фиктивных ячейках были заполнены в 1.
            double phi_g = -pi / 2.0;

            if (true)
            {
                double phi1, phi2, phi3, phi4;

                phi2 = PHI_CENTER(j);
                phi4 = PHI_CENTER(j + 1);
                phi1 = -pi - phi2;
                phi3 = -pi - phi4;

                rho_L = linear(phi3, sh_rho[i_l][j_l - 2], phi1, sh_rho[i_l][j_l - 1], phi2, sh_rho[i_l][j_l], phi_g);
                if (rho_L <= 0.0) rho_L = sh_rho[i_l][j_l];
                rho_R = linear(phi4, sh_rho[i_l][j_l + 1], phi2, sh_rho[i_l][j_l], phi1, sh_rho[i_l][j_l - 1], phi_g);
                if (rho_R <= 0.0) rho_R = sh_rho[i_l][j_l + 1];

                Vz_L = linear(phi3, sh_Vz[i_l][j_l - 2], phi1, sh_Vz[i_l][j_l - 1], phi2, sh_Vz[i_l][j_l], phi_g);
                Vz_R = linear(phi4, sh_Vz[i_l][j_l + 1], phi2, sh_Vz[i_l][j_l], phi1, sh_Vz[i_l][j_l - 1], phi_g);

                Bz_L = linear(phi3, sh_Bz[i_l][j_l - 2], phi1, sh_Bz[i_l][j_l - 1], phi2, sh_Bz[i_l][j_l], phi_g);
                Bz_R = linear(phi4, sh_Bz[i_l][j_l + 1], phi2, sh_Bz[i_l][j_l], phi1, sh_Bz[i_l][j_l - 1], phi_g);

                // Скорости Vx, Vy
                if (true)
                {
                    double Vr_L, Vphi_L, Vr_R, Vphi_R;

                    // Vr
                    if (true)
                    {
                        double Vr1 = sh_Vx[i_l][j_l - 1] * cos(phi1) + sh_Vy[i_l][j_l - 1] * sin(phi1);
                        double Vr2 = sh_Vx[i_l][j_l] * cos(phi2) + sh_Vy[i_l][j_l] * sin(phi2);
                        double Vr3 = sh_Vx[i_l][j_l - 2] * cos(phi3) + sh_Vy[i_l][j_l - 2] * sin(phi3);
                        double Vr4 = sh_Vx[i_l][j_l + 1] * cos(phi4) + sh_Vy[i_l][j_l + 1] * sin(phi4);

                        Vr_L = linear(phi3, Vr3, phi1, Vr1, phi2, Vr2, phi_g);
                        Vr_R = linear(phi4, Vr4, phi2, Vr2, phi1, Vr1, phi_g);
                    }

                    // Vphi
                    if (true)
                    {
                        double Vphi1 = -sh_Vx[i_l][j_l - 1] * sin(phi1) + sh_Vy[i_l][j_l - 1] * cos(phi1);
                        double Vphi2 = -sh_Vx[i_l][j_l] * sin(phi2) + sh_Vy[i_l][j_l] * cos(phi2);
                        double Vphi3 = -sh_Vx[i_l][j_l - 2] * sin(phi3) + sh_Vy[i_l][j_l - 2] * cos(phi3);
                        double Vphi4 = -sh_Vx[i_l][j_l + 1] * sin(phi4) + sh_Vy[i_l][j_l + 1] * cos(phi4);

                        Vphi_L = linear(phi3, Vphi3, phi1, Vphi1, phi2, Vphi2, phi_g);
                        Vphi_R = linear(phi4, Vphi4, phi2, Vphi2, phi1, Vphi1, phi_g);
                    }

                    Vx_L = Vr_L * cos(phi_g) - Vphi_L * sin(phi_g);
                    Vy_L = Vr_L * sin(phi_g) + Vphi_L * cos(phi_g);

                    Vx_R = Vr_R * cos(phi_g) - Vphi_R * sin(phi_g);
                    Vy_R = Vr_R * sin(phi_g) + Vphi_R * cos(phi_g);
                }

                // Магнитные поля Bx, By
                if (true)
                {
                    double Br_L, Bphi_L, Br_R, Bphi_R;

                    // Br
                    if (true)
                    {
                        double Br1 = sh_Bx[i_l][j_l - 1] * cos(phi1) + sh_By[i_l][j_l - 1] * sin(phi1);
                        double Br2 = sh_Bx[i_l][j_l] * cos(phi2) + sh_By[i_l][j_l] * sin(phi2);
                        double Br3 = sh_Bx[i_l][j_l - 2] * cos(phi3) + sh_By[i_l][j_l - 2] * sin(phi3);
                        double Br4 = sh_Bx[i_l][j_l + 1] * cos(phi4) + sh_By[i_l][j_l + 1] * sin(phi4);

                        Br_L = linear(phi3, Br3, phi1, Br1, phi2, Br2, phi_g);
                        Br_R = linear(phi4, Br4, phi2, Br2, phi1, Br1, phi_g);
                    }

                    // Bphi
                    if (true)
                    {
                        double Bphi1 = -sh_Bx[i_l][j_l - 1] * sin(phi1) + sh_By[i_l][j_l - 1] * cos(phi1);
                        double Bphi2 = -sh_Bx[i_l][j_l] * sin(phi2) + sh_By[i_l][j_l] * cos(phi2);
                        double Bphi3 = -sh_Bx[i_l][j_l - 2] * sin(phi3) + sh_By[i_l][j_l - 2] * cos(phi3);
                        double Bphi4 = -sh_Bx[i_l][j_l + 1] * sin(phi4) + sh_By[i_l][j_l + 1] * cos(phi4);

                        Bphi_L = linear(phi3, Bphi3, phi1, Bphi1, phi2, Bphi2, phi_g);
                        Bphi_R = linear(phi4, Bphi4, phi2, Bphi2, phi1, Bphi1, phi_g);
                    }

                    Bx_L = Br_L * cos(phi_g) - Bphi_L * sin(phi_g);
                    By_L = Br_L * sin(phi_g) + Bphi_L * cos(phi_g);

                    Bx_R = Br_R * cos(phi_g) - Bphi_R * sin(phi_g);
                    By_R = Br_R * sin(phi_g) + Bphi_R * cos(phi_g);
                }
            }

            // Считаем поток
            if (true)
            {
                // Надо будет ещё подпроавить Bn в ячейке
                double PQ = 0.0;
                double P[8];
                P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;

                tmin = my_min(tmin, HLLDQ_Korolkov(rho_L, 0.0, const_p * rho_L, Vx_L, Vy_L, Vz_L, 0.0, 0.0, 0.0,
                    rho_R, 0.0, const_p * rho_R, Vx_R, Vy_R, Vz_R, 0.0, 0.0, 0.0,
                    P, PQ, -sin(phi_g), cos(phi_g), 0.0, DPHI(j) * r, 1));

                int idx_h = i * (M + 1) + (j);
                h_Prho[idx_h] = P[0];
                h_Pvx[idx_h] = P[1];
                h_Pvy[idx_h] = P[2];
                h_Pvz[idx_h] = P[3];
                h_Pbx[idx_h] = P[4];
                h_Pby[idx_h] = P[5];
                h_Pbz[idx_h] = P[6];
            }
        }
    }


}

void test_polar_geometry(void)
{
    // Печатает именно границы всех ячеек

    ofstream fout;
    fout.open("setka_geometri.txt");



    fout << "TITLE = \"HP\"  VARIABLES = \"X\", \"Y\",  ZONE T = \"HP\", N = " << 4 * K //
        << " , E = " << K << ", F = FEPOINT, ET = quadrilateral" << endl;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            double r1, r2, phi1, phi2;
            phi1 = PHI_LEFT(j);
            phi2 = PHI_RIGHT(j);
            r1 = R_EDGE(i);
            r2 = R_EDGE(i + 1);

            double x, y;

            x = r1 * cos(phi1);
            y = r1 * sin(phi1);
            fout << x << " " << y << endl;

            x = r2 * cos(phi1);
            y = r2 * sin(phi1);
            fout << x << " " << y << endl;

            x = r2 * cos(phi2);
            y = r2 * sin(phi2);
            fout << x << " " << y << endl;

            x = r1 * cos(phi2);
            y = r1 * sin(phi2);
            fout << x << " " << y << endl;
        }
    }


    for (int k = 0; k < K; k = k + 1)
    {
        fout << 4 * k + 1 << " " << 4 * k + 2 << " " << 4 * k + 3 << " " << 4 * k + 4  << endl;
    }

    fout.close();
}



int main(void)
{
    bool read_setka = false;                     // Нужно ли считывать сетку
    string name1 = "save_zOph_1(350x256).bin";   // Откуда скачиваем сетку
    string name2 = "save_zOph_psi_2(350x256).bin";   // Куда сохраняем сетку
    int all_step = 24000 * 60 * 9; // Число шагов


    const int cellCount = N * M;
    const int hFaceCount = N * (M + 1);        // горизонтальные грани
    const int vFaceCount = (N + 1) * M;        // вертикальные грани
    const int nodeCount = (N + 1) * (M + 1);


    // Хост
    CellVars  h_cell;
    FaceVars  h_hFace;   // горизонтальные грани
    FaceVars  h_vFace;   // вертикальные грани
    NodeVars  h_node;

    // Девайс
    CellVars  d_cell;
    FaceVars  d_hFace;
    FaceVars  d_vFace;
    NodeVars  d_node;

    // Выделение памяти для всех массивов
    if (true)
    {
        // --- Ячейки (7 массивов по cellCount) ---
        h_cell.rho = allocateHost<double>(cellCount);
        h_cell.Vx = allocateHost<double>(cellCount);
        h_cell.Vy = allocateHost<double>(cellCount);
        h_cell.Vz = allocateHost<double>(cellCount);
        h_cell.Bx = allocateHost<double>(cellCount);
        h_cell.By = allocateHost<double>(cellCount);
        h_cell.Bz = allocateHost<double>(cellCount);

        d_cell.rho = allocateDevice<double>(cellCount);
        d_cell.Vx = allocateDevice<double>(cellCount);
        d_cell.Vy = allocateDevice<double>(cellCount);
        d_cell.Vz = allocateDevice<double>(cellCount);
        d_cell.Bx = allocateDevice<double>(cellCount);
        d_cell.By = allocateDevice<double>(cellCount);
        d_cell.Bz = allocateDevice<double>(cellCount);

        // --- Горизонтальные грани (8 массивов по hFaceCount) ---
        h_hFace.Prho = allocateHost<double>(hFaceCount);
        h_hFace.Pvx = allocateHost<double>(hFaceCount);
        h_hFace.Pvy = allocateHost<double>(hFaceCount);
        h_hFace.Pvz = allocateHost<double>(hFaceCount);
        h_hFace.Pbx = allocateHost<double>(hFaceCount);
        h_hFace.Pby = allocateHost<double>(hFaceCount);
        h_hFace.Pbz = allocateHost<double>(hFaceCount);
        h_hFace.Bn = allocateHost<double>(hFaceCount);

        d_hFace.Prho = allocateDevice<double>(hFaceCount);
        d_hFace.Pvx = allocateDevice<double>(hFaceCount);
        d_hFace.Pvy = allocateDevice<double>(hFaceCount);
        d_hFace.Pvz = allocateDevice<double>(hFaceCount);
        d_hFace.Pbx = allocateDevice<double>(hFaceCount);
        d_hFace.Pby = allocateDevice<double>(hFaceCount);
        d_hFace.Pbz = allocateDevice<double>(hFaceCount);
        d_hFace.Bn = allocateDevice<double>(hFaceCount);

        // --- Вертикальные грани (8 массивов по vFaceCount) ---
        h_vFace.Prho = allocateHost<double>(vFaceCount);
        h_vFace.Pvx = allocateHost<double>(vFaceCount);
        h_vFace.Pvy = allocateHost<double>(vFaceCount);
        h_vFace.Pvz = allocateHost<double>(vFaceCount);
        h_vFace.Pbx = allocateHost<double>(vFaceCount);
        h_vFace.Pby = allocateHost<double>(vFaceCount);
        h_vFace.Pbz = allocateHost<double>(vFaceCount);
        h_vFace.Bn = allocateHost<double>(vFaceCount);

        d_vFace.Prho = allocateDevice<double>(vFaceCount);
        d_vFace.Pvx = allocateDevice<double>(vFaceCount);
        d_vFace.Pvy = allocateDevice<double>(vFaceCount);
        d_vFace.Pvz = allocateDevice<double>(vFaceCount);
        d_vFace.Pbx = allocateDevice<double>(vFaceCount);
        d_vFace.Pby = allocateDevice<double>(vFaceCount);
        d_vFace.Pbz = allocateDevice<double>(vFaceCount);
        d_vFace.Bn = allocateDevice<double>(vFaceCount);

        // --- Узлы (1 массив по nodeCount) ---
        h_node.Ez = allocateHost<double>(nodeCount);
        d_node.Ez = allocateDevice<double>(nodeCount);
    }

    // Копирование всех массивов на device
    if (true)
    {
        // Ячейки
        copyToDevice(d_cell.rho, h_cell.rho, cellCount);
        copyToDevice(d_cell.Vx, h_cell.Vx, cellCount);
        copyToDevice(d_cell.Vy, h_cell.Vy, cellCount);
        copyToDevice(d_cell.Vz, h_cell.Vz, cellCount);
        copyToDevice(d_cell.Bx, h_cell.Bx, cellCount);
        copyToDevice(d_cell.By, h_cell.By, cellCount);
        copyToDevice(d_cell.Bz, h_cell.Bz, cellCount);

        // Горизонтальные грани
        copyToDevice(d_hFace.Prho, h_hFace.Prho, hFaceCount);
        copyToDevice(d_hFace.Pvx, h_hFace.Pvx, hFaceCount);
        copyToDevice(d_hFace.Pvy, h_hFace.Pvy, hFaceCount);
        copyToDevice(d_hFace.Pvz, h_hFace.Pvz, hFaceCount);
        copyToDevice(d_hFace.Pbx, h_hFace.Pbx, hFaceCount);
        copyToDevice(d_hFace.Pby, h_hFace.Pby, hFaceCount);
        copyToDevice(d_hFace.Pbz, h_hFace.Pbz, hFaceCount);
        copyToDevice(d_hFace.Bn, h_hFace.Bn, hFaceCount);

        // Вертикальные грани
        copyToDevice(d_vFace.Prho, h_vFace.Prho, vFaceCount);
        copyToDevice(d_vFace.Pvx, h_vFace.Pvx, vFaceCount);
        copyToDevice(d_vFace.Pvy, h_vFace.Pvy, vFaceCount);
        copyToDevice(d_vFace.Pvz, h_vFace.Pvz, vFaceCount);
        copyToDevice(d_vFace.Pbx, h_vFace.Pbx, vFaceCount);
        copyToDevice(d_vFace.Pby, h_vFace.Pby, vFaceCount);
        copyToDevice(d_vFace.Pbz, h_vFace.Pbz, vFaceCount);
        copyToDevice(d_vFace.Bn, h_vFace.Bn, vFaceCount);

        // Узлы
        copyToDevice(d_node.Ez, h_node.Ez, nodeCount);
    }

    // FREE_HOST(h_cell.rho);
    // FREE_DEVICE(d_cell.rho);

    // Глобальный цикл
    for (int step_ = 1; step_ <= all_step; step_++)
    {
        dim3 block(BX, BY);
        dim3 grid((N + BX - 1) / BX, (M + BY - 1) / BY);
        compute_fluxes << <grid, block >> > (
            d_cell.rho, d_cell.Vx, d_cell.Vy, d_cell.Vz,
            d_cell.Bx, d_cell.By, d_cell.Bz,
            d_hFace.Prho, d_hFace.Pvx, d_hFace.Pvy, d_hFace.Pvz,
            d_hFace.Pbx, d_hFace.Pby, d_hFace.Pbz, d_hFace.Bn,
            d_vFace.Prho, d_vFace.Pvx, d_vFace.Pvy, d_vFace.Pvz,
            d_vFace.Pbx, d_vFace.Pby, d_vFace.Pbz, d_vFace.Bn
            );
    }


    return 0;
}