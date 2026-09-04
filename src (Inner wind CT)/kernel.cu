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
#define print_i (0)           // предполагаем, что M чётное
#define print_j (10000000)           // предполагаем, что M чётное

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
__host__ __device__ double minmod(double x, double y);
__host__ __device__ double linear(double x1, double t1, double x2, double t2, double x3, double t3, double y);
__device__ void linear2(double x1, double t1, double x2, double t2, double x3, double t3, double y1, double y2,//
    double& A, double& B);

using namespace std;

// Переменные в центрах ячеек
struct CellVars {
    double* rho, * Vx, * Vy, * Vz, * Bx, * By, * Bz, * dVr;
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

template<typename T>
void copyFromDevice(T* h_ptr, const T* d_ptr, int count) {
    cudaMemcpy(h_ptr, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
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


__device__ double get_cell(const double* field, int N_, int M_, int i, int j) 
{
    // Экстраполяция: clamp к допустимым индексам
    if (i == -1) i = 0;
    if (i == -2) i = 1;
    if (i < -2) i = 0;

    if (i == N_) i = N_ - 1;
    if (i == N_ + 1) i = N_ - 2;
    if (i > N_) i = N_ - 1;


    if (j == -1) j = 0;
    if (j == -2) j = 1;
    if (j < -2) j = 0;


    if (j == M_) j = M_ - 1;
    if (j == M_ + 1) j = M_ - 2;
    if (j > M_) j = M_ - 1;

    return field[j * N_ + i];
}


__global__ void compute_fluxes(
    const double* rho, const double* Vx, const double* Vy, const double* Vz,
    const double* Bx, const double* By, const double* Bz, double* dVr,
    double* h_Prho, double* h_Pvx, double* h_Pvy, double* h_Pvz,
    double* h_Pbx, double* h_Pby, double* h_Pbz, const double* h_Bn,
    double* v_Prho, double* v_Pvx, double* v_Pvy, double* v_Pvz,
    double* v_Pbx, double* v_Pby, double* v_Pbz, const double* v_Bn, 
    double* dT)
{
    // Shared memory для всех 7 переменных ячеек
    __shared__ double sh_rho[SHARED_I][SHARED_J];
    __shared__ double sh_Vx[SHARED_I][SHARED_J];
    __shared__ double sh_Vy[SHARED_I][SHARED_J];
    __shared__ double sh_Vz[SHARED_I][SHARED_J];
    __shared__ double sh_Bx[SHARED_I][SHARED_J];
    __shared__ double sh_By[SHARED_I][SHARED_J];
    __shared__ double sh_Bz[SHARED_I][SHARED_J];
    __shared__ double sh_dt_min[BX * BY]; // размер BLOCK_SIZE = BX*BY

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

    sh_dt_min[tx + ty * BX] = 1.0E30;
    __syncthreads();

    // Локальные координаты в shared (с учётом запаса)
    // это координаты текущей ячейки в shared памяти (это важно, так как через них можно брать координаты соседей и т.д.)
    int i_l = tx + HALO;
    int j_l = ty + HALO;

    double tmin = 1.0E30;


    // Считаем потоки
    if (i < N && j < M)
    {
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

                    sh_rho[i_l][j_l + 2] = sh_rho[i_l][j_l + 1];
                    sh_Vy[i_l][j_l + 2] = sh_Vy[i_l][j_l + 1];
                    sh_By[i_l][j_l + 2] = sh_By[i_l][j_l + 1];
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

                tmin = my_min(tmin, HLLDQ_Korolkov(rho_L, 0.0, const_p * rho_L, Vx_L, Vy_L, Vz_L, Bx_L, By_L, Bz_L,
                    rho_R, 0.0, const_p * rho_R, Vx_R, Vy_R, Vz_R, Bx_R, By_R, Bz_R,
                    P, PQ, -sin(phi_g), cos(phi_g), 0.0, DPHI(j) * r, 1));

                /*if (i == 3 && j == 3)
                {
                    printf("AAA = %E, %E, %E, %E, %E, %E, %E, %E, %E, %E, %E \n", rho_L, rho_R, Vx_L, Vy_L, Vz_L, Vx_R, Vy_R, Vz_R, P[0], P[1], P[2]);
                }*/

                int idx_h = (j + 1) * N + i;
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

                int idx_h = j * N + i;
                h_Prho[idx_h] = P[0];
                h_Pvx[idx_h] = P[1];
                h_Pvy[idx_h] = P[2];
                h_Pvz[idx_h] = P[3];
                h_Pbx[idx_h] = P[4];
                h_Pby[idx_h] = P[5];
                h_Pbz[idx_h] = P[6];
            }
        }

        // 3. Правая вертикальная грань (она заполняется для всех ячеек)
        if (true)
        {
            double phi_g = PHI_CENTER(j);
            double r_g = R_EDGE(i + 1);

            if (true)
            {
                double r2, r3, r4;

                r2 = R_CENTER(i + 1, j);
                r3 = R_CENTER(i - 1, j);
                r4 = R_CENTER(i + 2, j);
                if (i == N - 1)
                {
                    r2 = 2 * Rb - r;
                    r4 = 2 * Rb - r3;
                }
                else if (i == N - 2)
                {
                    r4 = 2 * Rb - r2;
                }

                // Здесь фактически задаются граничные условия
                if (i == 0)
                {
                    r3 = 1.0;

                    double Vr, Vphi = 0.0;
                    // Vr
                    if (true)
                    {
                        double Vr1 = sh_Vx[i_l][j_l] * cos(phi_g) + sh_Vy[i_l][j_l] * sin(phi_g);
                        double Vr2 = sh_Vx[i_l + 1][j_l] * cos(phi_g) + sh_Vy[i_l + 1][j_l] * sin(phi_g);
                        Vr = Vr1 + (Vr2 - Vr1) / (r2 - r) * (r3 - r);
                        if (Vr < 0.000001) Vr = Vr1;
                        if (Vr <= 0.0) Vr = 0.0;
                    }

                    /*if (fabs(phi_g) > phi_init && fabs(Br + Br_dipole) > 0.0000001)
                    {
                        Vphi = Vr * (Bphi + Bphi_dipole) / (Br + Br_dipole);
                    }*/

                    sh_rho[i_l - 1][j_l] = rho_in;
                    sh_Vx[i_l - 1][j_l] = (Vr * cos(phi_g) - Vphi * sin(phi_g));
                    sh_Vy[i_l - 1][j_l] = (Vr * sin(phi_g) + Vphi * cos(phi_g));
                    sh_Vz[i_l - 1][j_l] = V_phi_init * sin(pi / 2.0 - phi_g);
                }



                rho_L = linear(r3, sh_rho[i_l - 1][j_l] * kv(r3), r, sh_rho[i_l][j_l] * kv(r), r2, sh_rho[i_l + 1][j_l] * kv(r2), r_g) / kv(r_g);
                if (rho_L <= 0.0) rho_L = sh_rho[i_l][j_l];
                rho_R = linear(r4, sh_rho[i_l + 2][j_l] * kv(r4), r2, sh_rho[i_l + 1][j_l] * kv(r2), r, sh_rho[i_l][j_l] * kv(r), r_g) / kv(r_g);
                if (rho_R <= 0.0) rho_R = sh_rho[i_l + 1][j_l];

                Vz_L = linear(r3, sh_Vz[i_l - 1][j_l], r, sh_Vz[i_l][j_l], r2, sh_Vz[i_l + 1][j_l], r_g);
                Vz_R = linear(r4, sh_Vz[i_l + 2][j_l], r2, sh_Vz[i_l + 1][j_l], r, sh_Vz[i_l][j_l], r_g);

                Bz_L = linear(r3, sh_Bz[i_l - 1][j_l], r, sh_Bz[i_l][j_l], r2, sh_Bz[i_l + 1][j_l], r_g);
                Bz_R = linear(r4, sh_Bz[i_l + 2][j_l], r2, sh_Bz[i_l + 1][j_l], r, sh_Bz[i_l][j_l], r_g);


                // Скорости Vx, Vy
                if (true)
                {
                    double Vr_L, Vphi_L, Vr_R, Vphi_R;

                    // Vr
                    if (true)
                    {
                        double Vr1 = sh_Vx[i_l][j_l] * cos(phi_g) + sh_Vy[i_l][j_l] * sin(phi_g);
                        double Vr2 = sh_Vx[i_l + 1][j_l] * cos(phi_g) + sh_Vy[i_l + 1][j_l] * sin(phi_g);
                        double Vr3 = sh_Vx[i_l - 1][j_l] * cos(phi_g) + sh_Vy[i_l - 1][j_l] * sin(phi_g);
                        double Vr4 = sh_Vx[i_l + 2][j_l] * cos(phi_g) + sh_Vy[i_l + 2][j_l] * sin(phi_g);

                        // Сохраняем dVr/dr для дальнейшего вычисления силы в ячейке
                        if (true)
                        {
                            double h1 = r - r3;
                            double h2 = r2 - r;
                            double dVr_;
                            //dVr_ = (h1 * h1 * Vr2 + (h2 * h2 - h1 * h1) * Vr1 - h2 * h2 * Vr3) / (h1 * h2 * (h1 + h2));  // Второй порядок
                            dVr_ = (Vr2 - Vr1) / h2;    // Первый порядок вправо
                            dVr[j * N + i] = dVr_;


                            double fline = F_line * sh_rho[i_l][j_l] * pow(fabs(dVr_) / sh_rho[i_l][j_l], alpha_line) / kv(r);
                            tmin = my_min(tmin, krit * h2 / (alpha_line * fline / max(fabs(dVr_), 0.0005)));
                        }
                        

                        Vr_L = linear(r3, Vr3, r, Vr1, r2, Vr2, r_g);
                        Vr_R = linear(r4, Vr4, r2, Vr2, r, Vr1, r_g);
                    }

                    // Vphi
                    if (true)
                    {
                        double Vr = -sh_Vx[i_l][j_l] * sin(phi_g) + sh_Vy[i_l][j_l] * cos(phi_g);
                        double Vr2 = -sh_Vx[i_l + 1][j_l] * sin(phi_g) + sh_Vy[i_l + 1][j_l] * cos(phi_g);
                        double Vr3 = -sh_Vx[i_l - 1][j_l] * sin(phi_g) + sh_Vy[i_l - 1][j_l] * cos(phi_g);
                        double Vr4 = -sh_Vx[i_l + 2][j_l] * sin(phi_g) + sh_Vy[i_l + 2][j_l] * cos(phi_g);

                        Vphi_L = linear(r3, Vr3, r, Vr, r2, Vr2, r_g);
                        Vphi_R = linear(r4, Vr4, r2, Vr2, r, Vr, r_g);
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
                        double Br1 = sh_Bx[i_l][j_l] * cos(phi_g) + sh_By[i_l][j_l] * sin(phi_g);
                        double Br2 = sh_Bx[i_l + 1][j_l] * cos(phi_g) + sh_By[i_l + 1][j_l] * sin(phi_g);
                        double Br3 = sh_Bx[i_l - 1][j_l] * cos(phi_g) + sh_By[i_l - 1][j_l] * sin(phi_g);
                        double Br4 = sh_Bx[i_l + 2][j_l] * cos(phi_g) + sh_By[i_l + 2][j_l] * sin(phi_g);

                        Br_L = linear(r3, Br3, r, Br1, r2, Br2, r_g);
                        Br_R = linear(r4, Br4, r2, Br2, r, Br1, r_g);
                    }

                    // Bphi
                    if (true)
                    {
                        double Br = -sh_Bx[i_l][j_l] * sin(phi_g) + sh_By[i_l][j_l] * cos(phi_g);
                        double Br2 = -sh_Bx[i_l + 1][j_l] * sin(phi_g) + sh_By[i_l + 1][j_l] * cos(phi_g);
                        double Br3 = -sh_Bx[i_l - 1][j_l] * sin(phi_g) + sh_By[i_l - 1][j_l] * cos(phi_g);
                        double Br4 = -sh_Bx[i_l + 2][j_l] * sin(phi_g) + sh_By[i_l + 2][j_l] * cos(phi_g);

                        Bphi_L = linear(r3, Br3, r, Br, r2, Br2, r_g);
                        Bphi_R = linear(r4, Br4, r2, Br2, r, Br, r_g);
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
                // Надо будет ещё подправить Bn в ячейке
                double PQ = 0.0;
                double P[8];
                P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;

                tmin = my_min(tmin, HLLDQ_Korolkov(rho_L, 0.0, const_p * rho_L, Vx_L, Vy_L, Vz_L, 0.0, 0.0, 0.0,
                    rho_R, 0.0, const_p * rho_R, Vx_R, Vy_R, Vz_R, 0.0, 0.0, 0.0,
                    P, PQ, cos(phi_g), sin(phi_g), 0.0, DR(i), 1));

                if (i == print_i && j == print_j)
                {
                    printf("Gran 0;100 := %E, %E, %E, %E, %E, %E, %E, %E, %E, %E, %E \n", rho_L, rho_R, Vx_L, Vy_L, Vz_L, Vx_R, Vy_R, Vz_R, P[0], P[1], P[2]);
                    printf("and := %E, %E, %E, %E \n", sh_rho[i_l][j_l], sh_rho[i_l + 1][j_l], sh_rho[i_l + 2][j_l], rho[j * N + i]);
                }

                int idx_h = j * (N + 1) + (i + 1);
                v_Prho[idx_h] = P[0];
                v_Pvx[idx_h] = P[1];
                v_Pvy[idx_h] = P[2];
                v_Pvz[idx_h] = P[3];
                v_Pbx[idx_h] = P[4];
                v_Pby[idx_h] = P[5];
                v_Pbz[idx_h] = P[6];
            }
        }

        // 4. Левая вертикальная грань (она заполняется только у левого ряда ячеек)
        if (i == 0)
        {
            double phi_g = PHI_CENTER(j);
            double r_g = 1.0;

            // Сносим переменные на грани minmod
            if (true)
            {
                double r1, r2, r4;
                r1 = 1.0;
                r2 = r;
                r4 = R_CENTER(i + 1, j);


                rho_L = sh_rho[i_l - 1][j_l];
                rho_R = linear(r4, sh_rho[i_l + 1][j_l] * kv(r4), r2, sh_rho[i_l][j_l] * kv(r2), r1, sh_rho[i_l - 1][j_l] * kv(r1), r_g) / kv(r_g);
                if (rho_R <= 0.0) rho_R = sh_rho[i_l][j_l];

                Vz_L = sh_Vz[i_l - 1][j_l];
                Vz_R = linear(r4, sh_Vz[i_l + 1][j_l], r2, sh_Vz[i_l][j_l], r1, sh_Vz[i_l - 1][j_l], r_g);

                Bz_L = sh_Bz[i_l - 1][j_l];
                Bz_R = linear(r4, sh_Bz[i_l + 1][j_l], r2, sh_Bz[i_l][j_l], r1, sh_Bz[i_l - 1][j_l], r_g);

                // Скорости Vx, Vy
                if (true)
                {
                    double Vr_R, Vphi_R;

                    // Vr
                    if (true)
                    {
                        double Vr1 = sh_Vx[i_l - 1][j_l] * cos(phi_g) + sh_Vy[i_l - 1][j_l] * sin(phi_g);
                        double Vr2 = sh_Vx[i_l][j_l] * cos(phi_g) + sh_Vy[i_l][j_l] * sin(phi_g);
                        double Vr4 = sh_Vx[i_l + 1][j_l] * cos(phi_g) + sh_Vy[i_l + 1][j_l] * sin(phi_g);

                        Vr_R = linear(r4, Vr4, r2, Vr2, r1, Vr1, r_g);
                    }

                    // Vphi
                    if (true)
                    {
                        double Vr1 = -sh_Vx[i_l - 1][j_l] * sin(phi_g) + sh_Vy[i_l - 1][j_l] * cos(phi_g);
                        double Vr2 = -sh_Vx[i_l][j_l] * sin(phi_g) + sh_Vy[i_l][j_l] * cos(phi_g);
                        double Vr4 = -sh_Vx[i_l + 1][j_l] * sin(phi_g) + sh_Vy[i_l + 1][j_l] * cos(phi_g);

                        Vphi_R = linear(r4, Vr4, r2, Vr2, r1, Vr1, r_g);
                    }

                    Vx_L = sh_Vx[i_l - 1][j_l];
                    Vy_L = sh_Vy[i_l - 1][j_l];

                    Vx_R = Vr_R * cos(phi_g) - Vphi_R * sin(phi_g);
                    Vy_R = Vr_R * sin(phi_g) + Vphi_R * cos(phi_g);
                }

                // Магнитные поля Bx, By
                if (true)
                {
                    double Br_R, Bphi_R;

                    // Br
                    if (true)
                    {
                        double Br1 = sh_Bx[i_l][j_l] * cos(phi_g) + sh_By[i_l][j_l] * sin(phi_g);
                        double Br2 = sh_Bx[i_l + 1][j_l] * cos(phi_g) + sh_By[i_l + 1][j_l] * sin(phi_g);
                        double Br4 = sh_Bx[i_l + 2][j_l] * cos(phi_g) + sh_By[i_l + 2][j_l] * sin(phi_g);

                        Br_R = linear(r4, Br4, r2, Br2, r1, Br1, r_g);
                    }

                    // Bphi
                    if (true)
                    {
                        double Br1 = -sh_Bx[i_l][j_l] * sin(phi_g) + sh_By[i_l][j_l] * cos(phi_g);
                        double Br2 = -sh_Bx[i_l + 1][j_l] * sin(phi_g) + sh_By[i_l + 1][j_l] * cos(phi_g);
                        double Br4 = -sh_Bx[i_l + 2][j_l] * sin(phi_g) + sh_By[i_l + 2][j_l] * cos(phi_g);

                        Bphi_R = linear(r4, Br4, r2, Br2, r1, Br1, r_g);
                    }

                    Bx_L = sh_Bx[i_l - 1][j_l];
                    By_L = sh_By[i_l - 1][j_l];

                    Bx_R = Br_R * cos(phi_g) - Bphi_R * sin(phi_g);
                    By_R = Br_R * sin(phi_g) + Bphi_R * cos(phi_g);
                }
            }

            // Считаем поток
            if (true)
            {
                // Надо будет ещё подправить Bn в ячейке
                double PQ = 0.0;
                double P[8];
                P[0] = P[1] = P[2] = P[3] = P[4] = P[5] = P[6] = P[7] = 0.0;

                tmin = my_min(tmin, HLLDQ_Korolkov(rho_L, 0.0, const_p * rho_L, Vx_L, Vy_L, Vz_L, 0.0, 0.0, 0.0,
                    rho_R, 0.0, const_p * rho_R, Vx_R, Vy_R, Vz_R, 0.0, 0.0, 0.0,
                    P, PQ, cos(phi_g), sin(phi_g), 0.0, DR(i), 1));

                int idx_h = j * (N + 1) + i;
                v_Prho[idx_h] = P[0];
                v_Pvx[idx_h] = P[1];
                v_Pvy[idx_h] = P[2];
                v_Pvz[idx_h] = P[3];
                v_Pbx[idx_h] = P[4];
                v_Pby[idx_h] = P[5];
                v_Pbz[idx_h] = P[6];
            }
        }
    }

    //atomicMinDouble(dT, tmin);

    // Найдём минимальное время
    if (true)
    {
        // 1. Редукция внутри блока
        int tid = tx + ty * BX;
        sh_dt_min[tid] = tmin;
        __syncthreads();

        // Пошаговая редукция (для примера используем простой метод с уменьшением вдвое)
        for (int stride = (BX * BY) / 2; stride > 0; stride >>= 1) 
        {
            if (tid < stride) {
                double other = sh_dt_min[tid + stride];
                if (other < sh_dt_min[tid]) sh_dt_min[tid] = other;
            }
            __syncthreads();
        }

        // 2. Запись минимума блока в глобальную переменную через атомик
        if (tid == 0) 
        {
            atomicMinDouble(dT, sh_dt_min[0]);
        }
    }
}

__global__ void update_cells(
    double* rho, double* Vx, double* Vy, double* Vz,
    double* Bx, double* By, double* Bz, const double* dVr,
    const double* h_Prho, const double* h_Pvx, const double* h_Pvy, const double* h_Pvz,
    const double* h_Pbx, const double* h_Pby, const double* h_Pbz,
    const double* v_Prho, const double* v_Pvx, const double* v_Pvy, const double* v_Pvz,
    const double* v_Pbx, const double* v_Pby, const double* v_Pbz,
    const double* dT)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= M) return;

    // Индексы ячейки
    int idx = j * N + i;

    double rho_1 = rho[idx];
    double Vx_1 = Vx[idx];
    double Vy_1 = Vy[idx];
    double Vz_1 = Vz[idx];
    double Bx_1 = Bx[idx];
    double By_1 = By[idx];
    double Bz_1 = Bz[idx];
    double dV = CELL_AREA(i, j);


    double S1, S2, S3, rho_2, Vx_2, Vy_2, Vz_2;
    S1 = DPHI(j) * R_EDGE(i + 1);
    S2 = DPHI(j) * R_EDGE(i);
    S3 = DR(i);
    double phi = PHI_CENTER(j);
    double r = R_CENTER(i, j);
    double x = r * cos(phi);

    double Fx = 0.0, Fy = 0.0;

    double dTime = *dT; // 1.0E-4; // *dT;

    // Вычисляем силы
    if (true)
    {
        double fr = 0.0;
        fr = (F_grav + F_continuum) * rho_1 / kv(r);  // Сила притяжения к звезде + радиационное отталкивание от континуума

        if (true)
        {
            double dVrdr = dVr[idx];
            double Vr1 = Vx_1 * cos(phi) + Vy_1 * sin(phi);

            double sigma = fabs(dVrdr * r / Vr1) - 1.0; 
            double muc = 1.0 - 1.0 / kv(r);

            double ff = 1.0;

            if (fabs(dVrdr) > 0.00001)
            {
                ff = (pow(1.0 + sigma, 1.0 + alpha_line) - pow(1.0 + sigma * muc, 1.0 + alpha_line)) /
                    ((1.0 + alpha_line) * (1.0 - muc) * sigma * pow(1.0 + sigma, alpha_line));

                if (isnan(ff) == true)
                {
                    printf("Problems ff = %lf, %lf, %lf \n", ff, sigma, muc);
                }
            }

            double fline = F_line * ff * rho_1 * pow(fabs(dVrdr) / rho_1, alpha_line) / kv(r);
            fr += fline;
        }


        Fx = fr * cos(phi);
        Fy = fr * sin(phi);
    }

    double ppp = 0.0;
    // Плотность
    if (true)
    {
        double P = 0.0;

        P += v_Prho[j * (N + 1) + (i + 1)] * S1;   // r+
        P -= v_Prho[j * (N + 1) + i] * S2;       // r-
        P += h_Prho[(j + 1) * N + i] * S3;  // phi+
        P -= h_Prho[j * N + i] * S3;      // phi-

        if (i == print_i && j == print_j)
        {
            printf("POTOK rho: %E, %E, %E, %E \n ", v_Prho[j * (N + 1) + (i + 1)], -v_Prho[j * (N + 1) + i], h_Prho[(j + 1) * N + i], -h_Prho[j * N + i]);
        }

        ppp = P;
        rho_2 = rho_1 - dTime * (P / dV + rho_1 * Vx_1 / x);
        rho[idx] = rho_2;
    }

    // Vx
    
    if (true)
    {
        double P = 0.0;

        P += v_Pvx[j * (N + 1) + (i + 1)] * S1;   // r+
        P -= v_Pvx[j * (N + 1) + i] * S2;       // r-
        P += h_Pvx[(j + 1) * N + i] * S3;  // phi+
        P -= h_Pvx[j * N + i] * S3;      // phi-

        if (i == print_i && j == print_j)
        {
            printf("POTOK vx: %E, %E, %E, %E \n ", v_Pvx[j * (N + 1) + (i + 1)], -v_Pvx[j * (N + 1) + i], h_Pvx[(j + 1) * N + i], -h_Pvx[j * N + i]);
        }

        Vx_2 = (rho_1 * Vx_1 - dTime * P / dV + dTime * (rho_1 * (kv(Vz_1) - kv(Vx_1)) + (kv(Bx_1) - kv(Bz_1)) / cpi4) / x + dTime * Fx) / rho_2;
        Vx[idx] = Vx_2;
    }

    // Vy
    if (true)
    {
        double P = 0.0;

        P += v_Pvy[j * (N + 1) + (i + 1)] * S1;   // r+
        P -= v_Pvy[j * (N + 1) + i] * S2;       // r-
        P += h_Pvy[(j + 1) * N + i] * S3;  // phi+
        P -= h_Pvy[j * N + i] * S3;      // phi-

        if (i == print_i && j == print_j)
        {
            printf("POTOK vy: %E, %E, %E, %E \n ", v_Pvy[j * (N + 1) + (i + 1)], -v_Pvy[j * (N + 1) + i], h_Pvy[(j + 1) * N + i], -h_Pvy[j * N + i]);
        }

        Vy_2 = (rho_1 * Vy_1 - dTime * P / dV - dTime * (rho_1 * Vx_1 * Vy_1 - Bx_1 * By_1 / cpi4) / x + dTime * Fy) / rho_2;
        Vy[idx] = Vy_2;
    }

    // Vz
    if (true)
    {
        double P = 0.0;

        P += v_Pvz[j * (N + 1) + (i + 1)] * S1;   // r+
        P -= v_Pvz[j * (N + 1) + i] * S2;       // r-
        P += h_Pvz[(j + 1) * N + i] * S3;  // phi+
        P -= h_Pvz[j * N + i] * S3;      // phi-

        if (i == print_i && j == print_j)
        {
            printf("POTOK vz: %E, %E, %E, %E \n ", v_Pvz[j * (N + 1) + (i + 1)], -v_Pvz[j * (N + 1) + i], h_Pvz[(j + 1) * N + i], -h_Pvz[j * N + i]);
        }

        Vz_2 = (rho_1 * Vz_1 - dTime * P / dV - 2.0 * dTime * (rho_1 * Vx_1 * Vz_1 - Bx_1 * Bz_1 / cpi4) / x) / rho_2;
        Vz[idx] = Vz_2;
    }

    if (i == print_i && j == print_j)
    {
        printf("CELL 0;100 =: %E, %E, %E, %E, %E, %E, %E \n ", rho_2, Vx_2, Vy_2, Vz_2, Fx, Fy, (ppp / dV + rho_1 * Vx_1 / x));
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
    int all_step = 20000; // 24000 * 60 * 9; // Число шагов
    double host_dT = 1.0E30;
    double host_all_T = 0.0;

    const int cellCount = N * M;
    const int hFaceCount = N * (M + 1);        // горизонтальные грани
    const int vFaceCount = (N + 1) * M;        // вертикальные грани
    const int nodeCount = (N + 1) * (M + 1);

    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

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
    double* dT;

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
        h_cell.dVr = allocateHost<double>(cellCount);

        d_cell.rho = allocateDevice<double>(cellCount);
        d_cell.Vx = allocateDevice<double>(cellCount);
        d_cell.Vy = allocateDevice<double>(cellCount);
        d_cell.Vz = allocateDevice<double>(cellCount);
        d_cell.Bx = allocateDevice<double>(cellCount);
        d_cell.By = allocateDevice<double>(cellCount);
        d_cell.Bz = allocateDevice<double>(cellCount);
        d_cell.dVr = allocateDevice<double>(cellCount);

        cudaMalloc((void**)&dT, sizeof(double));

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

    // Заполнение массивов начальными условиями
    if (true)
    {
        for (int k = 0; k < K; k++)  // Заполняем начальные условия
        {
            int n = k % N;                                   // номер ячейки по x (от 0)
            int m = (k - n) / N;                             // номер ячейки по y (от 0)
            double r, phi;
            r = R_CENTER(n, m);
            phi = PHI_CENTER(m);

            double x = r * cos(phi);
            double y = r * sin(phi);

            double dist = sqrt(x * x + y * y);
            double the = polar_angle(y, x);
            double vr = 0.0009 + pow(max(1.0 - 1.0 / dist, 0.0), 0.71);
            double vphi = V_phi_init * sin(the);
            double rho = rho_in / kv(dist);
            double B0 = Bo_init;

            h_cell.rho[k] = rho;
            h_cell.Vx[k] = vr * x / dist;
            h_cell.Vy[k] = vr * y / dist;
            h_cell.Vz[k] = vphi;
        }
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

    cudaEventRecord(start, 0);
    // Глобальный цикл
    for (int step_ = 1; step_ <= all_step; step_++)
    {
        if (step_ % 10000000 == 0)
        {
            cout << "Step = " << step_ << endl;
        }

        cudaError_t err = cudaMemcpy(dT, &host_dT, sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { printf("error 543thrf3fwrf34r23d324r: %s\n", cudaGetErrorString(err));}
        cudaStatus = cudaDeviceSynchronize();

        dim3 block(BX, BY);
        dim3 grid((N + BX - 1) / BX, (M + BY - 1) / BY);
        compute_fluxes << <grid, block >> > (
            d_cell.rho, d_cell.Vx, d_cell.Vy, d_cell.Vz,
            d_cell.Bx, d_cell.By, d_cell.Bz, d_cell.dVr,
            d_hFace.Prho, d_hFace.Pvx, d_hFace.Pvy, d_hFace.Pvz,
            d_hFace.Pbx, d_hFace.Pby, d_hFace.Pbz, d_hFace.Bn,
            d_vFace.Prho, d_vFace.Pvx, d_vFace.Pvy, d_vFace.Pvz,
            d_vFace.Pbx, d_vFace.Pby, d_vFace.Pbz, d_vFace.Bn, dT);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }

        //if (step_ % 1 == 0)
        if (true)
        {
            cudaMemcpy(&host_dT, dT, sizeof(double), cudaMemcpyDeviceToHost);
            host_all_T += host_dT;
            if (step_ % 1000 == 0)
            {
                cout << "Step = " << step_ <<"   All_Time = " <<  host_all_T * 1.09556 
                    << " hours,  dT =   " << std::scientific << host_dT * 1.09556 << endl;
            }
        }

        cudaStatus = cudaDeviceSynchronize();

        update_cells << <grid, block >> > (
            d_cell.rho, d_cell.Vx, d_cell.Vy, d_cell.Vz,
            d_cell.Bx, d_cell.By, d_cell.Bz, d_cell.dVr,
            d_hFace.Prho, d_hFace.Pvx, d_hFace.Pvy, d_hFace.Pvz,
            d_hFace.Pbx, d_hFace.Pby, d_hFace.Pbz,
            d_vFace.Prho, d_vFace.Pvx, d_vFace.Pvy, d_vFace.Pvz,
            d_vFace.Pbx, d_vFace.Pby, d_vFace.Pbz, dT);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "1  cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            exit(-1);
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time:  %.2f sec\n", elapsedTime / 1000.0);

    if (true)
    {
        copyFromDevice(h_cell.rho, d_cell.rho, cellCount);
        copyFromDevice(h_cell.Vx, d_cell.Vx, cellCount);
        copyFromDevice(h_cell.Vy, d_cell.Vy, cellCount);
        copyFromDevice(h_cell.Vz, d_cell.Vz, cellCount);
        copyFromDevice(h_cell.Bx, d_cell.Bx, cellCount);
        copyFromDevice(h_cell.By, d_cell.By, cellCount);
        copyFromDevice(h_cell.Bz, d_cell.Bz, cellCount);
    }

    // Печатаем результат 2D
    if (true)
    {
        ofstream fout5;
        fout5.open("param_for_texplot_all.txt");



        int nn = (int)((N + Nmin - 1) / Nmin);
        int mm = (int)((M + Nmin - 1) / Nmin);
        fout5 << "TITLE = \"HP\"  VARIABLES = \"X\", \"Y\", \"Ro\", \"Vx\", \"Vy\",\"Vr\", \"Vthe\", \"Vphi\", \"Bx\", \"By\",\"Br\", \"Bthe\", \"Bphi\",  ZONE T = \"HP\", N = " << K //
            << " , E = " << (N - 1) * (M - 1) << ", F = FEPOINT, ET = quadrilateral" << endl;

        for (int k = 0; k < K; k++)
        {
            int i = k % N;                                   // номер ячейки по x (от 0)
            int j = (k - i) / N;                             // номер ячейки по y (от 0)
            double r, phi;
            phi = PHI_CENTER(j);
            r = R_CENTER(i, j);
            //r = R_CENTER(i);

            double x, y;
            x = r * cos(phi);
            y = r * sin(phi);

            double bx = h_cell.Bx[k];// +Bx_dipole(r, phi);
            double by = h_cell.By[k];// +By_dipole(r, phi);


            /*double Max = 0.0, Temp = 0.0, Max_alf = 0.0;
            if (host_s[k].x > 0.0)
            {
                Max = sqrt((host_u[k].x * host_u[k].x + host_u[k].y * host_u[k].y + host_u[k].z * host_u[k].z) / (ggg * host_s[k].y / host_s[k].x));
                Temp = host_s[k].y / host_s[k].x;
                if (sqrt((bx * bx + by * by + host_b[k].z * host_b[k].z)) > 0.00001)
                {
                    Max_alf = sqrt((host_u[k].x * host_u[k].x + host_u[k].y * host_u[k].y + host_u[k].z * host_u[k].z)) * sqrt(4.0 * pi * host_s[k].x) /
                        sqrt((bx * bx + by * by + host_b[k].z * host_b[k].z));
                }
            }

            Max_alf = 0.0;*/

            double Vr = (h_cell.Vx[k] * x + h_cell.Vy[k] * y) / sqrt(x * x + y * y);
            double Vthe = (h_cell.Vx[k] * y - h_cell.Vy[k] * x) / sqrt(x * x + y * y);
            double Br = (bx * x + by * y) / sqrt(x * x + y * y);
            double Bthe = (bx * y - by * x) / sqrt(x * x + y * y);

            fout5 << x << " " << y << " " << h_cell.rho[k] <<//
                " " << h_cell.Vx[k] << " " << h_cell.Vy[k] << " " << Vr << " " << Vthe << " " << h_cell.Vz[k] <<
                " " << bx << " " << by << " " << Br << " " << Bthe << " " << h_cell.Bz[k] << endl;
        }

        for (int i = 0; i < N - 1; i++)
        {
            for (int j = 0; j < M - 1; j++)
            {
                int k1 = j * N + i;
                int k2 = j * N + i + 1;
                int k3 = (j + 1) * N + i + 1;
                int k4 = (j + 1) * N + i;
                fout5 << k1 + 1 << " " << k2 + 1 << " " << k3 + 1 << " " << k4 + 1 << endl;
            }
        }

        fout5.close();
    }

    // Печатаем результат 1D по r    
    if (true)
    {
        ofstream fout5;
        fout5.open("param_for_texplot_all.txt");



        int nn = (int)((N + Nmin - 1) / Nmin);
        int mm = (int)((M + Nmin - 1) / Nmin);
        fout5 << "TITLE = \"HP\"  VARIABLES = \"X\", \"Y\", \"Ro\", \"Vx\", \"Vy\",\"Vr\", \"Vthe\", \"Vphi\", \"Bx\", \"By\",\"Br\", \"Bthe\", \"Bphi\", \"Mach\",  ZONE T = \"HP\", N = " << K //
            << " , E = " << (N - 1) * (M - 1) << ", F = FEPOINT, ET = quadrilateral" << endl;

        for (int k = 0; k < K; k++)
        {
            int i = k % N;                                   // номер ячейки по x (от 0)
            int j = (k - i) / N;                             // номер ячейки по y (от 0)
            double r, phi;
            phi = PHI_CENTER(j);
            r = R_CENTER(i, j);
            //r = R_CENTER(i);

            double x, y;
            x = r * cos(phi);
            y = r * sin(phi);

            double bx = h_cell.Bx[k];// +Bx_dipole(r, phi);
            double by = h_cell.By[k];// +By_dipole(r, phi);


            double Vr = (h_cell.Vx[k] * x + h_cell.Vy[k] * y) / sqrt(x * x + y * y);
            double Vthe = (h_cell.Vx[k] * y - h_cell.Vy[k] * x) / sqrt(x * x + y * y);
            double Br = (bx * x + by * y) / sqrt(x * x + y * y);
            double Bthe = (bx * y - by * x) / sqrt(x * x + y * y);

            double Max = 0.0;
            if (h_cell.rho[k] > 0.0)
            {
                Max = sqrt(( kv(h_cell.Vx[k]) + kv(h_cell.Vy[k]) + kv(h_cell.Vz[k])) / (ggg * const_p));
            }

            fout5 << x << " " << y << " " << h_cell.rho[k] <<//
                " " << h_cell.Vx[k] << " " << h_cell.Vy[k] << " " << Vr << " " << Vthe << " " << h_cell.Vz[k] <<
                " " << bx << " " << by << " " << Br << " " << Bthe << " " << h_cell.Bz[k] << " " << Max << endl;
        }

        for (int i = 0; i < N - 1; i++)
        {
            for (int j = 0; j < M - 1; j++)
            {
                int k1 = j * N + i;
                int k2 = j * N + i + 1;
                int k3 = (j + 1) * N + i + 1;
                int k4 = (j + 1) * N + i;
                fout5 << k1 + 1 << " " << k2 + 1 << " " << k3 + 1 << " " << k4 + 1 << endl;
            }
        }

        fout5.close();
    }

    // Печатаем 1д файл по r
    if (true)
    {
        ofstream fout1dr;
        fout1dr.open("param_for_texplot_1d_r.txt");
        fout1dr << "TITLE = \"HP\"  VARIABLES = \"r\", \"Ro\", \"Vx\", \"Vy\",\"Vr\", \"Vthe\", \"Vphi\", \"Bx\", \"By\",\"Br\", \"Bthe\", \"Bphi\", \"Mach\",  ZONE T = \"HP\"" << endl;

        for (int i = 0; i < N - 1; i++)
        {
            int j = int(M / 2);
            int k = j * N + i;
            double r = R_CENTER(i, j);
            //double r = R_CENTER(i);
            double phi = PHI_CENTER(j);

            double x, y;
            x = r * cos(phi);
            y = r * sin(phi);

            double bx = h_cell.Bx[k];// +Bx_dipole(r, phi);
            double by = h_cell.By[k];// +By_dipole(r, phi);


            double Vr = (h_cell.Vx[k] * x + h_cell.Vy[k] * y) / sqrt(x * x + y * y);
            double Vthe = (h_cell.Vx[k] * y - h_cell.Vy[k] * x) / sqrt(x * x + y * y);
            double Br = (bx * x + by * y) / sqrt(x * x + y * y);
            double Bthe = (bx * y - by * x) / sqrt(x * x + y * y);

            double Max = 0.0;
            if (h_cell.rho[k] > 0.0)
            {
                Max = sqrt((kv(h_cell.Vx[k]) + kv(h_cell.Vy[k]) + kv(h_cell.Vz[k])) / (ggg * const_p));
            }

            fout1dr << r << " " << h_cell.rho[k] <<//
                " " << h_cell.Vx[k] << " " << h_cell.Vy[k] << " " << Vr << " " << Vthe << " " << h_cell.Vz[k] <<
                " " << bx << " " << by << " " << Br << " " << Bthe << " " << h_cell.Bz[k] << " " << Max << endl;
        }

        fout1dr.close();
    }


    return 0;
}