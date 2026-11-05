#include "cooling.h"
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iomanip>
#include <algorithm>

// ============================================================
//  Тест: согласованность Y-таблицы и Lambda-таблицы.
//
//  Проверяет:  dY/dlog10T  ==  -(Lambda_ref/T_ref) * T * ln(10) / Lambda(T)
//
//  Возвращает true, если таблицы согласованы.
//  Печатает статистику: max / медиана / 90-99 перцентили,
//  и «худшие» точки, чтобы было видно, где проблема.
//
//  Требует, чтобы host.logT_min/max/dlogT, host.n,
//  host.T_ref, host.Lambda_ref, host.Y_of_T[], host.logLambda[]
//  были уже заполнены (то есть вызывать ПОСЛЕ read_col и
//  проверок из load_cooling_tables).
// ============================================================
bool test_cooling_consistency(const CoolingTablesHost& host, bool verbose = true)
{
    const double ln10 = std::log(10.0);
    const int    n = host.n;

    // --- 1) Считаем относительную ошибку в каждой внутренней точке ---
    std::vector<double> rel_errs;
    rel_errs.reserve(n - 2);

    struct Worst { double rel, T, dY, exp; int i; } worst{ -1.0, 0, 0, 0, -1 };

    for (int i = 1; i < n - 1; ++i) {
        const double lT = host.logT_min + i * host.dlogT;
        const double T = std::pow(10.0, lT);
        const double Lam = std::pow(10.0, host.logLambda[i]);
        if (!(Lam > 0.0) || !std::isfinite(Lam)) continue;   // защита

        // dY/dlog10T центральной разностью по таблице
        const double dY_tab = (host.Y_of_T[i + 1] - host.Y_of_T[i - 1])
            / (2.0 * host.dlogT);

        // Ожидаемое значение из Lambda-таблицы
        const double dY_exp = -(host.Lambda_ref / host.T_ref)
            * T * ln10 / Lam;

        // Относительная ошибка (по модулю ожидаемого)
        const double rel = std::fabs(dY_tab - dY_exp)
            / std::fabs(dY_exp);
        rel_errs.push_back(rel);

        if (rel > worst.rel) {
            worst = { rel, T, dY_tab, dY_exp, i };
        }
    }

    if (rel_errs.empty()) {
        std::cerr << "[consistency] ERROR: no valid interior points\n";
        return false;
    }

    // --- 2) Статистика ---
    std::sort(rel_errs.begin(), rel_errs.end());
    auto pct = [&](double p) {
        size_t k = static_cast<size_t>(p * (rel_errs.size() - 1));
        return rel_errs[k];
        };
    const double med = pct(0.50);
    const double p90 = pct(0.90);
    const double p99 = pct(0.99);
    const double maxe = rel_errs.back();

    // --- 3) Печать ---
    if (verbose) {
        std::cout << "[consistency] dY/dlog10T vs -(?ref/Tref) T ln10/?(T)\n";
        std::cout << "  points     = " << rel_errs.size() << "\n";
        std::cout << "  max        = " << std::scientific << maxe << "\n";
        std::cout << "  p99        = " << p99 << "\n";
        std::cout << "  p90        = " << p90 << "\n";
        std::cout << "  median     = " << med << "\n";
        if (worst.i >= 0) {
            std::cout << "  worst at i = " << worst.i
                << "  T = " << std::scientific << worst.T << " K\n";
            std::cout << "    dY/dlogT (finite diff) = " << worst.dY << "\n";
            std::cout << "    expected               = " << worst.exp << "\n";
        }
        // несколько точек для наглядности
        std::cout << "  samples:\n";
        std::cout << "      i        T[K]        dY/dlogT      expected      rel\n";
        for (int i : { 2, 50, 200, 400, 600, 800, n - 3 }) {
            if (i < 1 || i >= n - 1) continue;
            const double lT = host.logT_min + i * host.dlogT;
            const double T = std::pow(10.0, lT);
            const double Lam = std::pow(10.0, host.logLambda[i]);
            const double dYt = (host.Y_of_T[i + 1] - host.Y_of_T[i - 1])
                / (2.0 * host.dlogT);
            const double dYe = -(host.Lambda_ref / host.T_ref)
                * T * ln10 / Lam;
            const double rel = std::fabs(dYt - dYe) / std::fabs(dYe);
            std::cout << "  " << std::setw(5) << i
                << "  " << std::setw(11) << std::scientific << T
                << "  " << std::setw(12) << dYt
                << "  " << std::setw(12) << dYe
                << "  " << std::setw(10) << rel << "\n";
        }
    }

    // --- 4) Критерий ---
    // Медиана должна быть на уровне погрешности центральной разности
    // на гладком участке: ~ dlogT^2 ? (0.0049)^2 ? 2.4e-5.
    // Допускаем запас до 1e-3 (учёт изломов ?).
    // 90-й перцентиль до 1e-2 — иначе явная рассинхронизация.
    bool ok = true;
    if (med > 1e-3) {
        std::cerr << "[consistency] FAIL: median rel.err = "
            << med << " > 1e-3  efgergvergvervevr!\n";
        ok = false;
    }
    if (p90 > 1e-2) {
        std::cerr << "[consistency] WARNING: 90th percentile = "
            << p90 << " > 1e-2  erfergfegrthrthrtgser\n";
        // не hard fail: возможны локальные изломы ?
    }
    if (ok && verbose) {
        std::cout << "[consistency] OK\n";
    }

    // Явная проверка на локальные выбросы: считаем точки с rel > 0.05
// (5% — заметно больше погрешности трапеций и изломов ?).
    int n_bad = 0;
    std::vector<int> bad_i;
    for (int i = 1; i < n - 1; ++i) {
        // rel_errs прорежены (иногда continue), так что пересчитаем по исходной формуле
        double lT = host.logT_min + i * host.dlogT;
        double T = std::pow(10.0, lT);
        double Lam = std::pow(10.0, host.logLambda[i]);
        if (!(Lam > 0.0)) continue;
        double dYt = (host.Y_of_T[i + 1] - host.Y_of_T[i - 1]) / (2.0 * host.dlogT);
        double dYe = -(host.Lambda_ref / host.T_ref) * T * ln10 / Lam;
        double rel = std::fabs(dYt - dYe) / std::fabs(dYe);
        if (rel > 0.05) { ++n_bad; bad_i.push_back(i); }
    }
    if (n_bad > 0) {
        std::cerr << "[consistency] WARNING: " << n_bad
            << " outlier point(s) with rel > 5%:\n";
        for (int i : bad_i) {
            std::cerr << "    i=" << i
                << "  T=" << std::pow(10.0, host.logT_min + i * host.dlogT)
                << " K\n";
        }
        // жёсткий fail при большом числе выбросов, warning при 1-2:
        if (n_bad > 3) ok = false;
    }


    return ok;

}


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

    // Y(T_ref) должен быть 0 с точностью интерполяции
    auto Y_at = [&](double T) {
        double lT = std::log10(T);
        double t = (lT - host.logT_min) / host.dlogT;
        int i = std::min(std::max((int)t, 0), host.n - 2);
        double w = t - i;
        return host.Y_of_T[i] * (1.0 - w) + host.Y_of_T[i + 1] * w;
        };
    std::cout << "  Y(T_ref)  = " << Y_at(host.T_ref) << "  (must be 0)\n";

    auto Lambda_at = [&](double T) {
        double lT = std::log10(T);
        double t = (lT - host.logT_min) / host.dlogT;
        int i = std::min(std::max((int)t, 0), host.n - 2);
        double w = t - i;
        return std::pow(10.0, host.logLambda[i] * (1.0 - w)
            + host.logLambda[i + 1] * w);
        };
    std::cout << "  L(T_ref) from table = " << Lambda_at(host.T_ref) << "\n";
    std::cout << "  L_ref  in params    = " << host.Lambda_ref << "\n";
    std::cout << "  ratio               = " << Lambda_at(host.T_ref) / host.Lambda_ref << "\n";

    // Точка i_ref: T = T_ref
    int i_ref = std::lround((std::log10(host.T_ref) - host.logT_min) / host.dlogT);
    i_ref = std::min(std::max(i_ref, 1), host.n - 2);
    double dY_tab_Tref = (host.Y_of_T[i_ref + 1] - host.Y_of_T[i_ref - 1])
        / (2.0 * host.dlogT);
    std::cout << "  i_ref = " << i_ref
        << "  dY/dlogT(T_ref) = " << dY_tab_Tref
        << "   expected = " << -std::log(10.0) << "\n";

    // dY/dlogT должно равняться -(Lambda_ref/T_ref) * T / Lambda(T)
    double max_consistency_err = 0.0;
    for (int i = 1; i < host.n - 1; ++i) {
        double lT = host.logT_min + i * host.dlogT;
        double T = std::pow(10.0, lT);
        double dYdlogT = (host.Y_of_T[i + 1] - host.Y_of_T[i - 1]) / (2.0 * host.dlogT);
        double Lambda_i = std::pow(10.0, host.logLambda[i]);
        double expected = -(host.Lambda_ref / host.T_ref) * T / Lambda_i;
        double rel = std::fabs(dYdlogT - expected) / std::fabs(expected);
        max_consistency_err = std::max(max_consistency_err, rel);
    }
    std::cout << "  dY/dlogT vs Lambda consistency max rel.err = "
        << max_consistency_err << "\n";
    //if (max_consistency_err > 1e-2)
    //     fail("Y table and Lambda table are inconsistent!");

    double Y_at_Tref = Y_of_T_h(host.T_ref);
    std::cout << "  Y(T_ref) = " << Y_at_Tref << "\n";
    if (std::abs(Y_at_Tref) > 1e-6 * (host.Y_of_T[0] - host.Y_of_T[host.n - 1]))
        std::cerr << "  WARNING: Y(T_ref) != 0  Y table ref temperature mismatch\n";

    std::cout << "[cooling] loaded tables OK\n";
    std::cout << "  N=" << host.n
        << "  logT=[" << host.logT_min << ","
        << host.logT_max << "]  dlogT=" << host.dlogT << "\n";
    std::cout << "  Y=[" << host.Y_of_T[host.n - 1] << ","
        << host.Y_of_T[0] << "]\n";
    std::cout << "  T_ref=" << host.T_ref
        << "  Lambda_ref=" << host.Lambda_ref << "\n";
    std::cout << "  closure max rel.err = " << max_err << "\n";

    // === НОВОЕ: проверка согласованности Y и Lambda ===
    if (!test_cooling_consistency(host, /*verbose=*/true))
        fail("Y-table and Lambda-table are inconsistent");


    // Окно вокруг худшей точки
    /*std::cout << "  window around i=818:\n";
    std::cout << "      i          T[K]            L              Y          dY/dlogT\n";
    for (int i = 813; i <= 823; ++i) {
        double T = std::pow(10.0, host.logT_min + i * host.dlogT);
        double L = std::pow(10.0, host.logLambda[i]);
        double dYt = (i > 0 && i < host.n - 1)
            ? (host.Y_of_T[i + 1] - host.Y_of_T[i - 1]) / (2.0 * host.dlogT)
            : 0.0;
        std::printf("  %4d  %12.4e  %12.4e  %12.4e  %12.4e\n",
            i, T, L, host.Y_of_T[i], dYt);
    }*/
    
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