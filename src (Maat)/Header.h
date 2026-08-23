#pragma once
#include "Cell.h"
#define krit 0.3  // 0.1 // 0.9
struct Cell;
extern __device__ double minmod(double x, double y);
extern __device__ double linear(double x1, double t1, double x2, double t2, double x3, double t3, double y);
extern __device__ void linear2(double x1, double t1, double x2, double t2, double x3, double t3, double y1, double y2, double& A, double& B);
extern __device__ int sign(double& x);