#pragma once
#ifndef CELL_H
#define CELL_H
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>

#define ae1  0.00280389          // 1 АЕ в безразмерным параметрах
#define phi_0  31.1486           // число Хи
#define rr_0  (40.0 * ae1)       // С какого расстояния начинаем считать (в безразмерном виде)
#define ddist rr_0
#define M_0  5.80029             // Число Маха на 1 АЕ
#define M_alf  1.2           // Число Альфвеновского Маха на 1 АЕ
#define ga (5.0/3.0)             // Показатель адиабаты
#define M_inf  1.9642            // Газодинамический Мах на бесконечности
#define M_infty  -M_inf          // Газодинамический Мах на бесконечности со знаком скорости
#define B_inf  10.3731           // Магнитное поле на бесконечности (можно было выразить через альфвеновских мах на бесконечности)
#define ddist2 (55.0 * ae1)      // До какого радиуса сносим в сферической ск?
#define a_2 0.135236  // 0.102578  // 0.10263
#define Kn_  0.506547 // 0.4326569808 // 0.4326569808 // 0.4326569808 // 6.0	


#define alpha  45.0

#define epsilon_ (1.0/M_alf)
#define step  70000
#define omega 0.0 //6 //1600


#define kk_ 196.0
#define betta 0.0 //4.2426
#define chi phi_0


#define ddist rr_0//0.3//25   // 0.7


#define kurant  0.9 // 0.9  Актуальный курант в файле header

		
//#define Kn_  0.2	                                            // Число Кнудсена
//#define a_2 0.102578  // 0.10263
#define n_p_LISM_ (3.0) 
#define n_H_LISM_ (1.0)
#define sigma(x) (kv(1.0 - a_2 * log(x)))               // Дифференциальное сечение перезарядки



#define ggg ga
#define kv(x) ( (x)*(x) )
#define kvv(x,y,z)  (kv(x) + kv(y) + kv(z))

//#define U8(ro, p, u, v, w, bx, by, bz)  ( (p) / (ggg - 1.0) + 0.5 * (ro) * kvv(u,v,w) + kvv(bx,by,bz))
#define U8(ro, p, u, v, w, bx, by, bz)  ( (p) / (ggg - 1.0) + 0.5 * (ro) * kvv(u,v,w) + kvv(bx,by,bz) / 25.13274122871834590768)
#define skk(u,v,w,bx,by,bz) ( (u)*(bx) + (v)*(by) + (w)*(bz) )
#define g1 (ga - 1.0)
#define gg1 (ga - 1.0)
#define g2 (ga + 1.0)
#define gg2 (ga + 1.0)
#define gp ((g2/ga)/2.0)
#define gm ((g1/ga)/2.0)
#define gga ga
#define Omega 0.0


#define eps 10e-10
#define eps8 10e-8
#define pi 3.14159265358979323846
#define PI 3.14159265358979323846
#define cpi4 12.56637061435917295384
#define cpi8 25.13274122871834590768
#define spi4 ( 3.544907701811032 )
#define epsb 1e-6
#define eps_p 1e-6
#define eps_d 1e-3


struct Cell // Класс ячейка
{
	double ro[2];
	double p[2];
	double u[2];
	double v[2];
	double w[2];
	double Bx[2];
	double By[2];
	double Bz[2];
	double dx;               // Половина ширины ячейки
	double dy;               // Половина глубины ячейки
	double dz;               // Половина высоты ячейки
	double x;                // Координата центра ячейки
	double y;                // Координата центра ячейки
	double z;                // Координата центра ячейки
	int l;
	int r;
};



#endif // APPROXIMATION_H