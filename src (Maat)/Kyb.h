#pragma once
#include <vector>
using namespace std;
class Kyb;

class Kyb
{
public:
	double dx;               // Половина ширины ячейки
	double dy;               // Половина глубины ячейки
	double dz;               // Половина высоты ячейки
	double x;                // Координата центра ячейки
	double y;                // Координата центра ячейки
	double z;                // Координата центра ячейки
	double ro;
	double p;
	double u;
	double v;
	double w;
	double Bx;
	double By;
	double Bz;
	double Q;
	double jx;
	double jy;
	double jz;
	double dpx;
	double dpy;
	double dpz;
	double dbbx;
	double dbby;
	double dbbz;
	bool j_;
	vector <Kyb*> sosed;
	int number;
	static int move_number;
	bool drob;               // Удобная булевская переменная для использования в разных функциях 


	Kyb(double x, double y, double z);      /// Конструктор класса

private:
	void initialization(double, double, double, int);
};

