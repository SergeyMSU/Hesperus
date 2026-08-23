#include "Kyb.h"
using namespace std;
int Kyb::move_number = 0;

Kyb::Kyb(double x, double y, double z)
{
	this->initialization(x, y, z, this->move_number);
	this->move_number++;
}

void Kyb::initialization(double x , double y, double z, int nn)
{
	this->x = x;
	this->y = y;
	this->z = z;
	this->number = nn;
	this->drob = false;
	this->ro = 0.0;
	this->p = 0.0;
	this->u = 0.0;
	this->v = 0.0;
	this->w = 0.0;
	this->Bx = 0.0;
	this->By = 0.0;
	this->Bz = 0.0;
	this->dx = 0.0;
	this->dy = 0.0;
	this->dz = 0.0;
	this->Q = 0.0;

	this->j_ = false;
	this->jx = 0.0;
	this->jy = 0.0;
	this->jz = 0.0;
	this->dpx = 0.0;
	this->dpy = 0.0;
	this->dpz = 0.0;
	this->dbbx = 0.0;
	this->dbby = 0.0;
	this->dbbz = 0.0;
}

