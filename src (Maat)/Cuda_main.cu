#include "Header.h"
#include "math.h"

__device__ double HLLD_Alexashov(const double& ro_L, const double& p_L, const double& v1_L, const double& v2_L, const double& v3_L,//
    const double& Bx_L, const double& By_L, const double& Bz_L, const double& ro_R, const double& p_R, const double& v1_R, const double& v2_R, const double& v3_R,//
    const double& Bx_R, const double& By_R, const double& Bz_R, double* P, const double& n1, const double& n2, const double& n3, double& rad, int metod = 0);

__device__ double minmod(double x, double y)
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

__device__ double linear(double x1, double t1, double x2, double t2, double x3, double t3, double y)
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

__device__ void f_TVD(const double& dx, const double& p1, const double& p2, const double& p3, const double& p4, 
    double& p12, double& p21, const double& s1, const double& s2, const double& s3)
{
    //double s1 = __dsqrt_rn(kv(x1 - x3) + kv(y1 - y3) + kv(z1 - z3));
    //double s2 = __dsqrt_rn(kv(x1 - x2) + kv(y1 - y2) + kv(z1 - z2));
    //double s3 = __dsqrt_rn(kv(x4 - x2) + kv(y4 - y2) + kv(z4 - z2));

    p12 = linear(-s1, p3, 0.0, p1, s2, p2, dx);
    p21 = linear(0.0, p1, s2, p2, s2 + s3, p4, dx);
}


//  void chlld(id_bn, n_state, n_disco, KOBL, i_in, j_in, k_in, kdir, al, be, ge, el, w, qqq1, qqq2, dsl, dsp, dsc, ythll, qqq)
__device__ double HLLD_Alexashov(const double& ro_L, const double& p_L, const double& v1_L, const double& v2_L, const double& v3_L,//
    const double& Bx_L, const double& By_L, const double& Bz_L, const double& ro_R, const double& p_R, const double& v1_R, const double& v2_R, const double& v3_R,//
    const double& Bx_R, const double& By_R, const double& Bz_R, double* P, const double& n1, const double& n2, const double& n3, double& rad, int metod)
{
    int x0 = 0, x1 = 1, x2 = 2;
    double aco[3][3];
    int n_state = metod;
    //c-------  n_state=0   - one speed LAX
    //c-------  n_state=1   - two speed LAX (HLL,(Harten-Lax-van-Leer))
    //c-------  n_state=2   - two-state (3 speed) HLLC (Contact Discontinuity)
    //c-------  n_state=3   - multi-state (5 speed) HLLD (All Discontinuity)

    double FR[8], FL[8], dq[8];
    double FW[8], UL[8], UZ[8], UR[8];
     double UZL[8], UZR[8];
     double UZZL[8], UZZR[8];
    double vL[3], vR[3], bL[3], bR[3];
     double vzL[3], vzR[3], bzL[3], bzR[3];
     double vzzL[3], vzzR[3], bzzL[3], bzzR[3];
    double qv[3], qb[3];



    double wv = 0.0;
    int n_disco = 0; // Для определения скоростей характеристик


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

    double ro = (r2 + r1) / x2;
    double au = (u2 + u1) / x2;
    double av = (v2 + v1) / x2;
    double aw = (w2 + w1) / x2;
    double ap = (p2 + p1) / x2;
    double abx = (bx2 + bx1) / x2;
    double aby = (by2 + by1) / x2;
    double abz = (bz2 + bz1) / x2;

    double al = n1;
    double be = n2;
    double ge = n3;

    double bk = abx * al + aby * be + abz * ge;
    double b2 = kv(abx) + kv(aby) + kv(abz);

    double  d = b2 - kv(bk);
    aco[0][0] = al;
    aco[1][0] = be;
    aco[2][0] = ge;

    if (d > 0.00001)
    {
        d = __dsqrt_rn(d);
        aco[0][1] = (abx - bk * al) / d;
        aco[1][1] = (aby - bk * be) / d;
        aco[2][1] = (abz - bk * ge) / d;
        aco[0][2] = (aby * ge - abz * be) / d;
        aco[1][2] = (abz * al - abx * ge) / d;
        aco[2][2] = (abx * be - aby * al) / d;
    }
    else
    {
        double aix, aiy, aiz;
        if ( (fabs(al) < fabs(be)) && (fabs(al) < fabs(ge)) )
        {
            aix = x1;
            aiy = x0;
            aiz = x0;
        }
        else if ( fabs(be) < fabs(ge) )
        {
            aix = x0;
            aiy = x1;
            aiz = x0;
        }
        else
        {
            aix = x0;
            aiy = x0;
            aiz = x1;
        }
        double aik = aix * al + aiy * be + aiz * ge;
        d = __dsqrt_rn(x1 - kv(aik));
        aco[0][1] = (aix - aik * al) / d;
        aco[1][1] = (aiy - aik * be) / d;
        aco[2][1] = (aiz - aik * ge) / d;
        aco[0][2] = (aiy * ge - aiz * be) / d;
        aco[1][2] = (aiz * al - aix * ge) / d;
        aco[2][2] = (aix * be - aiy * al) / d;
    }

    aco[0][0] = al;
    aco[1][0] = be;
    aco[2][0] = ge;

    //if (fabs(skk(aco[0][0], aco[1][0], aco[2][0], aco[0][1], aco[1][1], aco[2][1])) > 0.000001 || //
    //    fabs(skk(aco[0][0], aco[1][0], aco[2][0], aco[0][2], aco[1][2], aco[2][2])) > 0.000001 || //
    //    fabs(skk(aco[0][2], aco[1][2], aco[2][2], aco[0][1], aco[1][1], aco[2][1])) > 0.000001 || //
    //    fabs(kvv(aco[0][0], aco[1][0], aco[2][0]) - 1.0) > 0.000001 || fabs(kvv(aco[0][1], aco[1][1], aco[2][1]) - 1.0) > 0.000001 ||//
    //    fabs(kvv(aco[0][2], aco[1][2], aco[2][2]) - 1.0) > 0.000001)
    //{
    //    printf("Ne normal  174fdcdsaxes\n");
    //}


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
    double cfL = (qp + qm) / x2;
    double ptL = p1 + b2L / x2;

    double aaR = bR[0] / __dsqrt_rn(r2);
    double b2R = kv(bR[0]) + kv(bR[1]) + kv(bR[2]);
    double b22 = b2R / r2;
    double cR = __dsqrt_rn(ga * p2 / r2);
    qp = __dsqrt_rn(b22 + cR * (cR + 2.0 * aaR));
    qm = __dsqrt_rn(b22 + cR * (cR - 2.0 * aaR));
    double cfR = (qp + qm) / x2;
    double ptR = p2 + b2R / x2;

    double aC = (aaL + aaR) / x2;
    double b2o = (b22 + b21) / x2;
    double cC = __dsqrt_rn(ga * ap / ro);
    qp = __dsqrt_rn(b2o + cC * (cC + x2 * aC));
    qm = __dsqrt_rn(b2o + cC * (cC - x2 * aC));
    double cfC = (qp + qm) / x2;
    double vC1 = (vL[0] + vR[0]) / x2;

    double SL, SR;

    if(true)
    {
        SL = min(vL[0], vR[0]) - max(cfL, cfR);
        SR = max(vL[0], vR[0]) + max(cfL, cfR);
    }
    else if (n_disco == 1)
    {
        SL = min((vL[0] - cfL), (vC1 - cfC));
        SR = max((vR[0] + cfR), (vC1 + cfC));
    }
    else if (n_disco == 0)
    {
        SL = min((vL[0] - cfL), (vR[0] - cfR));
        SR = max((vL[0] + cfL), (vR[0] + cfR));
    }
    else if (n_disco == 2)
    {
        double SL_1 = min((vL[0] - cfL), (vC1 - cfC));
        double SR_1 = max((vR[0] + cfR), (vC1 + cfC));
        double SL_2 = min((vL[0] - cfL), (vR[0] - cfR));
        double SR_2 = max((vL[0] + cfL), (vR[0] + cfR));
        double oo = 0.75;
        double oo1 = 1.0 - oo;
        SL = oo * SL_1 + oo1 * SL_2;
        SR = oo * SR_1 + oo1 * SR_2;
    }


    double suR = SR - vR[0];
    double suL = SL - vL[0];
    double SM = (suR * r2 * vR[0] - ptR + ptL - suL * r1 * vL[0]) / (suR * r2 - suL * r1);

    // double dsl = SL;
    // double dsc = SM;
    // double dsp = SR;

    if ( (SR < SL)||(SL > SM)||(SR < SM) )
    {
        printf("ERROR -  254 fghrvtrgr\n");
        printf("%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",//
            vL[0], vR[0], cfL, cfR, ro_L, ro_R, p_L, p_R, suR, suL);
    }

    double UU = max(fabs(SL), fabs(SR));
    double time = krit * rad / UU;

    double TR0, TL0;
    if (n_state == 0)
    {
        TR0 = fabs(vL[0] + vR[0]) / x2 + cfC;
        TL0 = -TR0;
        SR = TR0;
        SL = TL0;
    }


    double upt1 = (kv(u1) + kv(v1) + kv(w1)) / 2.0;
    double sbv1 = u1 * bx1 + v1 * by1 + w1 * bz1;

    double upt2 = (kv(u2) + kv(v2) + kv(w2)) / 2.0;
    double sbv2 = u2 * bx2 + v2 * by2 + w2 * bz2;

    double e1 = p1 / g1 + r1 * upt1 + b2L / x2;
    double e2 = p2 / g1 + r2 * upt2 + b2R / x2;

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

    for (int i = 0; i < 3; i++)
    {

        UL[i + 1] = r1 * vL[i];
        UL[i + 5] = bL[i];
        UR[i + 1] = r2 * vR[i];
        UR[i + 5] = bR[i];
    }

    for (int ik = 0; ik < 8; ik++)
    {
        UZ[ik] = (SR * UR[ik] - SL * UL[ik] + FL[ik] - FR[ik]) / (SR - SL);
    }


    if (n_state <= 1)
    {

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
        else if ( (SL <= wv) && (wv <= SR) )
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
        else
        {
            printf("ERROR  329 87732, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", r1, r2, p1, p2, al, be, ge);
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

        double SN = max(fabs(SL), fabs(SR));

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
            wbn = wv * (bL[0] + bR[0]) / x2;
        }

        qb[0] = -SN * (bR[0] - bL[0]) - wbn;

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
    if (n_state == 3)
    {
         
        double ptz = (suR * r2 * ptL - suL * r1 * ptR + r1 * r2 * suR * suL * (vR[0] - vL[0])) / (suR * r2 - suL * r1);

        vzL[0] = SM;
        vzR[0] = SM;
        vzzL[0] = SM;
        vzzR[0] = SM;
        double ptzL = ptz;
        double ptzR = ptz;
        double ptzzL = ptz;
        double ptzzR = ptz;

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
        double tvR, tbR, tvL, tbL;
        if (fabs(ttR) <= 0.00000001)
        {
            tvR = x0;
            tbR = x0;
        }
        else
        {
            tvR = (SM - vR[0]) / ttR;
            tbR = (r2 * suR * suR - bn2) / ttR;
        }

        double ttL = r1 * suL * (SL - SM) - bn2;
        if (fabs(ttL) <= 0.00000001)
        {
            tvL = x0;
            tbL = x0;
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
        if (fabs(bn) > 0.000001)
        {
            sbn = 1.0 * sign(bn);
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
        }

        if ( (SL <= wv) && (SZL >= wv) )
        {
            int ik = 0;
            P[ik] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
            ik = 4;
            P[ik] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
            for (int ik = 1; ik < 4; ik++)
            {
                qv[ik - 1] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
            }

            for (int ik = 5; ik < 8; ik++)
            {
                qb[ik - 5] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
            }
        }
        //c------ FZZ
        if (ibn == 1)
        {

            if ( (SZL <= wv) && (SM >= wv) )
            {
                int ik = 0;
                P[ik] = FL[ik] + SZL * (UZZL[ik] - UZL[ik]) + SL * (UZL[ik] - UL[ik]) - wv * UZZL[ik];
                ik = 4;
                P[ik] = FL[ik] + SZL * (UZZL[ik] - UZL[ik]) + SL * (UZL[ik] - UL[ik]) - wv * UZZL[ik];
                for (int ik = 1; ik < 4; ik++)
                {
                    qv[ik - 1] = FL[ik] + SZL * (UZZL[ik] - UZL[ik]) + SL * (UZL[ik] - UL[ik]) - wv * UZZL[ik];
                }

                for (int ik = 5; ik < 8; ik++)
                {
                    qb[ik - 5] = FL[ik] + SZL * (UZZL[ik] - UZL[ik]) + SL * (UZL[ik] - UL[ik]) - wv * UZZL[ik];
                }
            }

            if ( (SM <= wv) && (SZR >= wv) )
            {
                int ik = 0;
                P[ik] = FR[ik] + SZR * (UZZR[ik] - UZR[ik]) + SR * (UZR[ik] - UR[ik]) - wv * UZZR[ik];
                ik = 4;
                P[ik] = FR[ik] + SZR * (UZZR[ik] - UZR[ik]) + SR * (UZR[ik] - UR[ik]) - wv * UZZR[ik];
                for (int ik = 1; ik < 4; ik++)
                {
                    qv[ik - 1] = FR[ik] + SZR * (UZZR[ik] - UZR[ik]) + SR * (UZR[ik] - UR[ik]) - wv * UZZR[ik];
                }

                for (int ik = 5; ik < 8; ik++)
                {
                    qb[ik - 5] = FR[ik] + SZR * (UZZR[ik] - UZR[ik]) + SR * (UZR[ik] - UR[ik]) - wv * UZZR[ik];
                }
            }

        }
        //c------ 
        if ( (SZR <= wv) && (SR >= wv) )
        {
            int ik = 0;
            P[ik] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];
            ik = 4;
            P[ik] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];

            for (int ik = 1; ik < 4; ik++)
            {
                qv[ik - 1] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];
            }

            for (int ik = 5; ik < 8; ik++)
            {
                qb[ik - 5] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];
            }
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
        }


        //c----- Bn
        //double SN = max(fabs(SL), fabs(SR));

        double SN = max(fabs(SL), fabs(SR));

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
            wbn = wv * (bL[0] + bR[0]) / x2;
        }

        qb[0] = -SN * (bR[0] - bL[0]) - wbn;

        //c-----


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
}


__device__ double HLLDQ_Alexashov(const double& ro_L, const double& Q_L, const double& p_L, const double& v1_L, const double& v2_L, const double& v3_L,//
    const double& Bx_L, const double& By_L, const double& Bz_L, const double& ro_R, const double& Q_R, const double& p_R, const double& v1_R, const double& v2_R, const double& v3_R,//
    const double& Bx_R, const double& By_R, const double& Bz_R, double* P, double& PQ, const double& n1, const double& n2, const double& n3, double& rad, int metod)
{   // Не работает, если скорость грани не нулевая
    int x0 = 0, x1 = 1, x2 = 2;
    double aco[3][3];
    int n_state = metod;
    //c-------  n_state=0   - one speed LAX
    //c-------  n_state=1   - two speed LAX (HLL,(Harten-Lax-van-Leer))
    //c-------  n_state=2   - two-state (3 speed) HLLC (Contact Discontinuity)
    //c-------  n_state=3   - multi-state (5 speed) HLLD (All Discontinuity)

    double FR[8], FL[8], dq[8];
    double FW[8], UL[8], UZ[8], UR[8];
    double UZL[8], UZR[8];
    double UZZL[8], UZZR[8];
    double vL[3], vR[3], bL[3], bR[3];
    double vzL[3], vzR[3], bzL[3], bzR[3];
    double vzzL[3], vzzR[3], bzzL[3], bzzR[3];
    double qv[3], qb[3];



    double wv = 0.0;
    int n_disco = 0; // Для определения скоростей характеристик

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

    double ro = (r2 + r1) / x2;
    double au = (u2 + u1) / x2;
    double av = (v2 + v1) / x2;
    double aw = (w2 + w1) / x2;
    double ap = (p2 + p1) / x2;
    double abx = (bx2 + bx1) / x2;
    double aby = (by2 + by1) / x2;
    double abz = (bz2 + bz1) / x2;

    double al = n1;
    double be = n2;
    double ge = n3;

    double bk = abx * al + aby * be + abz * ge;
    double b2 = kv(abx) + kv(aby) + kv(abz);

    double  d = b2 - kv(bk);
    aco[0][0] = al;
    aco[1][0] = be;
    aco[2][0] = ge;

    if (d > 0.00001)
    {
        d = __dsqrt_rn(d);
        aco[0][1] = (abx - bk * al) / d;
        aco[1][1] = (aby - bk * be) / d;
        aco[2][1] = (abz - bk * ge) / d;
        aco[0][2] = (aby * ge - abz * be) / d;
        aco[1][2] = (abz * al - abx * ge) / d;
        aco[2][2] = (abx * be - aby * al) / d;
    }
    else
    {
        double aix, aiy, aiz;
        if ((fabs(al) < fabs(be)) && (fabs(al) < fabs(ge)))
        {
            aix = x1;
            aiy = x0;
            aiz = x0;
        }
        else if (fabs(be) < fabs(ge))
        {
            aix = x0;
            aiy = x1;
            aiz = x0;
        }
        else
        {
            aix = x0;
            aiy = x0;
            aiz = x1;
        }
        double aik = aix * al + aiy * be + aiz * ge;
        d = __dsqrt_rn(x1 - kv(aik));
        aco[0][1] = (aix - aik * al) / d;
        aco[1][1] = (aiy - aik * be) / d;
        aco[2][1] = (aiz - aik * ge) / d;
        aco[0][2] = (aiy * ge - aiz * be) / d;
        aco[1][2] = (aiz * al - aix * ge) / d;
        aco[2][2] = (aix * be - aiy * al) / d;
    }

    aco[0][0] = al;
    aco[1][0] = be;
    aco[2][0] = ge;

    //if (fabs(skk(aco[0][0], aco[1][0], aco[2][0], aco[0][1], aco[1][1], aco[2][1])) > 0.000001 || //
    //    fabs(skk(aco[0][0], aco[1][0], aco[2][0], aco[0][2], aco[1][2], aco[2][2])) > 0.000001 || //
    //    fabs(skk(aco[0][2], aco[1][2], aco[2][2], aco[0][1], aco[1][1], aco[2][1])) > 0.000001 || //
    //    fabs(kvv(aco[0][0], aco[1][0], aco[2][0]) - 1.0) > 0.000001 || fabs(kvv(aco[0][1], aco[1][1], aco[2][1]) - 1.0) > 0.000001 ||//
    //    fabs(kvv(aco[0][2], aco[1][2], aco[2][2]) - 1.0) > 0.000001)
    //{
    //    printf("Ne normal  174fdcdsaxes\n");
    //}


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
    double cfL = (qp + qm) / x2;
    double ptL = p1 + b2L / x2;

    double aaR = bR[0] / __dsqrt_rn(r2);
    double b2R = kv(bR[0]) + kv(bR[1]) + kv(bR[2]);
    double b22 = b2R / r2;
    double cR = __dsqrt_rn(ga * p2 / r2);
    qp = __dsqrt_rn(b22 + cR * (cR + 2.0 * aaR));
    qm = __dsqrt_rn(b22 + cR * (cR - 2.0 * aaR));
    double cfR = (qp + qm) / x2;
    double ptR = p2 + b2R / x2;

    double aC = (aaL + aaR) / x2;
    double b2o = (b22 + b21) / x2;
    double cC = __dsqrt_rn(ga * ap / ro);
    qp = __dsqrt_rn(b2o + cC * (cC + x2 * aC));
    qm = __dsqrt_rn(b2o + cC * (cC - x2 * aC));
    double cfC = (qp + qm) / x2;
    double vC1 = (vL[0] + vR[0]) / x2;

    double SL, SR;

    if (true)
    {
        SL = min(vL[0], vR[0]) - max(cfL, cfR);
        SR = max(vL[0], vR[0]) + max(cfL, cfR);
    }
    else if (n_disco == 1)
    {
        SL = min((vL[0] - cfL), (vC1 - cfC));
        SR = max((vR[0] + cfR), (vC1 + cfC));
    }
    else if (n_disco == 0)
    {
        SL = min((vL[0] - cfL), (vR[0] - cfR));
        SR = max((vL[0] + cfL), (vR[0] + cfR));
    }
    else if (n_disco == 2)
    {
        double SL_1 = min((vL[0] - cfL), (vC1 - cfC));
        double SR_1 = max((vR[0] + cfR), (vC1 + cfC));
        double SL_2 = min((vL[0] - cfL), (vR[0] - cfR));
        double SR_2 = max((vL[0] + cfL), (vR[0] + cfR));
        double oo = 0.75;
        double oo1 = 1.0 - oo;
        SL = oo * SL_1 + oo1 * SL_2;
        SR = oo * SR_1 + oo1 * SR_2;
    }


    double suR = SR - vR[0];
    double suL = SL - vL[0];
    double SM = (suR * r2 * vR[0] - ptR + ptL - suL * r1 * vL[0]) / (suR * r2 - suL * r1);

    // double dsl = SL;
    // double dsc = SM;
    // double dsp = SR;

    if ((SR < SL) || (SL > SM) || (SR < SM))
    {
        printf("ERROR -  254 fghrvtrgr\n");
        printf("%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",//
            vL[0], vR[0], cfL, cfR, ro_L, ro_R, p_L, p_R, suR, suL);
    }

    double UU = max(fabs(SL), fabs(SR));
    double time = krit * rad / UU;

    double TR0, TL0;
    if (n_state == 0)
    {
        TR0 = fabs(vL[0] + vR[0]) / x2 + cfC;
        TL0 = -TR0;
        SR = TR0;
        SL = TL0;
    }


    double upt1 = (kv(u1) + kv(v1) + kv(w1)) / 2.0;
    double sbv1 = u1 * bx1 + v1 * by1 + w1 * bz1;

    double upt2 = (kv(u2) + kv(v2) + kv(w2)) / 2.0;
    double sbv2 = u2 * bx2 + v2 * by2 + w2 * bz2;

    double e1 = p1 / g1 + r1 * upt1 + b2L / x2;
    double e2 = p2 / g1 + r2 * upt2 + b2R / x2;
    double FQ_L, FQ_R;


    FQ_L = Q_L * vL[0];
    FL[0] = r1 * vL[0];
    FL[1] = r1 * vL[0] * vL[0] + ptL - kv(bL[0]);
    FL[2] = r1 * vL[0] * vL[1] - bL[0] * bL[1];
    FL[3] = r1 * vL[0] * vL[2] - bL[0] * bL[2];
    FL[4] = (e1 + ptL) * vL[0] - bL[0] * sbv1;
    FL[5] = 0.0;
    FL[6] = vL[0] * bL[1] - vL[1] * bL[0];
    FL[7] = vL[0] * bL[2] - vL[2] * bL[0];

    FQ_R = Q_R * vR[0];
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

    for (int i = 0; i < 3; i++)
    {

        UL[i + 1] = r1 * vL[i];
        UL[i + 5] = bL[i];
        UR[i + 1] = r2 * vR[i];
        UR[i + 5] = bR[i];
    }

    for (int ik = 0; ik < 8; ik++)
    {
        UZ[ik] = (SR * UR[ik] - SL * UL[ik] + FL[ik] - FR[ik]) / (SR - SL);
    }
    double UZQ = (SR * Q_R - SL * Q_L + FQ_L - FQ_R) / (SR - SL);

    if (n_state <= 1)
    {

        for (int ik = 0; ik < 8; ik++)
        {
            dq[ik] = UR[ik] - UL[ik];
        }
        double dqQ = Q_R - Q_L;



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
        else if ((SL <= wv) && (wv <= SR))
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
        else
        {
            printf("ERROR  329 87732, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n", r1, r2, p1, p2, al, be, ge);
        }


        double a = TR * TL;
        double b = TR - TL;

        PQ = (TR * FQ_L - TL * FQ_R + a * dqQ) / b;
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

        double SN = max(fabs(SL), fabs(SR));

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
            wbn = wv * (bL[0] + bR[0]) / x2;
        }

        qb[0] = -SN * (bR[0] - bL[0]) - wbn;

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
    if (n_state == 3)
    {

        double ptz = (suR * r2 * ptL - suL * r1 * ptR + r1 * r2 * suR * suL * (vR[0] - vL[0])) / (suR * r2 - suL * r1);

        vzL[0] = SM;
        vzR[0] = SM;
        vzzL[0] = SM;
        vzzR[0] = SM;
        double ptzL = ptz;
        double ptzR = ptz;
        double ptzzL = ptz;
        double ptzzR = ptz;

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
        double tvR, tbR, tvL, tbL;
        if (fabs(ttR) <= 0.000001)
        {
            tvR = x0;
            tbR = x0;
        }
        else
        {
            tvR = (SM - vR[0]) / ttR;
            tbR = (r2 * suR * suR - bn2) / ttR;
        }

        double ttL = r1 * suL * (SL - SM) - bn2;
        if (fabs(ttL) <= 0.000001)
        {
            tvL = x0;
            tbL = x0;
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
        if (fabs(bn) > 0.000001)
        {
            sbn = 1.0 * sign(bn);
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



        if (SL > wv)
        {
            P[0] = FL[0] - wv * UL[0];
            PQ = P[0] * Q_L / r1;
            P[4] = FL[4] - wv * UL[4];
            for (int ik = 1; ik < 4; ik++)
            {
                qv[ik - 1] = FL[ik] - wv * UL[ik];
            }

            for (int ik = 5; ik < 8; ik++)
            {
                qb[ik - 5] = FL[ik] - wv * UL[ik];
            }
        }

        if ((SL <= wv) && (SZL >= wv))
        {
            int ik = 0;
            P[ik] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
            PQ = P[ik] * Q_L / r1;
            ik = 4;
            P[ik] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
            for (int ik = 1; ik < 4; ik++)
            {
                qv[ik - 1] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
            }

            for (int ik = 5; ik < 8; ik++)
            {
                qb[ik - 5] = FL[ik] + SL * (UZL[ik] - UL[ik]) - wv * UZL[ik];
            }
        }
        //c------ FZZ
        if (ibn == 1)
        {

            if ((SZL <= wv) && (SM >= wv))
            {
                int ik = 0;
                P[ik] = FL[ik] + SZL * (UZZL[ik] - UZL[ik]) + SL * (UZL[ik] - UL[ik]) - wv * UZZL[ik];
                PQ = P[ik] * Q_L / r1;
                ik = 4;
                P[ik] = FL[ik] + SZL * (UZZL[ik] - UZL[ik]) + SL * (UZL[ik] - UL[ik]) - wv * UZZL[ik];
                for (int ik = 1; ik < 4; ik++)
                {
                    qv[ik - 1] = FL[ik] + SZL * (UZZL[ik] - UZL[ik]) + SL * (UZL[ik] - UL[ik]) - wv * UZZL[ik];
                }

                for (int ik = 5; ik < 8; ik++)
                {
                    qb[ik - 5] = FL[ik] + SZL * (UZZL[ik] - UZL[ik]) + SL * (UZL[ik] - UL[ik]) - wv * UZZL[ik];
                }
            }

            if ((SM <= wv) && (SZR >= wv))
            {
                int ik = 0;
                P[ik] = FR[ik] + SZR * (UZZR[ik] - UZR[ik]) + SR * (UZR[ik] - UR[ik]) - wv * UZZR[ik];
                PQ = P[ik] * Q_R / r2;
                ik = 4;
                P[ik] = FR[ik] + SZR * (UZZR[ik] - UZR[ik]) + SR * (UZR[ik] - UR[ik]) - wv * UZZR[ik];
                for (int ik = 1; ik < 4; ik++)
                {
                    qv[ik - 1] = FR[ik] + SZR * (UZZR[ik] - UZR[ik]) + SR * (UZR[ik] - UR[ik]) - wv * UZZR[ik];
                }

                for (int ik = 5; ik < 8; ik++)
                {
                    qb[ik - 5] = FR[ik] + SZR * (UZZR[ik] - UZR[ik]) + SR * (UZR[ik] - UR[ik]) - wv * UZZR[ik];
                }
            }

        }
        //c------ 
        if ((SZR <= wv) && (SR >= wv))
        {
            int ik = 0;
            P[ik] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];
            PQ = P[ik] * Q_R / r2;
            ik = 4;
            P[ik] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];

            for (int ik = 1; ik < 4; ik++)
            {
                qv[ik - 1] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];
            }

            for (int ik = 5; ik < 8; ik++)
            {
                qb[ik - 5] = FR[ik] + SR * (UZR[ik] - UR[ik]) - wv * UZR[ik];
            }
        }

        if (SR < wv)
        {
            P[0] = FR[0] - wv * UR[0];
            PQ = P[0] * Q_R / r2;
            P[4] = FR[4] - wv * UR[4];
            for (int ik = 1; ik < 4; ik++)
            {
                qv[ik - 1] = FR[ik] - wv * UR[ik];
            }


            for (int ik = 5; ik < 8; ik++)
            {
                qb[ik - 5] = FR[ik] - wv * UR[ik];
            }
        }


        //c----- Bn
        //double SN = max(fabs(SL), fabs(SR));

        double SN = max(fabs(SL), fabs(SR));

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
            wbn = wv * (bL[0] + bR[0]) / x2;
        }

        qb[0] = -SN * (bR[0] - bL[0]) - wbn;

        //c-----


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
}


__device__ double HLLDQ_Korolkov(const double& ro_L, const double& Q_L, const double& p_L, const double& v1_L, const double& v2_L, const double& v3_L,//
    const double& Bx_L, const double& By_L, const double& Bz_L, const double& ro_R, const double& Q_R, const double& p_R, const double& v1_R, const double& v2_R, const double& v3_R,//
    const double& Bx_R, const double& By_R, const double& Bz_R, double* P, double& PQ, const double& n1, const double& n2, const double& n3, double& rad, int metod)
{// Не работает, если скорость грани не нулевая
 // Нормаль здесь единичная по осям координат ! (иначе нужно немного переделывать)

        double bx_L = Bx_L / spi4;
        double by_L = By_L / spi4;
        double bz_L = Bz_L / spi4;

        double bx_R = Bx_R / spi4;
        double by_R = By_R / spi4;
        double bz_R = Bz_R / spi4;

        double t1 = 0.0;
        double t2 = 0.0;
        double t3 = 0.0;

        double m1 = 0.0;
        double m2 = 0.0;
        double m3 = 0.0;

        if (n1 > 0.1)
        {
            t2 = 1.0;
            m3 = 1.0;
        }
        else if (n2 > 0.1)
        {
            t3 = 1.0;
            m1 = 1.0;
        }
        else if (n3 > 0.1)
        {
            t1 = 1.0;
            m2 = 1.0;
        }
        else if (n1 < -0.1)
        {
            t3 = -1.0;
            m2 = -1.0;
        }
        else if (n2 < -0.1)
        {
            t1 = -1.0;
            m3 = -1.0;
        }
        else if (n3 < -0.1)
        {
            t1 = -1.0;
            m2 = -1.0;
        }
        else
        {
            printf("EROROR 1421  normal_error\n");
        }


        double u1, v1, w1, u2, v2, w2;
        u1 = v1_L * n1 + v2_L * n2 + v3_L * n3;
        v1 = v1_L * t1 + v2_L * t2 + v3_L * t3;
        w1 = v1_L * m1 + v2_L * m2 + v3_L * m3;
        u2 = v1_R * n1 + v2_R * n2 + v3_R * n3;
        v2 = v1_R * t1 + v2_R * t2 + v3_R * t3;
        w2 = v1_R * m1 + v2_R * m2 + v3_R * m3;

        double bn1, bt1, bm1, bn2, bt2, bm2;
        bn1 = bx_L * n1 + by_L * n2 + bz_L * n3;
        bt1 = bx_L * t1 + by_L * t2 + bz_L * t3;
        bm1 = bx_L * m1 + by_L * m2 + bz_L * m3;
        bn2 = bx_R * n1 + by_R * n2 + bz_R * n3;
        bt2 = bx_R * t1 + by_R * t2 + bz_R * t3;
        bm2 = bx_R * m1 + by_R * m2 + bz_R * m3;

        //cout << " = " << bt2 * bt2 + bm2 * bm2 << endl;

        double sqrtroL = sqrt(ro_L);
        double sqrtroR = sqrt(ro_R);
        double ca_L = bn1 / sqrtroL;
        double ca_R = bn2 / sqrtroR;
        double cL = sqrt(ggg * p_L / ro_L);
        double cR = sqrt(ggg * p_R / ro_R);

        double bb_L = kv(bx_L) + kv(by_L) + kv(bz_L);
        double bb_R = kv(bx_R) + kv(by_R) + kv(bz_R);

        double aL = (kv(bx_L) + kv(by_L) + kv(bz_L)) / ro_L;
        double aR = (kv(bx_L) + kv(by_L) + kv(bz_L)) / ro_L;

        double uu_L = (kv(v1_L) + kv(v2_L) + kv(v3_L)) / 2.0;
        double uu_R = (kv(v1_R) + kv(v2_R) + kv(v3_R)) / 2.0;

        double cfL = sqrt((ggg * p_L + bb_L + //
            sqrt(kv(ggg * p_L + bb_L) - 4.0 * ggg * p_L * kv(bn1))) / (2.0 * ro_L));
        double cfR = sqrt((ggg * p_R + bb_R + //
            sqrt(kv(ggg * p_R + bb_R) - 4.0 * ggg * p_R * kv(bn2))) / (2.0 * ro_R));


        double SL = min(u1, u2) - max(cfL, cfR);
        double SR = max(u1, u2) + max(cfL, cfR);

        double pTL = p_L + bb_L / 2.0;
        double pTR = p_R + bb_R / 2.0;

        double suR = (SR - u2);
        double suL = (SL - u1);

        double SM = (suR * ro_R * u2 - suL * ro_L * u1 - pTR + pTL) //
            / (suR * ro_R - suL * ro_L);

        double PTT = (suR * ro_R * pTL - suL * ro_L * pTR + ro_L * ro_R * suR * suL * (u2 - u1))//
            / (suR * ro_R - suL * ro_L);

        double UU = max(fabs(SL), fabs(SR));
        double time = krit * rad / UU;

        double FL[9], FR[9], UL[9], UR[9];

        double e1 = p_L / g1 + ro_L * uu_L + bb_L / 2.0;
        double e2 = p_R / g1 + ro_R * uu_R + bb_R / 2.0;


        FL[0] = ro_L * u1;
        FL[1] = ro_L * u1 * u1 + pTL - kv(bn1);
        FL[2] = ro_L * u1 * v1 - bn1 * bt1;
        FL[3] = ro_L * u1 * w1 - bn1 * bm1;
        FL[4] = (e1 + pTL) * u1 - bn1 * (u1 * bn1 + v1 * bt1 + w1 * bm1);
        //cout << uu_L << endl;
        FL[5] = 0.0;
        FL[6] = u1 * bt1 - v1 * bn1;
        FL[7] = u1 * bm1 - w1 * bn1;
        FL[8] = Q_L * u1;

        FR[0] = ro_R * u2;
        FR[1] = ro_R * u2 * u2 + pTR - kv(bn2);
        FR[2] = ro_R * u2 * v2 - bn2 * bt2;
        FR[3] = ro_R * u2 * w2 - bn2 * bm2;
        FR[4] = (e2 + pTR) * u2 - bn2 * (u2 * bn2 + v2 * bt2 + w2 * bm2);
        FR[5] = 0.0;
        FR[6] = u2 * bt2 - v2 * bn2;
        FR[7] = u2 * bm2 - w2 * bn2;
        FR[8] = Q_R * u2;

        UL[0] = ro_L;
        UL[1] = ro_L * u1;
        UL[2] = ro_L * v1;
        UL[3] = ro_L * w1;
        UL[4] = e1;
        UL[5] = bn1;
        UL[6] = bt1;
        UL[7] = bm1;
        UL[8] = Q_L;

        UR[0] = ro_R;
        UR[1] = ro_R * u2;
        UR[2] = ro_R * v2;
        UR[3] = ro_R * w2;
        UR[4] = e2;
        UR[5] = bn2;
        UR[6] = bt2;
        UR[7] = bm2;
        UR[8] = Q_R;

        double bn = (SR * UR[5] - SL * UL[5] + FL[5] - FR[5]) / (SR - SL);
        double bt = (SR * UR[6] - SL * UL[6] + FL[6] - FR[6]) / (SR - SL);
        double bm = (SR * UR[7] - SL * UL[7] + FL[7] - FR[7]) / (SR - SL);
        double bbn = bn * bn;

        double ro_LL = ro_L * (SL - u1) / (SL - SM);
        double ro_RR = ro_R * (SR - u2) / (SR - SM);
        double Q_LL = Q_L * (SL - u1) / (SL - SM);
        double Q_RR = Q_R * (SR - u2) / (SR - SM);

        if (metod == 2)   // HLLC  + mgd
        {
            double sbv1 = u1 * bn1 + v1 * bt1 + w1 * bm1;
            double sbv2 = u2 * bn2 + v2 * bt2 + w2 * bm2;

            double UZ0 = (SR * UR[0] - SL * UL[0] + FL[0] - FR[0]) / (SR - SL);
            double UZ1 = (SR * UR[1] - SL * UL[1] + FL[1] - FR[1]) / (SR - SL);
            double UZ2 = (SR * UR[2] - SL * UL[2] + FL[2] - FR[2]) / (SR - SL);
            double UZ3 = (SR * UR[3] - SL * UL[3] + FL[3] - FR[3]) / (SR - SL);
            double UZ4 = (SR * UR[4] - SL * UL[4] + FL[4] - FR[4]) / (SR - SL);
            double vzL, vzR, vLL, wLL, vRR, wRR, ppLR, btt1, bmm1, btt2, bmm2, ee1, ee2;


            double suRm = suR / (SR - SM);
            double suLm = suL / (SL - SM);
            double rzR = ro_R * suRm;
            double rzL = ro_L * suLm;

            double ptzR = pTR + ro_R * suR * (SM - u2);
            double ptzL = pTL + ro_L * suL * (SM - u1);
            double ptz = (ptzR + ptzL) / 2.0;


            vRR = UZ2 / UZ0;
            wRR = UZ3 / UZ0;
            vLL = vRR;
            wLL = wRR;

            /*vRR = v2 + bn * (bt2 - bt) / suR / ro_R;
            wRR = w2 + bn * (bm2 - bm) / suR / ro_R;
            vLL = v1 + bn * (bt1 - bt) / suL / ro_L;
            wLL = w1 + bn * (bm1 - bm) / suL / ro_L;*/

            btt2 = bt;
            bmm2 = bm;
            btt1 = btt2;
            bmm1 = bmm2;

            double sbvz = (bn * UZ1 + bt * UZ2 + bm * UZ3) / UZ0;

            ee2 = e2 * suRm + (ptz * SM - pTR * u2 + bn * (sbv2 - sbvz)) / (SR - SM);
            ee1 = e1 * suLm + (ptz * SM - pTL * u1 + bn * (sbv1 - sbvz)) / (SL - SM);

            /*if (fabs(bn) < 0.000001 )
            {
                vRR = v2;
                wRR = w2;
                vLL = v1;
                wLL = w1;
                btt2 = bt2 * suRm;
                bmm2 = bm2 * suRm;
                btt1 = bt1 * suLm;
                bmm1 = bm1 * suLm;
            }*/

            /*ppLR = (pTL + ro_L * (SL - u1) * (SM - u1) + pTR + ro_R * (SR - u2) * (SM - u2)) / 2.0;

            if (fabs(bn) < 0.000001)
            {
                vLL = v1;
                wLL = w1;
                vRR = v2;
                wRR = w2;

                btt1 = bt1 * (SL - u1) / (SL - SM);
                btt2 = bt2 * (SR - u2) / (SR - SM);

                bmm1 = bm1 * (SL - u1) / (SL - SM);
                bmm2 = bm2 * (SR - u2) / (SR - SM);

                ee1 = ((SL - u1) * e1 - pTL * u1 + ppLR * SM) / (SL - SM);
                ee2 = ((SR - u2) * e2 - pTL * u2 + ppLR * SM) / (SR - SM);
            }
            else
            {
                btt2 = btt1 = (SR * UR[6] - SL * UL[6] + FL[6] - FR[6]) / (SR - SL);
                bmm2 = bmm1 = (SR * UR[7] - SL * UL[7] + FL[7] - FR[7]) / (SR - SL);
                vLL = v1 + bn * (bt1 - btt1) / (ro_L * (SL - u1));
                vRR = v2 + bn * (bt2 - btt2) / (ro_R * (SR - u2));

                wLL = w1 + bn * (bm1 - bmm1) / (ro_L * (SL - u1));
                wRR = w2 + bn * (bm2 - bmm2) / (ro_R * (SR - u2));

                double sks1 = u1 * bn1 + v1 * bt1 + w1 * bm1 - SM * bn - vLL * btt1 - wLL * bmm1;
                double sks2 = u2 * bn2 + v2 * bt2 + w2 * bm2 - SM * bn - vRR * btt2 - wRR * bmm2;

                ee1 = ((SL - u1) * e1 - pTL * u1 + ppLR * SM + bn * sks1) / (SL - SM);
                ee2 = ((SR - u2) * e2 - pTR * u2 + ppLR * SM + bn * sks2) / (SR - SM);
            }*/


            double  ULL[9], URR[9], PO[9];
            ULL[0] = ro_LL;
            ULL[1] = ro_LL * SM;
            ULL[2] = ro_LL * vLL;
            ULL[3] = ro_LL * wLL;
            ULL[4] = ee1;
            ULL[5] = bn;
            ULL[6] = btt1;
            ULL[7] = bmm1;
            ULL[8] = Q_LL;

            URR[0] = ro_RR;
            URR[1] = ro_RR * SM;
            URR[2] = ro_RR * vRR;
            URR[3] = ro_RR * wRR;
            URR[4] = ee2;
            URR[5] = bn;
            URR[6] = btt2;
            URR[7] = bmm2;
            URR[8] = Q_RR;

            if (SL >= 0.0)
            {
                for (int i = 0; i < 9; i++)
                {
                    PO[i] = FL[i];
                }
            }
            else if (SL < 0.0 && SM >= 0.0)
            {
                for (int i = 0; i < 9; i++)
                {
                    PO[i] = FL[i] + SL * ULL[i] - SL * UL[i];
                }
            }
            else if (SR > 0.0 && SM < 0.0)
            {
                for (int i = 0; i < 9; i++)
                {
                    PO[i] = FR[i] + SR * URR[i] - SR * UR[i];
                }
            }
            else if (SR <= 0.0)
            {
                for (int i = 0; i < 9; i++)
                {
                    PO[i] = FR[i];
                }
            }



            double SN = max(fabs(SL), fabs(SR));

            PO[5] = -SN * (bn2 - bn1);

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
            P[7] = SWAP;
            return time;

        }
        else if (metod == 3)  // HLLD
        {

            double ttL = ro_L * suL * (SL - SM) - bbn;
            double ttR = ro_R * suR * (SR - SM) - bbn;

            double vLL, wLL, vRR, wRR, btt1, bmm1, btt2, bmm2;

            if (fabs(ttL) >= 0.000001)
            {
                vLL = v1 - bn * bt1 * (SM - u1) / ttL;
                wLL = w1 - bn * bm1 * (SM - u1) / ttL;
                btt1 = bt1 * (ro_L * suL * suL - bbn) / ttL;
                bmm1 = bm1 * (ro_L * suL * suL - bbn) / ttL;
            }
            else
            {
                vLL = v1;
                wLL = w1;
                btt1 = 0.0;
                bmm1 = 0.0;
            }

            if (fabs(ttR) >= 0.000001)
            {
                vRR = v2 - bn * bt2 * (SM - u2) / ttR;
                wRR = w2 - bn * bm2 * (SM - u2) / ttR;
                btt2 = bt2 * (ro_R * suR * suR - bbn) / ttR;
                bmm2 = bm2 * (ro_R * suR * suR - bbn) / ttR;
                //cout << "tbr = " << (ro_R * suR * suR - bbn) / ttR << endl;
                //cout << "bt2 = " << bt2 << endl;
            }
            else
            {
                vRR = v2;
                wRR = w2;
                btt2 = 0.0;
                bmm2 = 0.0;
            }

            double eLL = (e1 * suL + PTT * SM - pTL * u1 + bn * //
                ((u1 * bn1 + v1 * bt1 + w1 * bm1) - (SM * bn + vLL * btt1 + wLL * bmm1))) //
                / (SL - SM);
            double eRR = (e2 * suR + PTT * SM - pTR * u2 + bn * //
                ((u2 * bn2 + v2 * bt2 + w2 * bm2) - (SM * bn + vRR * btt2 + wRR * bmm2))) //
                / (SR - SM);

            double sqrtroLL = sqrt(ro_LL);
            double sqrtroRR = sqrt(ro_RR);
            double SLL = SM - fabs(bn) / sqrtroLL;
            double SRR = SM + fabs(bn) / sqrtroRR;

            double idbn = 1.0;
            if (fabs(bn) > 0.0001)
            {
                idbn = 1.0 * sign(bn);
            }
            else
            {
                idbn = 0.0;
                SLL = SM;
                SRR = SM;
            }

            double vLLL = (sqrtroLL * vLL + sqrtroRR * vRR + //
                idbn * (btt2 - btt1)) / (sqrtroLL + sqrtroRR);

            double wLLL = (sqrtroLL * wLL + sqrtroRR * wRR + //
                idbn * (bmm2 - bmm1)) / (sqrtroLL + sqrtroRR);

            double bttt = (sqrtroLL * btt2 + sqrtroRR * btt1 + //
                idbn * sqrtroLL * sqrtroRR * (vRR - vLL)) / (sqrtroLL + sqrtroRR);

            double bmmm = (sqrtroLL * bmm2 + sqrtroRR * bmm1 + //
                idbn * sqrtroLL * sqrtroRR * (wRR - wLL)) / (sqrtroLL + sqrtroRR);

            double eLLL = eLL - idbn * sqrtroLL * ((SM * bn + vLL * btt1 + wLL * bmm1) //
                - (SM * bn + vLLL * bttt + wLLL * bmmm));
            double eRRR = eRR + idbn * sqrtroRR * ((SM * bn + vRR * btt2 + wRR * bmm2) //
                - (SM * bn + vLLL * bttt + wLLL * bmmm));
            //cout << " = " << bn << " " << btt2 << " " << bmm2 << endl;
            //cout << "sbvr = " << (SM * bn + vRR * btt2 + wRR * bmm2) << endl;
            double  ULL[9], URR[9], ULLL[9], URRR[9];

            ULL[0] = ro_LL;
            ULL[1] = ro_LL * SM;
            ULL[2] = ro_LL * vLL;
            ULL[3] = ro_LL * wLL;
            ULL[4] = eLL;
            ULL[5] = bn;
            ULL[6] = btt1;
            ULL[7] = bmm1;
            ULL[8] = Q_LL;

            URR[0] = ro_RR;
            //cout << ro_RR << endl;
            URR[1] = ro_RR * SM;
            URR[2] = ro_RR * vRR;
            URR[3] = ro_RR * wRR;
            URR[4] = eRR;
            URR[5] = bn;
            URR[6] = btt2;
            URR[7] = bmm2;
            URR[8] = Q_RR;

            ULLL[0] = ro_LL;
            ULLL[1] = ro_LL * SM;
            ULLL[2] = ro_LL * vLLL;
            ULLL[3] = ro_LL * wLLL;
            ULLL[4] = eLLL;
            ULLL[5] = bn;
            ULLL[6] = bttt;
            ULLL[7] = bmmm;
            ULLL[8] = Q_LL;

            URRR[0] = ro_RR;
            URRR[1] = ro_RR * SM;
            URRR[2] = ro_RR * vLLL;
            URRR[3] = ro_RR * wLLL;
            URRR[4] = eRRR;
            URRR[5] = bn;
            URRR[6] = bttt;
            URRR[7] = bmmm;
            URRR[8] = Q_RR;

            double PO[9];

            if (SL >= 0.0)
            {
                //cout << "SL >= 0.0" << endl;
                for (int i = 0; i < 9; i++)
                {
                    PO[i] = FL[i];
                }
            }
            else if (SL < 0.0 && SLL >= 0.0)
            {
                //cout << "SL < 0.0 && SLL >= 0.0" << endl;
                for (int i = 0; i < 9; i++)
                {
                    PO[i] = FL[i] + SL * ULL[i] - SL * UL[i];
                }
                //cout << ULL[0] << endl;
            }
            else if (SLL <= 0.0 && SM >= 0.0)
            {
                //cout << "SLL <= 0.0 && SM >= 0.0" << endl;
                for (int i = 0; i < 9; i++)
                {
                    PO[i] = FL[i] + SLL * ULLL[i] - (SLL - SL) * ULL[i] - SL * UL[i];
                }
            }
            else if (SM < 0.0 && SRR > 0.0)
            {
                //cout << "SM < 0.0 && SRR > 0.0" << endl;
                for (int i = 0; i < 9; i++)
                {
                    PO[i] = FR[i] + SRR * URRR[i] - (SRR - SR) * URR[i] - SR * UR[i];
                }
                //cout << "P4 = " << URRR[4] << endl;
            }
            else if (SR > 0.0 && SRR <= 0.0)
            {
                //cout << "SR > 0.0 && SRR <= 0.0" << endl;
                for (int i = 0; i < 9; i++)
                {
                    PO[i] = FR[i] + SR * URR[i] - SR * UR[i];
                }
                //cout << URR[0] << endl;
            }
            else if (SR <= 0.0)
            {
                //cout << "SR <= 0.0" << endl;
                for (int i = 0; i < 9; i++)
                {
                    PO[i] = FR[i];
                }
            }



            double SN = max(fabs(SL), fabs(SR));

            PO[5] = -SN * (bn2 - bn1);

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
            P[7] = SWAP;
            return time;
        }

    }


/*{   // Не работает, если скорость грани не нулевая
    // Нормаль здесь единичная по осям координат

    double bx_L = Bx_L / spi4;
    double by_L = By_L / spi4;
    double bz_L = Bz_L / spi4;

    double bx_R = Bx_R / spi4;
    double by_R = By_R / spi4;
    double bz_R = Bz_R / spi4;

    double t1 = 0.0;
    double t2 = 0.0;
    double t3 = 0.0;

    double m1 = 0.0;
    double m2 = 0.0;
    double m3 = 0.0;

    if (n1 > 0.1)
    {
        t2 = 1.0;
        m3 = 1.0;
    }
    else if (n2 > 0.1)
    {
        t3 = 1.0;
        m1 = 1.0;
    }
    else if (n3 > 0.1)
    {
        t1 = 1.0;
        m2 = 1.0;
    }
    else if (n1 < -0.1)
    {
        t3 = -1.0;
        m2 = -1.0;
    }
    else if (n2 < -0.1)
    {
        t1 = -1.0;
        m3 = -1.0;
    }
    else if (n3 < -0.1)
    {
        t1 = -1.0;
        m2 = -1.0;
    }
    else
    {
        printf("EROROR 1421  normal_error\n");
    }


    double u1, v1, w1, u2, v2, w2;
    u1 = v1_L * n1 + v2_L * n2 + v3_L * n3;
    v1 = v1_L * t1 + v2_L * t2 + v3_L * t3;
    w1 = v1_L * m1 + v2_L * m2 + v3_L * m3;
    u2 = v1_R * n1 + v2_R * n2 + v3_R * n3;
    v2 = v1_R * t1 + v2_R * t2 + v3_R * t3;
    w2 = v1_R * m1 + v2_R * m2 + v3_R * m3;

    double bn1, bt1, bm1, bn2, bt2, bm2;
    bn1 = bx_L * n1 + by_L * n2 + bz_L * n3;
    bt1 = bx_L * t1 + by_L * t2 + bz_L * t3;
    bm1 = bx_L * m1 + by_L * m2 + bz_L * m3;
    bn2 = bx_R * n1 + by_R * n2 + bz_R * n3;
    bt2 = bx_R * t1 + by_R * t2 + bz_R * t3;
    bm2 = bx_R * m1 + by_R * m2 + bz_R * m3;

    double sqrtroL = sqrt(ro_L);
    double sqrtroR = sqrt(ro_R);
    double ca_L = bn1 / sqrtroL;
    double ca_R = bn2 / sqrtroR;
    double cL = sqrt(ggg * p_L / ro_L);
    double cR = sqrt(ggg * p_R / ro_R);

    double bb_L = kv(bx_L) + kv(by_L) + kv(bz_L);
    double bb_R = kv(bx_R) + kv(by_R) + kv(bz_R);

    double aL = (kv(bx_L) + kv(by_L) + kv(bz_L)) / ro_L;
    double aR = (kv(bx_L) + kv(by_L) + kv(bz_L)) / ro_L;

    double uu_L = (kv(v1_L) + kv(v2_L) + kv(v3_L)) / 2.0;
    double uu_R = (kv(v1_R) + kv(v2_R) + kv(v3_R)) / 2.0;

    //double cfL = sqrt((ggg * p_L + bb_L + //
    //    sqrt(kv(ggg * p_L + bb_L) - 4.0 * ggg * p_L * kv(bn1))) / (2.0 * ro_L));
    //double cfR = sqrt((ggg * p_R + bb_R + //
    //    sqrt(kv(ggg * p_R + bb_R) - 4.0 * ggg * p_L * kv(bn2))) / (2.0 * ro_R));

    //double cfL = sqrt((kv(cL) + kv(aL)) / 2.0 + 0.5 * sqrt(kv(kv(cL) + kv(aL)) - 4.0 * kv(cL) * kv(ca_L)));
    //double cfR = sqrt((kv(cR) + kv(aR)) / 2.0 + 0.5 * sqrt(kv(kv(cR) + kv(aR)) - 4.0 * kv(cR) * kv(ca_R)));

    double aaL = bn1 / sqrt(ro_L);
    double b2L = kv(bn1) + kv(bt1) + kv(bm1);
    double b21 = b2L / ro_L;
    //double cL = sqrt(ga * p_L / ro_L);
    double qp = sqrt(b21 + cL * (cL + 2.0 * aaL));
    double qm = sqrt(b21 + cL * (cL - 2.0 * aaL));
    double cfL = (qp + qm) / 2.0;

    double aaR = bn2 / sqrt(ro_R);
    double b2R = kv(bn2) + kv(bt2) + kv(bm2);
    double b22 = b2R / ro_R;
    //double cR = sqrt(ga * p_R / ro_R);
    qp = sqrt(b22 + cR * (cR + 2.0 * aaR));
    qm = sqrt(b22 + cR * (cR - 2.0 * aaR));
    double cfR = (qp + qm) / 2.0;


    //cout << "cfR = " << cfR << " " << bn2 << " " << bt2 << " " << bm2 << endl;
    double SL = min(u1, u2) - max(cfL, cfR);
    double SR = max(u1, u2) + max(cfL, cfR);

    double pTL = p_L + bb_L / 2.0;
    double pTR = p_R + bb_R / 2.0;

    double suR = (SR - u2);
    double suL = (SL - u1);

    double SM = (suR * ro_R * u2 - suL * ro_L * u1 - pTR + pTL) //
        / (suR * ro_R - suL * ro_L);

    double PTT = (suR * ro_R * pTL - suL * ro_L * pTR + ro_L * ro_R * suR * suL * (u2 - u1))//
        / (suR * ro_R - suL * ro_L);

    double UU = max(fabs(SL), fabs(SR));
    double time = krit * rad / UU;

    double FL[9], FR[9], UL[9], UR[9];

    double e1 = p_L / g1 + ro_L * uu_L + bb_L / 2.0;
    double e2 = p_R / g1 + ro_R * uu_R + bb_R / 2.0;


    FL[0] = ro_L * u1;
    FL[1] = ro_L * u1 * u1 + pTL - kv(bn1);
    FL[2] = ro_L * u1 * v1 - bn1 * bt1;
    FL[3] = ro_L * u1 * w1 - bn1 * bm1;
    FL[4] = (e1 + pTL) * u1 - bn1 * (u1 * bn1 + v1 * bt1 + w1 * bm1);
    FL[5] = 0.0;
    FL[6] = u1 * bt1 - v1 * bn1;
    FL[7] = u1 * bm1 - w1 * bn1;
    FL[8] = Q_L * u1;

    FR[0] = ro_R * u2;
    FR[1] = ro_R * u2 * u2 + pTR - kv(bn2);
    FR[2] = ro_R * u2 * v2 - bn2 * bt2;
    FR[3] = ro_R * u2 * w2 - bn2 * bm2;
    FR[4] = (e2 + pTR) * u2 - bn2 * (u2 * bn2 + v2 * bt2 + w2 * bm2);
    FR[5] = 0.0;
    FR[6] = u2 * bt2 - v2 * bn2;
    FR[7] = u2 * bm2 - w2 * bn2;
    FR[8] = Q_R * u2;

    UL[0] = ro_L;
    UL[1] = ro_L * u1;
    UL[2] = ro_L * v1;
    UL[3] = ro_L * w1;
    UL[4] = e1;
    UL[5] = bn1;
    UL[6] = bt1;
    UL[7] = bm1;
    UL[8] = Q_L;

    UR[0] = ro_R;
    UR[1] = ro_R * u2;
    UR[2] = ro_R * v2;
    UR[3] = ro_R * w2;
    UR[4] = e2;
    UR[5] = bn2;
    UR[6] = bt2;
    UR[7] = bm2;
    UR[8] = Q_R;

    double bn = (SR * UR[5] - SL * UL[5] + FL[5] - FR[5]) / (SR - SL);
    double bbn = bn * bn;

    double ro_LL = ro_L * (SL - u1) / (SL - SM);
    double ro_RR = ro_R * (SR - u2) / (SR - SM);
    double Q_LL = Q_L * (SL - u1) / (SL - SM);
    double Q_RR = Q_R * (SR - u2) / (SR - SM);


    double ttL = ro_L * suL * (SL - SM) - bbn;
    double ttR = ro_R * suR * (SR - SM) - bbn;

    double vLL, wLL, vRR, wRR, btt1, bmm1, btt2, bmm2;

    if (fabs(ttL) >= 0.0000001)
    {
        vLL = v1 - bn * bt1 * (SM - u1) / ttL;
        wLL = w1 - bn * bm1 * (SM - u1) / ttL;
        btt1 = bt1 * (ro_L * suL * suL - bbn) / ttL;
        bmm1 = bm1 * (ro_L * suL * suL - bbn) / ttL;
    }
    else
    {
        vLL = v1;
        wLL = w1;
        btt1 = 0.0;
        bmm1 = 0.0;
    }

    if (fabs(ttR) >= 0.0000001)
    {
        vRR = v2 - bn * bt2 * (SM - u2) / ttR;
        wRR = w2 - bn * bm2 * (SM - u2) / ttR;
        btt2 = bt2 * (ro_R * suR * suR - bbn) / ttR;
        bmm2 = bm2 * (ro_R * suR * suR - bbn) / ttR;
    }
    else
    {
        vRR = v2;
        wRR = w2;
        btt2 = 0.0;
        bmm2 = 0.0;
    }

    double eLL = (e1 * suL + PTT * SM - pTL * u1 + bn * //
        ((u1 * bn1 + v1 * bt1 + w1 * bm1) - (SM * bn + vLL * btt1 + wLL * bmm1))) //
        / (SL - SM);
    double eRR = (e2 * suR + PTT * SM - pTR * u2 + bn * //
        ((u2 * bn2 + v2 * bt2 + w2 * bm2) - (SM * bn + vRR * btt2 + wRR * bmm2))) //
        / (SR - SM);

    double sqrtroLL = sqrt(ro_LL);
    double sqrtroRR = sqrt(ro_RR);
    double SLL = SM - fabs(bn) / sqrtroLL;
    double SRR = SM + fabs(bn) / sqrtroRR;

    double idbn = 1.0;
    if (fabs(bn) > 0.000001)
    {
        idbn = 1.0 * sign(bn);
    }
    else
    {
        idbn = 0.0;
        SLL = SM;
        SRR = SM;
    }

    double vLLL = (sqrtroLL * vLL + sqrtroRR * vRR + //
        idbn * (btt2 - btt1)) / (sqrtroLL + sqrtroRR);

    double wLLL = (sqrtroLL * wLL + sqrtroRR * wRR + //
        idbn * (bmm2 - bmm1)) / (sqrtroLL + sqrtroRR);

    double bttt = (sqrtroLL * btt2 + sqrtroRR * btt1 + //
        idbn * sqrtroLL * sqrtroRR * (vRR - vLL)) / (sqrtroLL + sqrtroRR);

    double bmmm = (sqrtroLL * bmm2 + sqrtroRR * bmm1 + //
        idbn * sqrtroLL * sqrtroRR * (wRR - wLL)) / (sqrtroLL + sqrtroRR);

    double eLLL = eLL - idbn * sqrtroLL * ((SM * bn + vLL * btt1 + wLL * bmm1) //
        - (SM * bn + vLLL * bttt + wLLL * bmmm));
    double eRRR = eRR + idbn * sqrtroRR * ((SM * bn + vRR * btt2 + wRR * bmm2) //
        - (SM * bn + vLLL * bttt + wLLL * bmmm));

    double  ULL[9], URR[9], ULLL[9], URRR[9];

    ULL[0] = ro_LL;
    ULL[1] = ro_LL * SM;
    ULL[2] = ro_LL * vLL;
    ULL[3] = ro_LL * wLL;
    ULL[4] = eLL;
    ULL[5] = bn;
    ULL[6] = btt1;
    ULL[7] = bmm1;
    ULL[8] = Q_LL;

    URR[0] = ro_RR;
    URR[1] = ro_RR * SM;
    URR[2] = ro_RR * vRR;
    URR[3] = ro_RR * wRR;
    URR[4] = eRR;
    URR[5] = bn;
    URR[6] = btt2;
    URR[7] = bmm2;
    URR[8] = Q_RR;

    ULLL[0] = ro_LL;
    ULLL[1] = ro_LL * SM;
    ULLL[2] = ro_LL * vLLL;
    ULLL[3] = ro_LL * wLLL;
    ULLL[4] = eLLL;
    ULLL[5] = bn;
    ULLL[6] = bttt;
    ULLL[7] = bmmm;
    ULLL[8] = Q_LL;

    URRR[0] = ro_RR;
    URRR[1] = ro_RR * SM;
    URRR[2] = ro_RR * vLLL;
    URRR[3] = ro_RR * wLLL;
    URRR[4] = eRRR;
    URRR[5] = bn;
    URRR[6] = bttt;
    URRR[7] = bmmm;
    URRR[8] = Q_RR;

    double PO[9];

    if (SL >= 0.0)
    {
        for (int i = 0; i < 9; i++)
        {
            PO[i] = FL[i];
        }
    }
    else if (SL < 0.0 && SLL >= 0.0)
    {
        for (int i = 0; i < 9; i++)
        {
            PO[i] = FL[i] + SL * ULL[i] - SL * UL[i];
        }
    }
    else if (SLL <= 0.0 && SM >= 0.0)
    {
        for (int i = 0; i < 9; i++)
        {
            PO[i] = FL[i] + SLL * ULLL[i] - (SLL - SL) * ULL[i] - SL * UL[i];
        }
    }
    else if (SM < 0.0 && SRR > 0.0)
    {
        for (int i = 0; i < 9; i++)
        {
            PO[i] = FR[i] + SRR * URRR[i] - (SRR - SR) * URR[i] - SR * UR[i];
        }
    }
    else if (SR > 0.0 && SRR <= 0.0)
    {
        for (int i = 0; i < 9; i++)
        {
            PO[i] = FR[i] + SR * URR[i] - SR * UR[i];
        }
    }
    else if (SR <= 0.0)
    {
        for (int i = 0; i < 9; i++)
        {
            PO[i] = FR[i];
        }
    }



    double SN = max(fabs(SL), fabs(SR));

    PO[5] = -SN * (bn2 - bn1);

    P[1] = n1 * PO[1] + t1 * PO[2] + m1 * PO[3];
    P[2] = n2 * PO[1] + t2 * PO[2] + m2 * PO[3];
    P[3] = n3 * PO[1] + t3 * PO[2] + m3 * PO[3];
    P[5] = spi4 * (n1 * PO[5] + t1 * PO[6] + m1 * PO[7]);
    P[6] = spi4 * (n2 * PO[5] + t2 * PO[6] + m2 * PO[7]);
    P[7] = spi4 * (n3 * PO[5] + t3 * PO[6] + m3 * PO[7]);
    P[0] = PO[0];
    PQ = PO[8];

    double SWAP = P[4];
    P[4] = P[5];
    P[5] = P[6];
    P[6] = P[7];
    P[7] = SWAP;
    return time;

}*/