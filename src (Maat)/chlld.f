******7**10**********************************************************70
**                                                                  **
**    chlld   for                                                   **
**                                                                  **
******7**10**********************************************************70
**    shema HLLD(modification) for 3D MHD                           **
******7**10**********************************************************70
      subroutine chlld(id_bn,n_state,n_disco,KOBL,i_in,j_in,k_in,kdir,
     @                            al,be,ge,el,
     @                            w,qqq1,qqq2,
     @                            dsl,dsp,dsc,ythll,
     @                            qqq)
      implicit real*8 (a-h,o-z)
      dimension qqq(8),qqq1(8),qqq2(8)
      dimension FR(8),FL(8)
      dimension FW(8),UL(8),UZ(8),UR(8)
      dimension UZL(8),UZR(8)
      dimension UZZL(8),UZZR(8)
      dimension dq(8)

      dimension vL(3),vR(3),bL(3),bR(3)
      dimension vzL(3),vzR(3),bzL(3),bzR(3)
      dimension vzzL(3),vzzR(3),bzzL(3),bzzR(3)
      dimension aco(3,3),qv(3),qb(3)

c     common /test_hll/ SR,SL,SM,SZL,SZR

      data x0,x1,x2,x3,x4,x5,x6,x7,x8,x9/0.,1.,2.,3.,4.,5.,6.,7.,8.,9./


c-------  n_state=0   - one speed LAX
c-------  n_state=1   - two speed LAX (HLL,(Harten-Lax-van-Leer))
c-------  n_state=2   - two-state (3 speed) HLLC (Contact Discontinuity)
c-------  n_state=3   - multi-state (5 speed) HLLD (All Discontinuity)


      pi=dacos(-x1)
      cpi4=x4*pi
      cpi8=x8*pi
      spi4=dsqrt(cpi4)

      eps=1.D-12
      epsb=1.D-06
      eps_p=1.D-06
      eps_d=1.D-03
      

       ga=x5/x3
       g1=ga-x1


         wv=w


         r1 =qqq1(1)
         u1 =qqq1(2)
         v1 =qqq1(3)
         w1 =qqq1(4)
         p1 =qqq1(5)
         bx1=qqq1(6)/spi4
         by1=qqq1(7)/spi4
         bz1=qqq1(8)/spi4


         r2 =qqq2(1)
         u2 =qqq2(2)
         v2 =qqq2(3)
         w2 =qqq2(4)
         p2 =qqq2(5)
         bx2=qqq2(6)/spi4
         by2=qqq2(7)/spi4
         bz2=qqq2(8)/spi4

         ro=(r2+r1)/x2
         au=(u2+u1)/x2
         av=(v2+v1)/x2
         aw=(w2+w1)/x2
         ap=(p2+p1)/x2
         abx=(bx2+bx1)/x2
         aby=(by2+by1)/x2
         abz=(bz2+bz1)/x2


         bk=abx*al+aby*be+abz*ge
         b2=abx**2+aby**2+abz**2

          d=b2-(bk)**2
         aco(1,1)=al
         aco(2,1)=be
         aco(3,1)=ge
         if(d.gt.eps)then
         d=dsqrt(d)
         aco(1,2)=(abx-bk*al)/d
         aco(2,2)=(aby-bk*be)/d
         aco(3,2)=(abz-bk*ge)/d
         aco(1,3)=(aby*ge-abz*be)/d
         aco(2,3)=(abz*al-abx*ge)/d
         aco(3,3)=(abx*be-aby*al)/d
         else
         if(dabs(al).lt.dabs(be).and.dabs(al).lt.dabs(ge))then
          aix=x1
          aiy=x0
          aiz=x0
         elseif(dabs(be).lt.dabs(ge))then
          aix=x0
          aiy=x1
          aiz=x0
         else
          aix=x0
          aiy=x0
          aiz=x1
         endif
          aik=aix*al+aiy*be+aiz*ge
          d=dsqrt(x1-aik**2)
         aco(1,2)=(aix-aik*al)/d
         aco(2,2)=(aiy-aik*be)/d
         aco(3,2)=(aiz-aik*ge)/d
         aco(1,3)=(aiy*ge-aiz*be)/d
         aco(2,3)=(aiz*al-aix*ge)/d
         aco(3,3)=(aix*be-aiy*al)/d
         endif

          do i = 1, 3
          vL(i)=aco(1,i)*u1+aco(2,i)*v1+aco(3,i)*w1
          vR(i)=aco(1,i)*u2+aco(2,i)*v2+aco(3,i)*w2
          bL(i)=aco(1,i)*bx1+aco(2,i)*by1+aco(3,i)*bz1
          bR(i)=aco(1,i)*bx2+aco(2,i)*by2+aco(3,i)*bz2
          enddo

                aaL= bL(1)/dsqrt(r1)
                b2L= bL(1)**2+bL(2)**2+bL(3)**2
                b21=b2L/r1
                cL =dsqrt(ga*p1/r1)
                qp=dsqrt(b21+cL*(cL+x2*aaL))
                qm=dsqrt(b21+cL*(cL-x2*aaL))
                cfL=(qp+qm)/x2
                ptL=p1+b2L/x2

                aaR= bR(1)/dsqrt(r2)
                b2R= bR(1)**2+bR(2)**2+bR(3)**2
                b22=b2R/r2
                cR =dsqrt(ga*p2/r2)
                qp=dsqrt(b22+cR*(cR+x2*aaR))
                qm=dsqrt(b22+cR*(cR-x2*aaR))
                cfR=(qp+qm)/x2
                ptR=p2+b2R/x2

                aC = (aaL+aaR)/x2
                b2o=(b22+b21)/x2
                cC =dsqrt(ga*ap/ro)
                qp=dsqrt(b2o+cC*(cC+x2*aC))
                qm=dsqrt(b2o+cC*(cC-x2*aC))
                cfC=(qp+qm)/x2
                vC1=(vL(1)+vR(1))/x2

        if(n_disco.eq.1)then
        SL=dmin1( (vL(1)-cfL),(vC1-cfC) )
        SR=dmax1( (vR(1)+cfR),(vC1+cfC) )
        endif

c       SL=dmin1( (vL(1)-cfL),(vR(1)-cfR),(vC1-cfC) )
c       SR=dmax1( (vL(1)+cfL),(vR(1)+cfR),(vC1+cfC) )

        if(n_disco.eq.0)then
        SL=dmin1( (vL(1)-cfL),(vR(1)-cfR) )
        SR=dmax1( (vL(1)+cfL),(vR(1)+cfR) )
        endif

        if(n_disco.eq.2)then
        SL_1=dmin1( (vL(1)-cfL),(vC1-cfC) )
        SR_1=dmax1( (vR(1)+cfR),(vC1+cfC) )
        SL_2=dmin1( (vL(1)-cfL),(vR(1)-cfR) )
        SR_2=dmax1( (vL(1)+cfL),(vR(1)+cfR) )
              oo = 0.75d0
              oo1= 1.d0-oo
          SL= oo*SL_1 + oo1*SL_2
          SR= oo*SR_1 + oo1*SR_2
        endif

c       SL=vC1-cfC
c       SR=vC1+cfC

c               cf=dmax1(cfL,cfR)

c           SL=dmin1(vL(1),vR(1))-cf
c           SR=dmax1(vL(1),vR(1))+cf


                 suR=SR-vR(1)
                 suL=SL-vL(1)
            SM=(suR*r2*vR(1)-ptR+ptL-suL*r1*vL(1))
     .             /(suR*r2-suL*r1)

            dsl=SL
            dsc=SM
            dsp=SR

            if(id_bn.eq.-1)then
c                   dsc=(vR(1)+vL(1))/2.d0
                    return
            endif

c           cfC1=dmax1(cfC,cfL,cfR)

c           dsl=vC1-cfC1
c           dsp=vC1+cfC1

         if(n_state.eq.0)then
            TR0=dabs(vL(1)+vR(1))/x2+cfC
            TL0=-TR0
            SR=TR0
            SL=TL0
         endif


         upt1=(u1**2+v1**2+w1**2)/x2
         sbv1=u1*bx1+v1*by1+w1*bz1

         upt2=(u2**2+v2**2+w2**2)/x2
         sbv2=u2*bx2+v2*by2+w2*bz2

         e1=p1/g1+r1*upt1+b2L/x2
         e2=p2/g1+r2*upt2+b2R/x2

         FL(1)=r1*vL(1)
         FL(2)=r1*vL(1)*vL(1)+ptL-bL(1)**2
         FL(3)=r1*vL(1)*vL(2)    -bL(1)*bL(2)
         FL(4)=r1*vL(1)*vL(3)    -bL(1)*bL(3)
         FL(5)=(e1+ptL)*vL(1)    -bL(1)*sbv1
         FL(6)=x0 
         FL(7)=vL(1)*bL(2)-vL(2)*bL(1)  
         FL(8)=vL(1)*bL(3)-vL(3)*bL(1)  

         FR(1)=r2*vR(1)
         FR(2)=r2*vR(1)*vR(1)+ptR-bR(1)**2
         FR(3)=r2*vR(1)*vR(2)    -bR(1)*bR(2)
         FR(4)=r2*vR(1)*vR(3)    -bR(1)*bR(3)
         FR(5)=(e2+ptR)*vR(1)    -bR(1)*sbv2
         FR(6)=x0
         FR(7)=vR(1)*bR(2)-vR(2)*bR(1)  
         FR(8)=vR(1)*bR(3)-vR(3)*bR(1)  

               UL(1)=r1
               UL(5)=e1
               UR(1)=r2
               UR(5)=e2
               do ik=1,3
               UL(ik+1)=r1*vL(ik)
               UL(ik+5)=   bL(ik)
               UR(ik+1)=r2*vR(ik)
               UR(ik+5)=   bR(ik)
               enddo

       do ik=1,8
       UZ(ik)=(SR*UR(ik)-SL*UL(ik)+FL(ik)-FR(ik))/(SR-SL)
       enddo

c-------- choise for Bn [=UZ(6)] through fan:
       if(id_bn.eq.1)UZ(6)=x0
        
c----
        if(n_state.le.1)then

                do ik=1,8
                dq(ik)=UR(ik)-UL(ik)
                enddo

                   TL=SL
                   TR=SR
                if(SL.gt.wv)then
                   TL=x0
                   do ik=1, 8
                   FW(ik)=wv*UL(ik)
                   enddo
                endif
                if(SL.le.wv.and.wv.le.SR)then
                   do ik=1, 8
                   FW(ik)=wv*UZ(ik)
                   enddo
                endif
                if(SR.lt.wv)then
                   TR=x0
                   do ik=1, 8
                   FW(ik)=wv*UR(ik)
                   enddo
                endif


        a=TR*TL
        b=TR-TL

        qqq(1)=(TR*FL(1)-TL*FR(1)+a*dq(1))/b-FW(1)
        qqq(5)=(TR*FL(5)-TL*FR(5)+a*dq(5))/b-FW(5)
        do ik=2,4
            qv(ik-1)=(TR*FL(ik)-TL*FR(ik)+a*dq(ik))/b-FW(ik)
        enddo
        do ik=6,8
            qb(ik-5)=(TR*FL(ik)-TL*FR(ik)+a*dq(ik))/b-FW(ik)
        enddo

        do i = 1,3
        qqq(i+1)=aco(i,1)*qv(1)+aco(i,2)*qv(2)+aco(i,3)*qv(3)
        qqq(i+5)=aco(i,1)*qb(1)+aco(i,2)*qb(2)+aco(i,3)*qb(3)
        qqq(i+5)=spi4*qqq(i+5)
        enddo

        do ik=1,8
         qqq(ik)=ythll*el*qqq(ik)
        enddo
           return
        endif
c----
        if(n_state.eq.2)then

           suRm=suR/(SR-SM)
           suLm=suL/(SL-SM)
           rzR=r2*suRm
           rzL=r1*suLm
           vzR(1)=SM
           vzL(1)=SM
           ptzR=ptR+r2*suR*(SM-vR(1))
           ptzL=ptL+r1*suL*(SM-vL(1))
           ptz=(ptzR+ptzL)/x2
           bzR(1)=UZ(6)!bR(1)
           bzL(1)=UZ(6)!bL(1)

           vzR(2)=UZ(3)/UZ(1)
           vzR(3)=UZ(4)/UZ(1)
           vzL(2)=vzR(2)
           vzL(3)=vzR(3)

           vzR(2)=vR(2)+UZ(6)*(bR(2)-UZ(7))/suR/r2
           vzR(3)=vR(3)+UZ(6)*(bR(3)-UZ(8))/suR/r2
           vzL(2)=vL(2)+UZ(6)*(bL(2)-UZ(7))/suL/r1
           vzL(3)=vL(3)+UZ(6)*(bL(3)-UZ(8))/suL/r1

           bzR(2)=UZ(7)
           bzR(3)=UZ(8)
           bzL(2)=bzR(2)
           bzL(3)=bzR(3)

           sbvz=( UZ(6)*UZ(2)+UZ(7)*UZ(3)+UZ(8)*UZ(4) )/UZ(1)

           ezR=e2*suRm+(ptz*SM-ptR*vR(1)+UZ(6)*(sbv2-sbvz))/(SR-SM)
           ezL=e1*suLm+(ptz*SM-ptL*vL(1)+UZ(6)*(sbv1-sbvz))/(SL-SM)

      if(dabs(UZ(6)).lt.epsb)then
           vzR(2)=vR(2)
           vzR(3)=vR(3)
           vzL(2)=vL(2)
           vzL(3)=vL(3)
           bzR(2)=bR(2)*suRm
           bzR(3)=bR(3)*suRm
           bzL(2)=bL(2)*suLm
           bzL(3)=bL(3)*suLm
      endif
             UZL(1)=rzL
             UZL(5)=ezL
             UZR(1)=rzR
             UZR(5)=ezR
             do ik=1,3
             UZL(ik+1)=vzL(ik)*rzL
             UZL(ik+5)=bzL(ik)
             UZR(ik+1)=vzR(ik)*rzR
             UZR(ik+5)=bzR(ik)
             enddo

           if(SL.gt.wv)then
             qqq(1)=FL(1)-wv*UL(1)
             qqq(5)=FL(5)-wv*UL(5)
             do ik=2,4
                 qv(ik-1)=FL(ik)-wv*UL(ik)
             enddo
             do ik=6,8
                 qb(ik-5)=FL(ik)-wv*UL(ik)
             enddo
           endif

           if(SL.le.wv.and.SM.ge.wv)then
             qqq(1)=FL(1)+SL*(rzL-r1) -wv*UZL(1)
             qqq(5)=FL(5)+SL*(ezL-e1) -wv*UZL(5)
             do ik=2,4
         qv(ik-1)=FL(ik)+SL*(UZL(ik)-UL(ik))-wv*UZL(ik)
             enddo
             do ik=6,8
         qb(ik-5)=FL(ik)+SL*(UZL(ik)-UL(ik))-wv*UZL(ik)
             enddo
           endif

           if(SM.le.wv.and.SR.ge.wv)then
             qqq(1)=FR(1)+SR*(rzR-r2) -wv*UZR(1)
             qqq(5)=FR(5)+SR*(ezR-e2) -wv*UZR(5)
             do ik=2,4
         qv(ik-1)=FR(ik)+SR*(UZR(ik)-UR(ik))-wv*UZR(ik)
             enddo
             do ik=6,8
         qb(ik-5)=FR(ik)+SR*(UZR(ik)-UR(ik))-wv*UZR(ik)
             enddo
           endif

           if(SR.lt.wv)then
             qqq(1)=FR(1)-wv*UR(1)
             qqq(5)=FR(5)-wv*UR(5)
             do ik=2,4
                 qv(ik-1)=FR(ik)-wv*UR(ik)
             enddo
             do ik=6,8
                 qb(ik-5)=FR(ik)-wv*UR(ik)
             enddo
           endif

        do i = 1,3
        qqq(i+1)=aco(i,1)*qv(1)+aco(i,2)*qv(2)+aco(i,3)*qv(3)
        qqq(i+5)=aco(i,1)*qb(1)+aco(i,2)*qb(2)+aco(i,3)*qb(3)
        qqq(i+5)=spi4*qqq(i+5)
        enddo

        do ik=1,8
         qqq(ik)=ythll*el*qqq(ik)
        enddo

           return
        endif
c----
        if(n_state.eq.3)then

        ptz=(suR*r2*ptL-suL*r1*ptR+r1*r2*suR*suL*(vR(1)-vL(1)))
     .      /(suR*r2-suL*r1)

        vzL(1)=SM
        vzR(1)=SM
        vzzL(1)=SM
        vzzR(1)=SM
        ptzL=ptz
        ptzR=ptz
        ptzzL=ptz
        ptzzR=ptz

           suRm=suR/(SR-SM)
           suLm=suL/(SL-SM)
           rzR=r2*suRm
           rzL=r1*suLm

           bn=UZ(6)
           bn2=bn*bn
           bzL(1)=bn
           bzR(1)=bn
           bzzL(1)=bn
           bzzR(1)=bn
           
           ttR=r2*suR*(SR-SM)-bn2
           if(dabs(ttR).le.1.D-09)then
           tvR=x0
           tbR=x0
           else
           tvR=(SM-vR(1))/ttR
           tbR=(r2*suR*suR-bn2)/ttR
           endif

           ttL=r1*suL*(SL-SM)-bn2
           if(dabs(ttL).le.1.D-09)then
           tvL=x0
           tbL=x0
           else
           tvL=(SM-vL(1))/ttL
           tbL=(r1*suL*suL-bn2)/ttL
           endif

        vzL(2)=vL(2)-bn*bL(2)*tvL
        vzL(3)=vL(3)-bn*bL(3)*tvL
        vzR(2)=vR(2)-bn*bR(2)*tvR
        vzR(3)=vR(3)-bn*bR(3)*tvR

        bzL(2)=bL(2)*tbL
        bzL(3)=bL(3)*tbL
        bzR(2)=bR(2)*tbR
        bzR(3)=bR(3)*tbR

           sbvL=bzL(1)*vzL(1)+bzL(2)*vzL(2)+bzL(3)*vzL(3)
           sbvR=bzR(1)*vzR(1)+bzR(2)*vzR(2)+bzR(3)*vzR(3)

           ezR=e2*suRm+(ptz*SM-ptR*vR(1)+bn*(sbv2-sbvR))/(SR-SM)
           ezL=e1*suLm+(ptz*SM-ptL*vL(1)+bn*(sbv1-sbvL))/(SL-SM)

           rzzR=rzR
           rzzL=rzL
           rzRs=dsqrt(rzR)
           rzLs=dsqrt(rzL)
           rzss=rzRs+rzLs
           rzps=rzRs*rzLs

           SZL=SM-dabs(bn)/rzLs
           SZR=SM+dabs(bn)/rzRs

                   ibn=0
           if(dabs(bn).gt.epsb)then
                   sbn=dabs(bn)/bn
                   ibn=1
           else
                   sbn=x0
                   ibn=0
                   SZL=SM
                   SZR=SM
           endif

           vzzL(2)=(rzLs*vzL(2)+rzRs*vzR(2)
     .             +sbn*(bzR(2)-bzL(2)) )/rzss
           vzzL(3)=(rzLs*vzL(3)+rzRs*vzR(3)
     .             +sbn*(bzR(3)-bzL(3)) )/rzss
           vzzR(2)=vzzL(2)
           vzzR(3)=vzzL(3)

           bzzL(2)=(rzLs*bzR(2)+rzRs*bzL(2)
     .             +sbn*rzps*(vzR(2)-vzL(2)) )/rzss
           bzzL(3)=(rzLs*bzR(3)+rzRs*bzL(3)
     .             +sbn*rzps*(vzR(3)-vzL(3)) )/rzss
           bzzR(2)=bzzL(2)
           bzzR(3)=bzzL(3)

           sbzz=bzzL(1)*vzzL(1)+bzzL(2)*vzzL(2)+bzzL(3)*vzzL(3)

           ezzR=ezR+rzRs*sbn*(sbvR-sbzz)
           ezzL=ezL-rzLs*sbn*(sbvL-sbzz)

             UZL(1)=rzL
             UZL(5)=ezL
             UZR(1)=rzR
             UZR(5)=ezR
             do ik=1,3
             UZL(ik+1)=vzL(ik)*rzL
             UZL(ik+5)=bzL(ik)
             UZR(ik+1)=vzR(ik)*rzR
             UZR(ik+5)=bzR(ik)
             enddo

             UZZL(1)=rzzL
             UZZL(5)=ezzL
             UZZR(1)=rzzR
             UZZR(5)=ezzR
             do ik=1,3
             UZZL(ik+1)=vzzL(ik)*rzzL
             UZZL(ik+5)=bzzL(ik)
             UZZR(ik+1)=vzzR(ik)*rzzR
             UZZR(ik+5)=bzzR(ik)
             enddo

             j_ccs=-1

           if(SL.gt.wv)then
             qqq(1)=FL(1)-wv*UL(1)
             qqq(5)=FL(5)-wv*UL(5)
             do ik=2,4
                 qv(ik-1)=FL(ik)-wv*UL(ik)
             enddo
             do ik=6,8
                 qb(ik-5)=FL(ik)-wv*UL(ik)
             enddo
             j_ccs= 1
           endif

           if(SL.le.wv.and.SZL.ge.wv)then
              ik=1     
          qqq(ik)=FL(ik)+SL*(UZL(ik)-UL(ik))-wv*UZL(ik)
              ik=5     
          qqq(ik)=FL(ik)+SL*(UZL(ik)-UL(ik))-wv*UZL(ik)
             do ik=2,4
         qv(ik-1)=FL(ik)+SL*(UZL(ik)-UL(ik))-wv*UZL(ik)
             enddo
             do ik=6,8
         qb(ik-5)=FL(ik)+SL*(UZL(ik)-UL(ik))-wv*UZL(ik)
             enddo
             j_ccs= 2
           endif
c------ FZZ
       if(ibn.eq.1)then

           if(SZL.le.wv.and.SM.ge.wv)then
              ik=1
          qqq(ik)=FL(ik)+SZL*(UZZL(ik)-UZL(ik))
     .                  + SL*( UZL(ik)- UL(ik))-wv*UZZL(ik)
              ik=5
          qqq(ik)=FL(ik)+SZL*(UZZL(ik)-UZL(ik))
     .                  + SL*( UZL(ik)- UL(ik))-wv*UZZL(ik)
             do ik=2,4
         qv(ik-1)=FL(ik)+SZL*(UZZL(ik)-UZL(ik))
     .                  + SL*( UZL(ik)- UL(ik))-wv*UZZL(ik)
             enddo
             do ik=6,8
         qb(ik-5)=FL(ik)+SZL*(UZZL(ik)-UZL(ik))
     .                  + SL*( UZL(ik)- UL(ik))-wv*UZZL(ik)
             enddo
             j_ccs= 3
           endif

           if(SM.le.wv.and.SZR.ge.wv)then
              ik=1
          qqq(ik)=FR(ik)+SZR*(UZZR(ik)-UZR(ik))
     .                  + SR*( UZR(ik)- UR(ik))-wv*UZZR(ik)
              ik=5
          qqq(ik)=FR(ik)+SZR*(UZZR(ik)-UZR(ik))
     .                  + SR*( UZR(ik)- UR(ik))-wv*UZZR(ik)
             do ik=2,4
         qv(ik-1)=FR(ik)+SZR*(UZZR(ik)-UZR(ik))
     .                  + SR*( UZR(ik)- UR(ik))-wv*UZZR(ik)
             enddo
             do ik=6,8
         qb(ik-5)=FR(ik)+SZR*(UZZR(ik)-UZR(ik))
     .                  + SR*( UZR(ik)- UR(ik))-wv*UZZR(ik)
             enddo
             j_ccs= 4
           endif

       endif
c------ 
           if(SZR.le.wv.and.SR.ge.wv)then
              ik=1
          qqq(ik)=FR(ik)+SR*(UZR(ik)-UR(ik))-wv*UZR(ik)
              ik=5
          qqq(ik)=FR(ik)+SR*(UZR(ik)-UR(ik))-wv*UZR(ik)
             do ik=2,4
         qv(ik-1)=FR(ik)+SR*(UZR(ik)-UR(ik))-wv*UZR(ik)
             enddo
             do ik=6,8
         qb(ik-5)=FR(ik)+SR*(UZR(ik)-UR(ik))-wv*UZR(ik)
             enddo
             j_ccs= 5
           endif

           if(SR.lt.wv)then
             qqq(1)=FR(1)-wv*UR(1)
             qqq(5)=FR(5)-wv*UR(5)
             do ik=2,4
                 qv(ik-1)=FR(ik)-wv*UR(ik)
             enddo
             do ik=6,8
                 qb(ik-5)=FR(ik)-wv*UR(ik)
             enddo
             j_ccs= 6
           endif

           if(j_ccs.eq.-1)then
              print*,'HLLD solver, nstate=3, wrong choise!!!!',j_ccs
              print*,'w SL SZL SM SZR SR'
              print*,w,SL,SZL,SM,SZR,SR
              pause
           endif

c----- Bn
             SN = dmax1(dabs(SL),dabs(SR))

                     wbn=x0
             if(wv.ge.SR)then
                     wbn=wv*bR(1)
             elseif(wv.le.SL)then
                     wbn=wv*bL(1)
             else
                     wbn=wv*(bL(1)+bR(1))/x2
             endif

             qb(1)=-SN*(bR(1)-bL(1))-wbn

c-----

        do i = 1,3
        qqq(i+1)=aco(i,1)*qv(1)+aco(i,2)*qv(2)+aco(i,3)*qv(3)
        qqq(i+5)=aco(i,1)*qb(1)+aco(i,2)*qb(2)+aco(i,3)*qb(3)
        qqq(i+5)=spi4*qqq(i+5)
        enddo

        do ik=1,8
         qqq(ik)=ythll*el*qqq(ik)
        enddo


           return
        endif


      return
      end
******7**10**********************************************************70

