# https://github.com/PrincetonUniversity/Athena-Cversion

```console
apps/rps.c:		Real radiusgpu = 0.000001;
apps/rps.c:		Epotdm = -pi * densDMConst * pow(rDMConst*Mpc,(2.0)) * (pi - 2.0*(1.0 + (rDMConst/radiusgpu)) * atan(radiusgpu/rDMConst) 
apps/rps.c:						+ 2.0*(1.0 + (rDMConst/radiusgpu)) * log(1.0 + (radiusgpu/rDMConst)) - (1-(rDMConst/radiusgpu)) * log(1.0 + pow((radiusgpu/rDMConst),(2.0))));
apps/rps.c:		Real radiusggpu = 0.000001;
apps/rps.c:		Epotdm = -pi * densDMConst * pow(rDMConst*Mpc,(2.0)) * (pi - 2.0*(1.0 + (rDMConst/radiusggpu))
apps/rps.c:																* atan(radiusggpu/rDMConst) + 2.0*(1.0 + (rDMConst/radiusggpu)) * log(1.0 + (radiusggpu/rDMConst)) - (1-(rDMConst/radiusggpu)) 
apps/rps.c:																* log(1.0 + pow((radiusggpu/rDMConst),(2.0))));

```
