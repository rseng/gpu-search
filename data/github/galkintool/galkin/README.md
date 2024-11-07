# https://github.com/galkintool/galkin

```console
galkin/readparsFile.py:    flagGMC=-10000;flagHou09tabA2=-10000;flagOPENCLUSTERS=-10000;flagFrinchaboyMajewski08=-10000;flagPLANETARYNEBULAE=-10000;flagDurand98=-10000
galkin/readparsFile.py:	if(values[0]=='flagOPENCLUSTERS'): flagOPENCLUSTERS=int(values[1])
galkin/readparsFile.py:    return (R0,V0,UsunINUSE,VsunINUSE,WsunINUSE,SYSTDISP, flagPROPERMOTIONS,flagHITERMINAL,flagFich89tab2,flagMalhotra95,flagMcClureGriffithsDickey07, flagHITHICKNESS,flagHonmaSofue97,flagCOTERMINAL,flagBurtonGordon78,flagClemens85,flagKnapp85,flagLuna06, flagHIIREGIONS,flagBlitz79,flagFich89tab1,flagTurbideMoffat93,flagBrandBlitz93,flagHou09tabA1, flagGMC,flagHou09tabA2,flagOPENCLUSTERS,flagFrinchaboyMajewski08,flagPLANETARYNEBULAE,flagDurand98,flagCEPHEIDS,flagPont94,flagPont97, flagCSTARS,flagDemersBattinelli07,flagBattinelli12, flagMASERS,flagReid14,flagHonma12,flagStepanishchevBobylev11,flagXu13,flagBobylevBajkova13,flagastropy);
galkin/readparsFile.py:    flagGMC=-10000;flagHou09tabA2=-10000;flagOPENCLUSTERS=-10000;flagFrinchaboyMajewski08=-10000;flagPLANETARYNEBULAE=-10000;flagDurand98=-10000
galkin/readparsFile.py:    flagOPENCLUSTERS=int(Config.get("Flags","flagOPENCLUSTERS"))
galkin/readparsFile.py:    return (R0,V0,UsunINUSE,VsunINUSE,WsunINUSE,SYSTDISP, flagPROPERMOTIONS,flagHITERMINAL,flagFich89tab2,flagMalhotra95,flagMcClureGriffithsDickey07, flagHITHICKNESS,flagHonmaSofue97,flagCOTERMINAL,flagBurtonGordon78,flagClemens85,flagKnapp85,flagLuna06, flagHIIREGIONS,flagBlitz79,flagFich89tab1,flagTurbideMoffat93,flagBrandBlitz93,flagHou09tabA1, flagGMC,flagHou09tabA2,flagOPENCLUSTERS,flagFrinchaboyMajewski08,flagPLANETARYNEBULAE,flagDurand98,flagCEPHEIDS,flagPont94,flagPont97, flagCSTARS,flagDemersBattinelli07,flagBattinelli12, flagMASERS,flagReid14,flagHonma12,flagStepanishchevBobylev11,flagXu13,flagBobylevBajkova13,flagastropy);
galkin/readparsFile.py:    flagGMC=pars[24];flagHou09tabA2=pars[25];flagOPENCLUSTERS=pars[26];flagFrinchaboyMajewski08=pars[27];flagPLANETARYNEBULAE=pars[28];flagDurand98=pars[29]
galkin/readparsFile.py:    if flagOPENCLUSTERS!=0 and flagOPENCLUSTERS!=1:
galkin/readparsFile.py:       print '*** error: wrong open clusters flag: ', flagOPENCLUSTERS,'. exiting now.';sys.exit()
galkin/readparsFile.py:    if flagOPENCLUSTERS==0 and flagFrinchaboyMajewski08==1:
galkin/readparsFile.py:    print ' use open clusters?                    ', flagOPENCLUSTERS
galkin/readparsFile.py:    if flagOPENCLUSTERS==1:
bin/inputpars.txt:#  - flagOPENCLUSTERS:			whether to use open clusters
bin/inputpars.txt:flagOPENCLUSTERS: 1
bin/galkin_data_fast.py:flagOPENCLUSTERS=1				# whether to use open clusters
bin/galkin_data_fast.py:inputpars=(R0,V0,UsunINUSE,VsunINUSE,WsunINUSE,SYSTDISP, flagPROPERMOTIONS,flagHITERMINAL,flagFich89tab2,flagMalhotra95,flagMcClureGriffithsDickey07, flagHITHICKNESS,flagHonmaSofue97,flagCOTERMINAL,flagBurtonGordon78,flagClemens85,flagKnapp85,flagLuna06, flagHIIREGIONS,flagBlitz79,flagFich89tab1,flagTurbideMoffat93,flagBrandBlitz93,flagHou09tabA1, flagGMC,flagHou09tabA2,flagOPENCLUSTERS,flagFrinchaboyMajewski08,flagPLANETARYNEBULAE,flagDurand98,flagCEPHEIDS,flagPont94,flagPont97, flagCSTARS,flagDemersBattinelli07,flagBattinelli12, flagMASERS,flagReid14,flagHonma12,flagStepanishchevBobylev11,flagXu13,flagBobylevBajkova13,flagastropy)

```
