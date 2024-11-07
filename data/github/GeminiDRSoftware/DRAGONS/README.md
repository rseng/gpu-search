# https://github.com/GeminiDRSoftware/DRAGONS

```console
geminidr/f2/primitives_f2_image.py:                                             dark=None, do_cal='procmode')
geminidr/f2/primitives_f2_spect.py:                                         dark=None, do_cal='procmode')
geminidr/ghost/primitives_calibdb_ghost.py:    #    procmode = 'sq' if self.mode == 'sq' else None
geminidr/ghost/primitives_calibdb_ghost.py:    #    cals = self.caldb.get_processed_arc(adinputs, procmode=procmode)
geminidr/ghost/primitives_calibdb_ghost.py:        procmode = 'sq' if self.mode == 'sq' else None
geminidr/ghost/primitives_calibdb_ghost.py:        cals = self.caldb.get_processed_slit(adinputs, procmode=procmode)
geminidr/ghost/primitives_calibdb_ghost.py:        procmode = 'sq' if self.mode == 'sq' else None
geminidr/ghost/primitives_calibdb_ghost.py:        cals = self.caldb.get_processed_slitflat(adinputs, procmode=procmode)
geminidr/core/primitives_preprocess.py:        do_cal: str [procmode|force|skip]
geminidr/core/tests/test_visualize.py:        do_cal_bias = 'skip' if master_bias is None else 'procmode'
geminidr/core/tests/test_visualize.py:        do_cal_flat = 'skip' if master_quartz is None else 'procmode'
geminidr/core/tests/test_calibdb.py:    saved_procmode = None
geminidr/core/tests/test_calibdb.py:    def mock_get_processed_arc(adinputs, procmode=None):
geminidr/core/tests/test_calibdb.py:        nonlocal saved_procmode
geminidr/core/tests/test_calibdb.py:        saved_procmode = procmode
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode == None)
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode == 'sq')
geminidr/core/tests/test_calibdb.py:    saved_procmode = None
geminidr/core/tests/test_calibdb.py:    def mock_get_processed_bias(adinputs, procmode=None):
geminidr/core/tests/test_calibdb.py:        nonlocal saved_procmode
geminidr/core/tests/test_calibdb.py:        saved_procmode = procmode
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode == None)
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode == 'sq')
geminidr/core/tests/test_calibdb.py:    saved_procmode = None
geminidr/core/tests/test_calibdb.py:    def mock_get_processed_dark(adinputs, procmode=None):
geminidr/core/tests/test_calibdb.py:        nonlocal saved_procmode
geminidr/core/tests/test_calibdb.py:        saved_procmode = procmode
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode == None)
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode == 'sq')
geminidr/core/tests/test_calibdb.py:    saved_procmode = None
geminidr/core/tests/test_calibdb.py:    def mock_get_processed_flat(adinputs, procmode=None):
geminidr/core/tests/test_calibdb.py:        nonlocal saved_procmode
geminidr/core/tests/test_calibdb.py:        saved_procmode = procmode
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode == None)
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode == 'sq')
geminidr/core/tests/test_calibdb.py:    saved_procmode = None
geminidr/core/tests/test_calibdb.py:    def mock_get_processed_fringe(adinputs, procmode=None):
geminidr/core/tests/test_calibdb.py:        nonlocal saved_procmode
geminidr/core/tests/test_calibdb.py:        saved_procmode = procmode
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode == None)
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode == 'sq')
geminidr/core/tests/test_calibdb.py:    saved_procmode = None
geminidr/core/tests/test_calibdb.py:    def mock_get_processed_standard(adinputs, procmode=None):
geminidr/core/tests/test_calibdb.py:        nonlocal saved_procmode
geminidr/core/tests/test_calibdb.py:        saved_procmode = procmode
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode == None)
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode == 'sq')
geminidr/core/tests/test_calibdb.py:    saved_procmode = None
geminidr/core/tests/test_calibdb.py:    def mock_get_processed_slitillum(adinputs, procmode=None):
geminidr/core/tests/test_calibdb.py:        nonlocal saved_procmode
geminidr/core/tests/test_calibdb.py:        saved_procmode = procmode
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode == None)
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode == 'sq')
geminidr/core/tests/test_calibdb.py:    saved_procmode = None
geminidr/core/tests/test_calibdb.py:    def mock_get_processed_bpm(adinputs, procmode=None):
geminidr/core/tests/test_calibdb.py:        nonlocal saved_procmode
geminidr/core/tests/test_calibdb.py:        saved_procmode = procmode
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode is None)
geminidr/core/tests/test_calibdb.py:    assert(saved_procmode == 'sq')
geminidr/core/tests/test_calibdb.py:    assert(saved_arc.phu['PROCMODE'] == 'ql')
geminidr/core/tests/test_calibdb.py:    assert(saved_arc.phu['PROCMODE'] == 'ql')
geminidr/core/tests/test_calibdb.py:    assert(saved_arc.phu['PROCMODE'] == 'ql')
geminidr/core/tests/test_calibdb.py:    assert(saved_arc.phu['PROCMODE'] == 'ql')
geminidr/core/tests/test_calibdb.py:    assert(saved_arc.phu['PROCMODE'] == 'ql')
geminidr/core/tests/test_calibdb.py:    assert(sci.phu['PROCMODE'] == 'ql')
geminidr/core/tests/test_calibdb.py:    assert(saved_arc.phu['PROCMODE'] == 'ql')
geminidr/core/tests/test_calibdb.py:    assert(saved_arc.phu['PROCMODE'] == 'ql')
geminidr/core/tests/test_calibdb.py:    assert(saved_bpm.phu['PROCMODE'] == 'ql')
geminidr/core/primitives_calibdb.py:        procmode = 'sq' if self.mode == 'sq' else None
geminidr/core/primitives_calibdb.py:        cals = self.caldb.get_processed_arc(adinputs, procmode=procmode)
geminidr/core/primitives_calibdb.py:        procmode = 'sq' if self.mode == 'sq' else None
geminidr/core/primitives_calibdb.py:        cals = self.caldb.get_processed_bias(adinputs, procmode=procmode)
geminidr/core/primitives_calibdb.py:        procmode = 'sq' if self.mode == 'sq' else None
geminidr/core/primitives_calibdb.py:        cals = self.caldb.get_processed_dark(adinputs, procmode=procmode)
geminidr/core/primitives_calibdb.py:        procmode = 'sq' if self.mode == 'sq' else None
geminidr/core/primitives_calibdb.py:        cals = self.caldb.get_processed_flat(adinputs, procmode=procmode)
geminidr/core/primitives_calibdb.py:        procmode = 'sq' if self.mode == 'sq' else None
geminidr/core/primitives_calibdb.py:        cals = self.caldb.get_processed_fringe(adinputs, procmode=procmode)
geminidr/core/primitives_calibdb.py:        procmode = 'sq' if self.mode == 'sq' else None
geminidr/core/primitives_calibdb.py:        cals = self.caldb.get_processed_pinhole(adinputs, procmode=procmode)
geminidr/core/primitives_calibdb.py:        procmode = 'sq' if self.mode == 'sq' else None
geminidr/core/primitives_calibdb.py:        cals = self.caldb.get_processed_standard(adinputs, procmode=procmode)
geminidr/core/primitives_calibdb.py:        procmode = 'sq' if self.mode == 'sq' else None
geminidr/core/primitives_calibdb.py:        cals = self.caldb.get_processed_slitillum(adinputs, procmode=procmode)
geminidr/core/primitives_calibdb.py:        procmode = 'sq' if self.mode == 'sq' else None
geminidr/core/primitives_calibdb.py:        cals = self.caldb.get_processed_bpm(adinputs, procmode=procmode)
geminidr/core/primitives_calibdb.py:            if 'PROCMODE' not in ad.phu:
geminidr/core/primitives_calibdb.py:                ad.phu.set('PROCMODE', self.mode)
geminidr/core/primitives_calibdb.py:            mode = ad.phu['PROCMODE']
geminidr/core/primitives_calibdb.py:                # the update_filename method, keeps the procmode suffix
geminidr/core/primitives_calibdb.py:            # if store has already been run and PROCMODE set, do not let
geminidr/core/primitives_calibdb.py:            # a subsequent call to store change the PROCMODE.  Eg. a subsequent
geminidr/core/primitives_calibdb.py:            # procmode that was used when the data was reduced.
geminidr/core/primitives_calibdb.py:            if 'PROCMODE' not in ad.phu:
geminidr/core/primitives_calibdb.py:                ad.phu.set('PROCMODE', self.mode)
geminidr/core/primitives_calibdb.py:            mode = ad.phu['PROCMODE']
geminidr/core/primitives_spect.py:        do_cal: str [procmode|force|skip]
geminidr/core/primitives_image.py:            if do_cal == 'procmode' or do_cal == 'skip':
geminidr/core/primitives_image.py:            if do_cal == 'procmode' and not correct:
geminidr/core/primitives_telluric.py:        do_cal: str ["procmode" | "force" | "skip"]
geminidr/core/parameters_generic.py:                    allowed={"procmode": "Use the default rules set by the processing"
geminidr/core/parameters_generic.py:                    default="procmode",
geminidr/interactive/server.py:        #         "--disable-gpu",
geminidr/gmos/primitives_gmos_longslit.py:            Perform slit illumination correction? (Default: 'procmode')
geminidr/gmos/tests/spect/test_calculate_sensitivity.py:        p.biasCorrect(bias=bias_master, do_cal='procmode')
geminidr/gmos/primitives_gmos_nodandshuffle.py:        do_cal: str ("force", "skip", "procmode")
geminidr/__init__.py:            procmode=self.mode)
recipe_system/cal_service/userdb.py:                         procmode=None)
recipe_system/cal_service/userdb.py:    def _get_calibrations(self, adinputs, caltype=None, procmode=None,
recipe_system/cal_service/localmanager.py:                                 descriptors=descripts, types=types, procmode=rq.procmode)
recipe_system/cal_service/localdb.py:    def __init__(self, dbfile, name=None, valid_caltypes=None, procmode=None,
recipe_system/cal_service/localdb.py:                         procmode=procmode)
recipe_system/cal_service/localdb.py:    def _get_calibrations(self, adinputs, caltype=None, procmode=None,
recipe_system/cal_service/localdb.py:        cal_requests = get_cal_requests(adinputs, caltype, procmode=procmode,
recipe_system/cal_service/caldb.py:    procmode : str
recipe_system/cal_service/caldb.py:                 valid_caltypes=None, procmode=None, log=None):
recipe_system/cal_service/caldb.py:        self.procmode = procmode
recipe_system/cal_service/caldb.py:    def get_calibrations(self, adinputs, caltype=None, procmode=None,
recipe_system/cal_service/caldb.py:        procmode : str/None
recipe_system/cal_service/caldb.py:        if procmode is None:
recipe_system/cal_service/caldb.py:            procmode = self.procmode
recipe_system/cal_service/caldb.py:                                                 procmode=procmode, howmany=howmany)
recipe_system/cal_service/caldb.py:                caltype=caltype, procmode=procmode)
recipe_system/cal_service/caldb.py:    def _get_calibrations(self, adinputs, caltype=None, procmode=None):
recipe_system/cal_service/remotedb.py:                 store_cal=False, store_science=False, procmode=None, log=None,
recipe_system/cal_service/remotedb.py:                         procmode=procmode)
recipe_system/cal_service/remotedb.py:    def _get_calibrations(self, adinputs, caltype=None, procmode=None,
recipe_system/cal_service/remotedb.py:        cal_requests = get_cal_requests(adinputs, caltype, procmode=procmode,
recipe_system/cal_service/remotedb.py:            procstr = "" if procmode is None else f"/{procmode}"
recipe_system/cal_service/__init__.py:def init_calibration_databases(inst_lookups=None, procmode=None,
recipe_system/cal_service/__init__.py:        kwargs["procmode"] = procmode
recipe_system/cal_service/calrequestlib.py:    def __init__(self, ad, caltype=None, procmode=None):
recipe_system/cal_service/calrequestlib.py:        self.procmode = procmode
recipe_system/cal_service/calrequestlib.py:             'procmode'   : self.procmode,
recipe_system/cal_service/calrequestlib.py:def get_cal_requests(inputs, caltype, procmode=None, is_local=True):
recipe_system/cal_service/calrequestlib.py:        rq = CalibrationRequest(ad, caltype, procmode)

```
