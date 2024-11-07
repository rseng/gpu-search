# https://github.com/spacetelescope/mirage

```console
mirage/yaml/yaml_generator.py:                pupilkey = 'LongPupil'
mirage/yaml/yaml_generator.py:                    if ((self.info['LongPupil'][i]=='n/a' and self.info['detector'][i]=='A5') or
mirage/yaml/yaml_generator.py:                    pupilname = self.info['LongPupil'][i]
mirage/yaml/yaml_generator.py:                pupilkey = 'LongPupil'
mirage/apt/read_apt_xml.py:        FilterParams_keys = ['ShortFilter', 'LongFilter', 'ShortPupil', 'LongPupil',
mirage/apt/read_apt_xml.py:        dictionary['LongPupil'].append(tup[18])
mirage/apt/read_apt_xml.py:                        LongPupil, LongFilter = self.separate_pupil_and_filter(value)
mirage/apt/read_apt_xml.py:                        filter_config_dict['LongPupil'] = LongPupil
mirage/apt/read_apt_xml.py:                    if key not in ['ShortFilter', 'ShortPupil', 'LongFilter', 'LongPupil']:
mirage/apt/read_apt_xml.py:        long_pupil_filter = template.find(ns + 'LongPupilFilter').text
mirage/apt/read_apt_xml.py:        exposures_dictionary['LongPupil'] = ['CLEAR', long_pupil]
mirage/apt/read_apt_xml.py:        long_pupil = template.find(ns + 'LongPupil').text
mirage/apt/read_apt_xml.py:        exposures_dictionary['LongPupil'] = ['CLEAR', long_pupil]
mirage/apt/read_apt_xml.py:                    exp_seq_dict['LongPupil'] = [grism_long_pupil, direct_long_pupil]
mirage/apt/read_apt_xml.py:                    exp_seq_dict['LongPupil'] = [grism_long_pupil]
mirage/apt/read_apt_xml.py:            out_of_field_dict['LongPupil'] = [direct_long_pupil] * 2
mirage/apt/read_apt_xml.py:            ta_dict['LongPupil'] = long_pupil
mirage/apt/read_apt_xml.py:            astrometric_exp_dict['LongPupil'] = long_pupil
mirage/apt/read_apt_xml.py:                exposure_dict['LongPupil'] = long_pupil
mirage/apt/read_apt_xml.py:                exp_seq_dict['LongPupil'] = [long_pupil] * repeats
mirage/apt/apt_inputs.py:            long_pupils = np.array(pointing_info['LongPupil'])[good]
tests/test_data/NIRCam/NIRCamTest.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/NIRCam/1144-OTE-10.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/NIRCam/1144-OTE-10.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/54321/54321_niriss_wfss_prime_nircam_imaging_parallel.txt:PI_Name Proposal_category ProposalID Science_category Title Module Subarray Instrument PrimaryDitherType PrimaryDithers SubpixelPositions SubpixelDitherType CoordinatedParallel ParallelInstrument ObservationID TileNumber APTTemplate ApertureOverride ObservationName DitherPatternType ImageDithers number_of_dithers FiducialPointOverride ShortFilter LongFilter ShortPupil LongPupil ReadoutPattern Groups Integrations PupilWheel FilterWheel Mode Grism IntegrationsShort GroupsShort Dither GroupsLong ReadoutPatternShort IntegrationsLong Exposures Wavelength ReadoutPatternLong Filter EtcIdLong EtcIdShort EtcId
tests/test_data/misc/01142/OTE08-1142.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01142/OTE08-1142.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01148/OTE13-1148.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/12345/12345_nircam_imaging_prime_niriss_wfss_parallel.txt:PI_Name Proposal_category ProposalID Science_category Title Module Subarray Instrument PrimaryDitherType PrimaryDithers SubpixelPositions SubpixelDitherType CoordinatedParallel ParallelInstrument ObservationID TileNumber APTTemplate ApertureOverride ObservationName DitherPatternType ImageDithers number_of_dithers FiducialPointOverride ShortFilter LongFilter ShortPupil LongPupil ReadoutPattern Groups Integrations PupilWheel FilterWheel Mode Grism IntegrationsShort GroupsShort Dither GroupsLong ReadoutPatternShort IntegrationsLong Exposures Wavelength ReadoutPatternLong Filter EtcIdLong EtcIdShort EtcId
tests/test_data/misc/01144/OTE-10_1144.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01144/OTE-10_1144.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01144/OTE-10_1144.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01144/OTE-10_1144.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01144/OTE-10_1144.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01144/OTE-10_1144.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01144/OTE-10_1144.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01144/OTE-10_1144.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01141/OTE07-1141.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01141/OTE07-1141.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01141/OTE07-1141.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/01141/OTE07-1141.xml:                                <ncei:LongPupil>CLEAR</ncei:LongPupil>
tests/test_data/misc/08888/08888_niriss_nircam_wfss.txt:PI_Name Proposal_category ProposalID Science_category Title Module Subarray Instrument PrimaryDitherType PrimaryDithers SubpixelPositions SubpixelDitherType CoordinatedParallel ParallelInstrument ObservationID TileNumber APTTemplate ApertureOverride ObservationName DitherPatternType ImageDithers number_of_dithers FiducialPointOverride ShortFilter LongFilter ShortPupil LongPupil ReadoutPattern Groups Integrations PupilWheel FilterWheel Mode Grism IntegrationsShort GroupsShort Dither GroupsLong ReadoutPatternShort IntegrationsLong Exposures Wavelength ReadoutPatternLong Filter EtcIdLong EtcIdShort EtcId
tests/test_data/misc/01063/NCam010_pid1063v4.xml:                                <ncei:LongPupil>MASKRND</ncei:LongPupil>
tests/test_data/misc/01063/NCam010_pid1063v4.xml:                                <ncei:LongPupil>MASKBAR</ncei:LongPupil>
tests/test_data/misc/01063/NCam010_pid1063v4.xml:                                <ncei:LongPupil>MASKRND</ncei:LongPupil>
tests/test_data/misc/01063/NCam010_pid1063v4.xml:                                <ncei:LongPupil>MASKBAR</ncei:LongPupil>
tests/test_yaml_generator.py:    lw_pupilnames = np.array(yam.info['LongPupil'])
examples/tso_example_data/wasp-79_example_TSO.xml:                        <ncgts:LongPupilFilter>GRISMR+F444W</ncgts:LongPupilFilter>
examples/tso_example_data/wasp-79_example_TSO.xml:                        <ncgts:LongPupilFilter>GRISMR+F322W2</ncgts:LongPupilFilter>
examples/tso_example_data/wasp-79_example_TSO.xml:                        <ncts:LongPupil>F470N</ncts:LongPupil>

```
