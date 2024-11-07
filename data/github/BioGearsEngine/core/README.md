# https://github.com/BioGearsEngine/core

```console
share/xsd/cdm/Circuit.xsd:          <xs:element name="Switch" type="enumOpenClosed" minOccurs="0"  maxOccurs="1"/>
share/xsd/cdm/Circuit.xsd:          <xs:element name="NextSwitch" type="enumOpenClosed" minOccurs="0"  maxOccurs="1"/>
share/xsd/cdm/Circuit.xsd:          <xs:element name="Valve" type="enumOpenClosed" minOccurs="0"  maxOccurs="1"/>
share/xsd/cdm/Circuit.xsd:          <xs:element name="NextValve" type="enumOpenClosed" minOccurs="0"  maxOccurs="1"/>
share/xsd/cdm/Circuit.xsd:          <xs:element name="PolarizedState" type="enumOpenClosed" minOccurs="0"  maxOccurs="1"/>
share/xsd/cdm/Circuit.xsd:          <xs:element name="NextPolarizedState" type="enumOpenClosed" minOccurs="0"  maxOccurs="1"/>
share/xsd/cdm/Properties.xsd:  <xs:simpleType name="enumOpenClosed">
share/proto/properties.proto:enum EnumOpenClosed {
share/data/config/PlotRun-1_of_2.config:MainstemIntubation=ActionEventPlotter NoGrid Header=RightLungPulmonary-Volume(mL) VerificationDir=Patient OutputOverride=doc/doxygen/html/plots/Respiratory/ NoEvents RemoveLegends OutputFilename=MainstemIntubation_RightLungVolume.jpg
share/data/config/PlotRun-1_of_2.config:MainstemIntubation=ActionEventPlotter NoGrid Header=LeftLungPulmonary-Volume(mL) VerificationDir=Patient OutputOverride=doc/doxygen/html/plots/Respiratory/ NoEvents RemoveLegends OutputFilename=MainstemIntubation_LeftLungVolume.jpg
share/data/config/PlotRun-1_of_2.config:TensionPneumothoraxOpenVaried=ActionEventPlotter NoGrid Header=LeftLungPulmonary-Volume(mL) VerificationDir=Patient OutputOverride=doc/doxygen/html/plots/Respiratory/ NoEvents RemoveLegends OutputFilename=TensionPneumothoraxOpenVaried_LeftLungVolume.jpg
share/data/config/PlotRun-1_of_2.config:TensionPneumothoraxClosedVaried=ActionEventPlotter NoGrid Header=RightLungPulmonary-Volume(mL) VerificationDir=Patient OutputOverride=doc/doxygen/html/plots/Respiratory/ NoEvents RemoveLegends OutputFilename=TensionPneumothoraxClosedVaried_RightLungVolume.jpg
share/data/states/Male_22_Fit_Soldier@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_22_Fit_Soldier@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_22_Fit_Soldier@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_22_Fit_Soldier@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_22_Fit_Soldier@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_22_Fit_Soldier@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_22_Fit_Soldier@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_22_Fit_Soldier@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_44_Bradycardic@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_44_Bradycardic@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_44_Bradycardic@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_44_Bradycardic@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_44_Bradycardic@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_44_Bradycardic@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_44_Bradycardic@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_44_Bradycardic@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Female_40_Overweight@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Female_40_Overweight@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Female_40_Overweight@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Female_40_Overweight@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Female_40_Overweight@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Female_40_Overweight@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Female_40_Overweight@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Female_40_Overweight@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Female_18_Normal@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Female_18_Normal@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Female_18_Normal@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Female_18_Normal@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Female_18_Normal@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Female_18_Normal@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Female_18_Normal@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Female_18_Normal@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/DefaultTemplateFemale@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/DefaultTemplateFemale@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/DefaultTemplateFemale@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/DefaultTemplateFemale@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/DefaultTemplateFemale@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/DefaultTemplateFemale@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/DefaultTemplateFemale@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/DefaultTemplateFemale@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_24_Normal_hidrosis2@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_24_Normal_hidrosis2@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_24_Normal_hidrosis2@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_24_Normal_hidrosis2@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_24_Normal_hidrosis2@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_24_Normal_hidrosis2@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_24_Normal_hidrosis2@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_24_Normal_hidrosis2@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Female@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Female@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Female@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Female@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Female@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Female@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Female@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Female@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/StandardMale@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/StandardMale@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/StandardMale@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/StandardMale@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/StandardMale@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/StandardMale@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/StandardMale@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/StandardMale@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_44_Normal_rr12@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_44_Normal_rr12@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_44_Normal_rr12@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_44_Normal_rr12@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_44_Normal_rr12@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_44_Normal_rr12@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_44_Normal_rr12@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_44_Normal_rr12@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_44_SleepDeprived@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_44_SleepDeprived@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_44_SleepDeprived@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_44_SleepDeprived@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_44_SleepDeprived@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_44_SleepDeprived@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_44_SleepDeprived@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_44_SleepDeprived@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_28_Normal_hr109_rr18@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_28_Normal_hr109_rr18@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_28_Normal_hr109_rr18@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_28_Normal_hr109_rr18@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_28_Normal_hr109_rr18@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_28_Normal_hr109_rr18@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_28_Normal_hr109_rr18@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_28_Normal_hr109_rr18@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_32_Normal_hr93_rr14@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_32_Normal_hr93_rr14@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_32_Normal_hr93_rr14@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_32_Normal_hr93_rr14@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_32_Normal_hr93_rr14@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_32_Normal_hr93_rr14@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_32_Normal_hr93_rr14@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_32_Normal_hr93_rr14@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_44_Normal_hr109_rr15@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_44_Normal_hr109_rr15@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_44_Normal_hr109_rr15@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_44_Normal_hr109_rr15@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_44_Normal_hr109_rr15@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_44_Normal_hr109_rr15@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_44_Normal_hr109_rr15@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_44_Normal_hr109_rr15@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/StandardFemale@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/StandardFemale@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/StandardFemale@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/StandardFemale@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/StandardFemale@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/StandardFemale@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/StandardFemale@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/StandardFemale@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_25_Normal@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_25_Normal@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_25_Normal@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_25_Normal@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_25_Normal@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_25_Normal@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_25_Normal@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_25_Normal@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Female_30_Normal@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Female_30_Normal@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Female_30_Normal@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Female_30_Normal@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Female_30_Normal@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Female_30_Normal@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Female_30_Normal@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Female_30_Normal@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_44_Tachycardic@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_44_Tachycardic@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_44_Tachycardic@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_44_Tachycardic@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/Male_44_Tachycardic@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/Male_44_Tachycardic@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/Male_44_Tachycardic@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/Male_44_Tachycardic@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/DefaultTemplateMale@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/DefaultTemplateMale@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/DefaultTemplateMale@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/DefaultTemplateMale@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/states/DefaultTemplateMale@0s.xml:      <Name>LeftLungPulmonary</Name>
share/data/states/DefaultTemplateMale@0s.xml:      <Child>LeftLungPulmonary</Child>
share/data/states/DefaultTemplateMale@0s.xml:      <Child>RightLungPulmonary</Child>
share/data/states/DefaultTemplateMale@0s.xml:      <Name>RightLungPulmonary</Name>
share/data/validation/RespiratoryValidationResults.csv:Time(s),AorticWallStrain,Afferent_AorticBaroreceptor,PleuralPressure,Test_FilteredCVP,Test_CentralSignal,MeanLungVolume,MuscleAutoregulation,Afferent_Baroreceptor,Afferent_Chemoreceptor,Afferent_PulmonaryStretch,Ursino_SympatheticNode,Ursino_SympatheticPeripheral,Afferent_Strain,Ursino_Parasympathetic,HypoxiaThreshold_Heart,HypoxiaThreshold_Peripheral,HypocapniaThreshold_Heart,HypocapniaThreshold_Peripheral,BaroreceptorOperatingPoint,SympatheticFatigue,HeartRateMod_Sympathetic,HeartRateMod_Vagal,ResistanceScale_Muscle,ResistanceScale_Splanchnic,ResistanceScale_Extrasplanchnic,ResistanceScale_Myocardium,Resistance_TotalPeripheral(mmHg_s_Per_mL),ElastanceScale,ComplianceScale,ComplianceModifier,VenaCava_Volume(mL),AlveolarArterialGradient(mmHg),CarricoIndex(mmHg),EndTidalCarbonDioxideFraction,EndTidalCarbonDioxidePressure(cmH2O),ExpiratoryFlow(L/s),InspiratoryExpiratoryRatio,InspiratoryFlow(L/s),PulmonaryCompliance(L/cmH2O),PulmonaryResistance(cmH2O_s/L),RespirationDriverPressure(cmH2O),RespirationMusclePressure(cmH2O),RespirationRate(1/min),SpecificVentilation,TidalVolume(mL),TotalAlveolarVentilation(L/min),TotalDeadSpaceVentilation(L/min),TotalLungVolume(L),TotalPulmonaryVentilation(L/min),TranspulmonaryPressure(cmH2O),PatientWeight(g),Trachea-Pressure(cmH2O),Trachea-Oxygen-PartialPressure(mmHg),Trachea-CarbonDioxide-PartialPressure(mmHg),LeftPleuralCavity-Pressure(cmH2O),RightPleuralCavity-Pressure(cmH2O),PulmonaryLungs-Volume(L),LeftLungPulmonary-Pressure(cmH2O),RightLungPulmonary-Pressure(cmH2O),LeftAlveoli-Volume(L),LeftAlveoli-Pressure(cmH2O),LeftAlveoli-Oxygen-PartialPressure(mmHg),LeftAlveoli-CarbonDioxide-PartialPressure(mmHg),RightAlveoli-Volume(L),RightAlveoli-Pressure(cmH2O),RightAlveoli-Oxygen-PartialPressure(mmHg),RightAlveoli-CarbonDioxide-PartialPressure(mmHg),LeftBronchi-Volume(mL),LeftBronchi-Oxygen-PartialPressure(mmHg),LeftBronchi-CarbonDioxide-PartialPressure(mmHg),RightBronchi-Volume(mL),RightBronchi-Oxygen-PartialPressure(mmHg),RightBronchi-CarbonDioxide-PartialPressure(mmHg)
share/data/validation/RespiratoryValidation.csv:LeftLungPulmonary-Pressure,cmH2O,"[1034.0, 1038.2],
share/data/validation/RespiratoryValidation.csv:LeftLungPulmonary-Pressure,cmH2O,"[1028.2, 1032.4],
share/data/validation/RespiratoryValidation.csv:RightLungPulmonary-Pressure,cmH2O,"[1034.0, 1038.2],
share/data/validation/RespiratoryValidation.csv:RightLungPulmonary-Pressure,cmH2O,"[1028.2, 1032.4],
share/doc/methodology/RespiratoryMethodology.md:	* RightLung, RightLungPulmonary
share/Scenarios/Patient/EsophagealIntubation.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary"  Name="Volume" Unit="mL" Precision="0"/>
share/Scenarios/Patient/EsophagealIntubation.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Name="Volume" Unit="mL" Precision="0"/>
share/Scenarios/Patient/MainstemIntubation.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary"  Name="Volume" Unit="mL" Precision="0"/>
share/Scenarios/Patient/MainstemIntubation.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Name="Volume" Unit="mL" Precision="0"/>
share/Scenarios/Patient/TensionPneumothoraxBilateral.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary"                            Name="Volume"          Unit="mL"   Precision="0"/>	
share/Scenarios/Patient/TensionPneumothoraxBilateral.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary"  Substance="Oxygen"        Name="PartialPressure" Unit="mmHg" Precision="1"/>
share/Scenarios/Patient/TensionPneumothoraxBilateral.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary"  Substance="CarbonDioxide" Name="PartialPressure" Unit="mmHg" Precision="1"/>	
share/Scenarios/Patient/TensionPneumothoraxBilateral.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary"                           Name="Volume"          Unit="mL"   Precision="0"/>	
share/Scenarios/Patient/TensionPneumothoraxBilateral.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Substance="Oxygen"        Name="PartialPressure" Unit="mmHg" Precision="1"/>
share/Scenarios/Patient/TensionPneumothoraxBilateral.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Substance="CarbonDioxide" Name="PartialPressure" Unit="mmHg" Precision="1"/>
share/Scenarios/Patient/TensionPneumothoraxOpenVaried.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary"                            Name="Volume"          Unit="mL"   Precision="0"/>	
share/Scenarios/Patient/TensionPneumothoraxOpenVaried.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary"  Substance="Oxygen"        Name="PartialPressure" Unit="mmHg" Precision="1"/>
share/Scenarios/Patient/TensionPneumothoraxOpenVaried.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary"  Substance="CarbonDioxide" Name="PartialPressure" Unit="mmHg" Precision="1"/>	
share/Scenarios/Patient/TensionPneumothoraxOpenVaried.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary"                           Name="Volume"          Unit="mL"   Precision="0"/>	
share/Scenarios/Patient/TensionPneumothoraxOpenVaried.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Substance="Oxygen"        Name="PartialPressure" Unit="mmHg" Precision="1"/>
share/Scenarios/Patient/TensionPneumothoraxOpenVaried.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Substance="CarbonDioxide" Name="PartialPressure" Unit="mmHg" Precision="1"/>
share/Scenarios/Patient/TensionPneumothoraxClosedVaried.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary"                            Name="Volume"          Unit="mL"   Precision="0"/>	
share/Scenarios/Patient/TensionPneumothoraxClosedVaried.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary"  Substance="Oxygen"        Name="PartialPressure" Unit="mmHg" Precision="1"/>
share/Scenarios/Patient/TensionPneumothoraxClosedVaried.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary"  Substance="CarbonDioxide" Name="PartialPressure" Unit="mmHg" Precision="1"/>	
share/Scenarios/Patient/TensionPneumothoraxClosedVaried.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary"                           Name="Volume"          Unit="mL"   Precision="0"/>	
share/Scenarios/Patient/TensionPneumothoraxClosedVaried.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Substance="Oxygen"        Name="PartialPressure" Unit="mmHg" Precision="1"/>
share/Scenarios/Patient/TensionPneumothoraxClosedVaried.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Substance="CarbonDioxide" Name="PartialPressure" Unit="mmHg" Precision="1"/>
share/Scenarios/Validation/RespiratoryValidation.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary" Name="Pressure" Unit="cmH2O" Precision="4"/>
share/Scenarios/Validation/RespiratoryValidation.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Name="Pressure" Unit="cmH2O" Precision="4"/>	
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<!-- LeftLungPulmonary -->	
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary" Name="InFlow" Unit="mL/s" Precision="0"/>	
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary" Name="OutFlow" Unit="mL/s" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary" Name="Pressure"  Unit="mmHg" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary" Name="Volume"  Unit="mL" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary" Substance="Oxygen" Name="Volume"  Unit="mL" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary" Substance="Oxygen" Name="VolumeFraction"  Unit="" Precision="3"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary" Substance="Oxygen" Name="PartialPressure"  Unit="mmHg" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary" Substance="CarbonDioxide" Name="Volume"  Unit="mL" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary" Substance="CarbonDioxide" Name="VolumeFraction"  Unit="" Precision="4"/>	
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary" Substance="CarbonDioxide" Name="PartialPressure"  Unit="mmHg" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary" Substance="Nitrogen" Name="Volume"  Unit="mL" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary" Substance="Nitrogen" Name="VolumeFraction"  Unit="" Precision="3"/>	
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="LeftLungPulmonary" Substance="Nitrogen" Name="PartialPressure"  Unit="mmHg" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<!-- RightLungPulmonary -->	
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Name="InFlow" Unit="mL/s" Precision="0"/>	
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Name="OutFlow" Unit="mL/s" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Name="Pressure"  Unit="mmHg" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Name="Volume"  Unit="mL" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Substance="Oxygen" Name="Volume"  Unit="mL" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Substance="Oxygen" Name="VolumeFraction"  Unit="" Precision="3"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Substance="Oxygen" Name="PartialPressure"  Unit="mmHg" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Substance="CarbonDioxide" Name="Volume"  Unit="mL" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Substance="CarbonDioxide" Name="VolumeFraction"  Unit="" Precision="4"/>	
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Substance="CarbonDioxide" Name="PartialPressure"  Unit="mmHg" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Substance="Nitrogen" Name="Volume"  Unit="mL" Precision="0"/>
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Substance="Nitrogen" Name="VolumeFraction"  Unit="" Precision="3"/>	
share/Scenarios/Compartments/PulmonaryCompartments.xml:	<DataRequest xsi:type="GasCompartmentDataRequestData" Compartment="RightLungPulmonary" Substance="Nitrogen" Name="PartialPressure"  Unit="mmHg" Precision="0"/>
share/website/validation/RespiratoryCompartmentsValidationTable.md:|LeftLungPulmonary-Pressure(cmH2O)                      |[1034.0, 1038.2] @cite otis1947measurement                                |Maximum of 1033.7     |<span class="success">-0%</span>               |Assume  P<sub>atm</sub> = 1033.2 cmH2O     |
share/website/validation/RespiratoryCompartmentsValidationTable.md:|LeftLungPulmonary-Pressure(cmH2O)                      |[1028.2, 1032.4] @cite otis1947measurement                                |Minimum of 1032.8     |<span class="success">0%</span>                |Assume  P<sub>atm</sub> = 1033.2 cmH2O     |
share/website/validation/RespiratoryCompartmentsValidationTable.md:|RightLungPulmonary-Pressure(cmH2O)                     |[1034.0, 1038.2] @cite otis1947measurement                                |Maximum of 1033.7     |<span class="success">-0%</span>               |Assume  P<sub>atm</sub> = 1033.2 cmH2O     |
share/website/validation/RespiratoryCompartmentsValidationTable.md:|RightLungPulmonary-Pressure(cmH2O)                     |[1028.2, 1032.4] @cite otis1947measurement                                |Minimum of 1032.8     |<span class="success">0%</span>                |Assume  P<sub>atm</sub> = 1033.2 cmH2O     |
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Open);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Closed);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Open);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Closed);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Open);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Closed);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Open);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Closed);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Open);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Closed);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Open);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Closed);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Open);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Closed);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Open);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:        m_Circuits->GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Closed);
projects/circuit_profiler/src/circuit_tester/circuit_tester.h:  Path4.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/AdvancedCircuitTest.cpp:  Path2.SetNextPolarizedState(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path2.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path2.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path2.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path3.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path3.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path3.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path4.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path4.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path4.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp://  Path3.SetSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path3.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path6.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path3.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path6.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path3.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path6.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path4.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path4.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path4.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path2.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path2.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path2.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path2.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path2.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path2.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path2.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path4.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path2.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path4.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path2.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path4.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path2.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path4.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path4.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path4.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path4.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:  Path4.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path2")->SetNextSwitch(SEOpenClosed::Closed);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libCircuitTest/src/Circuits/BasicCircuitTest.cpp:        m_Circuits.GetFluidPath("Path3")->SetNextSwitch(SEOpenClosed::Closed);
projects/biogears/libBiogears/unit/CircuitTest/test_circuit_Gastrointestinal.cpp:  ActiveToClothing.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_Switch = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_Valve = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_NextSwitch = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_NextValve = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_NextPolarizedState = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_PolarizedState = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_Switch = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_Valve = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_NextSwitch = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_NextValve = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_NextPolarizedState = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_PolarizedState = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:SEOpenClosed SECircuitPath<CIRCUIT_PATH_TYPES>::GetSwitch() const
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:void SECircuitPath<CIRCUIT_PATH_TYPES>::SetSwitch(SEOpenClosed state)
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_Switch = (m_Switch == SEOpenClosed::Open) ? SEOpenClosed::Closed : SEOpenClosed::Open;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  return m_Switch == (SEOpenClosed)-1 ? false : true;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_Switch = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:SEOpenClosed SECircuitPath<CIRCUIT_PATH_TYPES>::GetNextSwitch() const
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:void SECircuitPath<CIRCUIT_PATH_TYPES>::SetNextSwitch(SEOpenClosed state)
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_NextSwitch = (m_NextSwitch == SEOpenClosed::Open) ? SEOpenClosed::Closed : SEOpenClosed::Open;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  return m_NextSwitch == (SEOpenClosed)-1 ? false : true;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_NextSwitch = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:SEOpenClosed SECircuitPath<CIRCUIT_PATH_TYPES>::GetValve() const
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:void SECircuitPath<CIRCUIT_PATH_TYPES>::SetValve(SEOpenClosed state)
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_Valve = (m_Valve == SEOpenClosed::Open) ? SEOpenClosed::Closed : SEOpenClosed::Open;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  return m_Valve == (SEOpenClosed)-1 ? false : true;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_Valve = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:SEOpenClosed SECircuitPath<CIRCUIT_PATH_TYPES>::GetNextValve() const
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:void SECircuitPath<CIRCUIT_PATH_TYPES>::SetNextValve(SEOpenClosed state)
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_NextValve = (m_NextValve == SEOpenClosed::Open) ? SEOpenClosed::Closed : SEOpenClosed::Open;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  return m_NextValve == (SEOpenClosed)-1 ? false : true;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_NextValve = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:SEOpenClosed SECircuitPath<CIRCUIT_PATH_TYPES>::GetNextPolarizedState() const
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:void SECircuitPath<CIRCUIT_PATH_TYPES>::SetNextPolarizedState(SEOpenClosed state)
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_NextPolarizedState = (m_NextPolarizedState == SEOpenClosed::Open) ? SEOpenClosed::Closed : SEOpenClosed::Open;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  return m_NextPolarizedState == (SEOpenClosed)-1 ? false : true;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_NextPolarizedState = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:SEOpenClosed SECircuitPath<CIRCUIT_PATH_TYPES>::GetPolarizedState() const
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:void SECircuitPath<CIRCUIT_PATH_TYPES>::SetPolarizedState(SEOpenClosed state)
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_PolarizedState = (m_PolarizedState == SEOpenClosed::Open) ? SEOpenClosed::Closed : SEOpenClosed::Open;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  return m_PolarizedState == (SEOpenClosed)-1 ? false : true;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.inl:  m_PolarizedState = (SEOpenClosed)-1;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitCalculator.inl:      p->SetNextPolarizedState(SEOpenClosed::Closed);
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitCalculator.inl:    if (p->HasNextPotentialSource() || (p->NumberOfNextElements() < 1) || (p->HasNextValve() && p->GetNextValve() == SEOpenClosed::Closed) || (p->HasNextSwitch() && p->GetNextSwitch() == SEOpenClosed::Closed)) {
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitCalculator.inl:      if (p->HasNextPolarizedState() && p->GetNextPolarizedState() == SEOpenClosed::Open) { //Polarized elements that are open are done exactly the same as a open switch.
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitCalculator.inl:        if (p->GetNextSwitch() == SEOpenClosed::Open) {
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitCalculator.inl:        if (p->GetNextValve() == SEOpenClosed::Open) {
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitCalculator.inl:    } else if ((p->HasNextSwitch() && p->GetNextSwitch() == SEOpenClosed::Open) || (p->HasNextValve() && p->GetNextValve() == SEOpenClosed::Open) || (p->HasNextPolarizedState() && p->GetNextPolarizedState() == SEOpenClosed::Open)) {
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitCalculator.inl:        if (p->GetPolarizedState() == SEOpenClosed::Open) {
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitCalculator.inl:        if (p->GetNextPolarizedState() == SEOpenClosed::Open) {
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitCalculator.inl:    if ((p->GetNextValve() == SEOpenClosed::Closed && p->GetNextFlux().GetValue(m_FluxUnit) < -ZERO_APPROX)
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitCalculator.inl:      || (p->GetNextValve() == SEOpenClosed::Open && (p->GetSourceNode().GetNextPotential().GetValue(m_PotentialUnit) - p->GetTargetNode().GetNextPotential().GetValue(m_PotentialUnit)) > ZERO_APPROX)) {
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitCalculator.inl:    if (p->GetNextPolarizedState() == SEOpenClosed::Closed && (p->GetSourceNode().GetNextPotential().GetValue(m_PotentialUnit) - p->GetTargetNode().GetNextPotential().GetValue(m_PotentialUnit)) < -ZERO_APPROX) {
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitCalculator.inl:    if (pValve->GetNextValve() == SEOpenClosed::Closed)
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitCalculator.inl:    if (pPolarizedElement->GetNextPolarizedState() == SEOpenClosed::Closed)
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  virtual SEOpenClosed GetSwitch() const;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  virtual void SetSwitch(SEOpenClosed state);
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  virtual SEOpenClosed GetNextSwitch() const;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  virtual void SetNextSwitch(SEOpenClosed state);
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  virtual SEOpenClosed GetValve() const;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  virtual void SetValve(SEOpenClosed state);
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  virtual SEOpenClosed GetNextValve() const;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  virtual void SetNextValve(SEOpenClosed state);
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  virtual SEOpenClosed GetPolarizedState() const;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  virtual void SetPolarizedState(SEOpenClosed state);
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  virtual SEOpenClosed GetNextPolarizedState() const;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  virtual void SetNextPolarizedState(SEOpenClosed state);
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  SEOpenClosed m_Switch;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  SEOpenClosed m_NextSwitch;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  SEOpenClosed m_Valve;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  SEOpenClosed m_NextValve;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  SEOpenClosed m_PolarizedState;
projects/biogears/libBiogears/include/biogears/cdm/circuit/SECircuitPath.h:  SEOpenClosed m_NextPolarizedState;
projects/biogears/libBiogears/include/biogears/cdm/enums/SEPropertyEnums.h:enum class BIOGEARS_API SEOpenClosed {
projects/biogears/libBiogears/include/biogears/cdm/enums/SEPropertyEnums.h:inline std::string ToString(const SEOpenClosed& e)
projects/biogears/libBiogears/include/biogears/cdm/enums/SEPropertyEnums.h:  case SEOpenClosed::Closed:
projects/biogears/libBiogears/include/biogears/cdm/enums/SEPropertyEnums.h:  case SEOpenClosed::Open:
projects/biogears/libBiogears/include/biogears/cdm/enums/SEPropertyEnums.h:inline std::ostream& operator<<(std::ostream& os, const SEOpenClosed& e)
projects/biogears/libBiogears/include/biogears/engine/BioGearsPhysiologyEngine.h:    DEFINE_STATIC_STRING_EX(LeftLung, LeftLungPulmonary);
projects/biogears/libBiogears/include/biogears/engine/BioGearsPhysiologyEngine.h:    DEFINE_STATIC_STRING_EX(RightLung, RightLungPulmonary);
projects/biogears/libBiogears/src/io/cdm/Property.cpp:  // SEOpenClosed
projects/biogears/libBiogears/src/io/cdm/Property.cpp:  void Property::UnMarshall(const CDM::enumOpenClosed& in, SEOpenClosed& out)
projects/biogears/libBiogears/src/io/cdm/Property.cpp:      case CDM::enumOpenClosed::Open:
projects/biogears/libBiogears/src/io/cdm/Property.cpp:        out = SEOpenClosed::Open;
projects/biogears/libBiogears/src/io/cdm/Property.cpp:      case CDM::enumOpenClosed::Closed:
projects/biogears/libBiogears/src/io/cdm/Property.cpp:        out = SEOpenClosed::Closed;
projects/biogears/libBiogears/src/io/cdm/Property.cpp:        out = SEOpenClosed::Invalid;
projects/biogears/libBiogears/src/io/cdm/Property.cpp:      out = SEOpenClosed::Invalid;
projects/biogears/libBiogears/src/io/cdm/Property.cpp:  void Property::Marshall(const SEOpenClosed& in, CDM::enumOpenClosed& out)
projects/biogears/libBiogears/src/io/cdm/Property.cpp:    case SEOpenClosed::Open:
projects/biogears/libBiogears/src/io/cdm/Property.cpp:      out = CDM::enumOpenClosed::Open;
projects/biogears/libBiogears/src/io/cdm/Property.cpp:    case SEOpenClosed::Closed:
projects/biogears/libBiogears/src/io/cdm/Property.cpp:      out = CDM::enumOpenClosed::Closed;
projects/biogears/libBiogears/src/io/cdm/Property.cpp:      // out = (CDM::enumOpenClosed::value)-1;
projects/biogears/libBiogears/src/io/cdm/Property.cpp:bool operator==(CDM::enumOpenClosed const& lhs, SEOpenClosed const& rhs)
projects/biogears/libBiogears/src/io/cdm/Property.cpp:  case SEOpenClosed::Open:
projects/biogears/libBiogears/src/io/cdm/Property.cpp:    return (CDM::enumOpenClosed::Open == lhs);
projects/biogears/libBiogears/src/io/cdm/Property.cpp:  case SEOpenClosed::Closed:
projects/biogears/libBiogears/src/io/cdm/Property.cpp:    return (CDM::enumOpenClosed::Closed == lhs);
projects/biogears/libBiogears/src/io/cdm/Property.cpp:  case SEOpenClosed::Invalid:
projects/biogears/libBiogears/src/io/cdm/Property.cpp:    return ((CDM::enumOpenClosed::value)-1 == lhs);
projects/biogears/libBiogears/src/io/cdm/Patient.cpp:  // SEOpenClosed
projects/biogears/libBiogears/src/io/cdm/Property.h:    // SEOpenClosed
projects/biogears/libBiogears/src/io/cdm/Property.h:    static void UnMarshall(const CDM::enumOpenClosed& in, SEOpenClosed& out);
projects/biogears/libBiogears/src/io/cdm/Property.h:    static void Marshall(const SEOpenClosed& in, CDM::enumOpenClosed& out);
projects/biogears/libBiogears/src/io/cdm/Property.h:bool operator==(CDM::enumOpenClosed const& lhs, SEOpenClosed const& rhs);
projects/biogears/libBiogears/src/io/cdm/Property.h:inline bool operator==(SEOpenClosed const& lhs, CDM::enumOpenClosed const& rhs)
projects/biogears/libBiogears/src/io/cdm/Property.h:inline bool operator!=(CDM::enumOpenClosed const& lhs, SEOpenClosed const& rhs)
projects/biogears/libBiogears/src/io/cdm/Property.h:inline bool operator!=(SEOpenClosed const& lhs, CDM::enumOpenClosed const& rhs)
projects/biogears/libBiogears/src/cdm/utils/DataTrack.cpp:      Probe(std::string { p->GetName() } + "_Switch", p->GetSwitch() == SEOpenClosed::Open ? 1 : 0);
projects/biogears/libBiogears/src/cdm/utils/DataTrack.cpp:      Probe(std::string { p->GetName() } + "_Valve", p->GetValve() == SEOpenClosed::Closed ? 1 : 0);
projects/biogears/libBiogears/src/cdm/utils/DataTrack.cpp:      Probe(std::string { p->GetName() } + "_Switch", p->GetSwitch() == SEOpenClosed::Open ? 1 : 0);
projects/biogears/libBiogears/src/cdm/utils/DataTrack.cpp:      Probe(std::string { p->GetName() } + "_Valve", p->GetValve() == SEOpenClosed::Closed ? 1 : 0);
projects/biogears/libBiogears/src/cdm/utils/DataTrack.cpp:      Probe(std::string { p->GetName() } + "_Switch", p->GetSwitch() == SEOpenClosed::Open ? 1 : 0);
projects/biogears/libBiogears/src/cdm/utils/DataTrack.cpp:      Probe(std::string { p->GetName() } + "_Valve", p->GetValve() == SEOpenClosed::Closed ? 1 : 0);
projects/biogears/libBiogears/src/cdm/utils/DataTrack.cpp:      Track(std::string { p->GetName() } + "_Switch", time_s, p->GetSwitch() == SEOpenClosed::Open ? 1 : 0);
projects/biogears/libBiogears/src/cdm/utils/DataTrack.cpp:      Track(std::string { p->GetName() } + "_Valve", time_s, p->GetValve() == SEOpenClosed::Closed ? 1 : 0);
projects/biogears/libBiogears/src/cdm/utils/DataTrack.cpp:      Track(std::string { p->GetName() } + "_Switch", time_s, p->GetSwitch() == SEOpenClosed::Open ? 1 : 0);
projects/biogears/libBiogears/src/cdm/utils/DataTrack.cpp:      Track(std::string { p->GetName() } + "_Valve", time_s, p->GetValve() == SEOpenClosed::Closed ? 1 : 0);
projects/biogears/libBiogears/src/cdm/utils/DataTrack.cpp:      Track(std::string { p->GetName() } + "_Switch", time_s, p->GetSwitch() == SEOpenClosed::Open ? 1 : 0);
projects/biogears/libBiogears/src/cdm/utils/DataTrack.cpp:      Track(std::string { p->GetName() } + "_Valve", time_s, p->GetValve() == SEOpenClosed::Closed ? 1 : 0);
projects/biogears/libBiogears/src/engine/Systems/Respiratory.cpp:        m_EnvironmentToLeftChestLeak->SetNextValve(SEOpenClosed::Open);
projects/biogears/libBiogears/src/engine/Systems/Respiratory.cpp:        m_EnvironmentToRightChestLeak->SetNextValve(SEOpenClosed::Open);
projects/biogears/libBiogears/src/engine/Systems/Respiratory.cpp:        m_LeftAlveoliLeakToLeftPleuralCavity->SetNextValve(SEOpenClosed::Open);
projects/biogears/libBiogears/src/engine/Systems/Respiratory.cpp:        m_RightAlveoliLeakToRightPleuralCavity->SetNextValve(SEOpenClosed::Open);
projects/biogears/libBiogears/src/engine/Systems/Environment.cpp:  m_ActiveSwitchPath->SetNextSwitch(SEOpenClosed::Open);
projects/biogears/libBiogears/src/engine/Systems/Environment.cpp:    m_ActiveSwitchPath->SetNextSwitch(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Equipment/AnesthesiaMachine.cpp:  if (!IsEventActive(SEAnesthesiaMachineEvent::ReliefValveActive) && m_pSelectorToReliefValve->GetNextValve() == SEOpenClosed::Closed) {
projects/biogears/libBiogears/src/engine/Equipment/AnesthesiaMachine.cpp:  } else if (IsEventActive(SEAnesthesiaMachineEvent::ReliefValveActive) && m_pSelectorToReliefValve->GetNextValve() == SEOpenClosed::Open) {
projects/biogears/libBiogears/src/engine/Equipment/AnesthesiaMachine.cpp:  m_pSelectorToReliefValve->SetNextValve(SEOpenClosed::Open);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  RightAtrium2ToRightVentricle1.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  RightVentricle1ToMainPulmonaryArteries.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  // MainPulmonaryArteriesToRightIntermediatePulmonaryArteries.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  // RightIntermediatePulmonaryVeinsToLeftAtrium2.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  // MainPulmonaryArteriesToLeftIntermediatePulmonaryArteries.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  // LeftIntermediatePulmonaryVeinsToLeftAtrium2.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  LeftAtrium2ToLeftVentricle1.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  LeftVentricle1ToAorta2.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  pCerebralVeinsCheckToVeins2.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  RightNetGlomerularCapillariesToNetBowmansCapsules.SetNextPolarizedState(SEOpenClosed::Open);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  RightUreterToBladder.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  LeftNetGlomerularCapillariesToNetBowmansCapsules.SetNextPolarizedState(SEOpenClosed::Open);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  LeftUreterToBladder.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  FatL2ToLymph.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  BoneL2ToLymph.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  BrainL2ToLymph.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  GutL2ToLymph.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  LeftKidneyL2ToLymph.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  LeftLungL2ToLymph.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  LiverL2ToLymph.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  MuscleL2ToLymph.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  MyocardiumL2ToLymph.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  RightKidneyL2ToLymph.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  RightLungL2ToLymph.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  SkinL2ToLymph.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  SpleenL2ToLymph.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  RightBronchiToRightPleuralConnection.SetNextPolarizedState(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  LeftBronchiToLeftPleuralConnection.SetNextPolarizedState(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  RightAlveoliToRightPleuralConnection.SetNextPolarizedState(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  LeftAlveoliToLeftPleuralConnection.SetNextPolarizedState(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  RightAlveoliToRightAlveoliLeak.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  LeftAlveoliToLeftAlveoliLeak.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  RightChestLeakToRightPleuralCavity.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  LeftChestLeakToLeftPleuralCavity.SetNextValve(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  VentilatorToVentilatorConnection.SetNextPolarizedState(SEOpenClosed::Closed);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  SelectorToReliefValve.SetNextValve(SEOpenClosed::Open);
projects/biogears/libBiogears/src/engine/Controller/BioGears.cpp:  ActiveToClothing.SetNextSwitch(SEOpenClosed::Open);
projects/biogears/swig_bindings/bindings/biogears/cdm/properties/PropertyEnum.swg:%nspace biogears::SEOpenClosed;
projects/biogears/swig_bindings/bindings/biogears/cdm/properties/PropertyEnum.swg:%rename(SEOpenClosed_toString) biogears::ToString(const SEOpenClosed& e);
projects/biogears/swig_bindings/bindings/biogears/cdm/properties/PropertyEnum.swg:%ignore operator<<(std::ostream& os, const SEOpenClosed& e);
projects/biogears/swig_bindings/bindings/biogears/cdm/CommonDataModel.swg:    class enumOpenClosed {

```
