# https://github.com/RamanLab/iCOMIC

```console
icomic/rules/Mutect2.smk:        "VALIDATION_STRINGENCY=SILENT SO=coordinate RGLB=lib1 RGPL=illumina RGPU={sample}-{unit}-normal RGSM={sample}-{unit}-normal"
icomic/rules/Mutect2.smk:        "VALIDATION_STRINGENCY=SILENT SO=coordinate RGLB=lib1 RGPL=illumina RGPU={sample}-{unit}-tumor RGSM={sample}-{unit}-tumor"
icomic/rules/Bowtie2.smk:#        "VALIDATION_STRINGENCY=SILENT SO=coordinate RGLB=lib1 RGPL=illumina RGPU={sample}-#{unit}-{condition} RGSM={sample}-{unit}-{condition}"
icomic/rules/bcftools_call.smk:        "VALIDATION_STRINGENCY=SILENT SO=coordinate RGLB=lib1 RGPL=illumina RGPU={sample}-{unit}-{condition} RGSM={sample}-{unit}-{condition}"
icomic/rules/freebayes.smk:        "VALIDATION_STRINGENCY=SILENT SO=coordinate RGLB=lib1 RGPL=illumina RGPU={sample}-{unit}-{condition} RGSM={sample}-{unit}-{condition}"
icomic/rules/GATK_HC.smk:        "VALIDATION_STRINGENCY=SILENT SO=coordinate RGLB=lib1 RGPL=illumina RGPU={sample}-{unit}-{condition} RGSM={sample}-{unit}-{condition}"
icomic/rules/GEM3.smk:#        "VALIDATION_STRINGENCY=SILENT SO=coordinate RGLB=lib1 RGPL=illumina RGPU={sample}-#{unit}-{condition} RGSM={sample}-{unit}-{condition}"
icomic/icomic_v0.py:        self.runctagpushButton = QtWidgets.QPushButton(self.CTAG)
icomic/icomic_v0.py:        self.runctagpushButton.setGeometry(QtCore.QRect(270, 350, 200, 30))
icomic/icomic_v0.py:        self.runctagpushButton.setObjectName("runctagpushButton")
icomic/icomic_v0.py:        self.resultctagpushButton = QtWidgets.QPushButton(self.CTAG)
icomic/icomic_v0.py:        self.resultctagpushButton.setGeometry(QtCore.QRect(290, 410, 150, 50))
icomic/icomic_v0.py:        self.resultctagpushButton.setObjectName("resultctagpushButton")
icomic/icomic_v0.py:        self.runctagpushButton.setText(_translate("MainWindow", "cTaG"))
icomic/icomic_v0.py:        self.runctagpushButton.setIcon(QtGui.QIcon(os.path.join(module_dir,'./icons/run1.svg')))
icomic/icomic_v0.py:        self.runctagpushButton.setText(_translate("MainWindow", "  Run cTaG"))
icomic/icomic_v0.py:        self.runctagpushButton.setIconSize(QtCore.QSize (22, 22))        
icomic/icomic_v0.py:        self.resultctagpushButton.setText(_translate("MainWindow", "cTAG"))
icomic/icomic_v0.py:        self.resultctagpushButton.setIcon(QtGui.QIcon(os.path.join(module_dir,'./icons/document.svg')))
icomic/icomic_v0.py:        self.resultctagpushButton.setText(_translate("MainWindow", " View Results"))
icomic/icomic_v0.py:        self.resultctagpushButton.setIconSize(QtCore.QSize (22, 22))                
icomic/icomic_v0.py:        self.runctagpushButton.clicked.connect(self.on_click_run_ctag)
icomic/icomic_v0.py:        self.resultctagpushButton.clicked.connect(self.on_click_result_ctag)

```
