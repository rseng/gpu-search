# https://github.com/omuse-geoscience/omuse

```console
support/setup_codes.py:         "build variants of the codes (gpu versions etc)"),
support/setup_codes.py:        self.found_cuda = False
support/setup_codes.py:        self.set_cuda_variables()
support/setup_codes.py:    def set_cuda_variables(self):
support/setup_codes.py:        if self.config and self.config.cuda.is_enabled:
support/setup_codes.py:            self.found_cuda = True
support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L'+self.config.cuda.toolkit_path+'/lib' + ' -L'+self.config.cuda.toolkit_path+'/lib64'
support/setup_codes.py:            self.environment['CUDA_TK'] = self.config.cuda.toolkit_path
support/setup_codes.py:            self.environment['CUDA_SDK'] = self.config.cuda.sdk_path
support/setup_codes.py:            if hasattr(self.config.cuda, 'cuda_libs'):
support/setup_codes.py:                self.environment['CUDA_LIBS'] = self.config.cuda.cuda_libs
support/setup_codes.py:                raise DistutilsError("configuration is not up to date for cuda, please reconfigure amuse by running 'configure --enable-cuda'")
support/setup_codes.py:        if self.config and not self.config.cuda.is_enabled:
support/setup_codes.py:            self.found_cuda = True
support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L/NOCUDACONFIGURED/lib' + ' -LNOCUDACONFIGURED/lib64'
support/setup_codes.py:            self.environment['CUDA_LIBS'] = '-lnocuda'
support/setup_codes.py:            self.environment['CUDART_LIBS'] = '-lnocudart'
support/setup_codes.py:            self.environment['CUDA_TK'] = '/NOCUDACONFIGURED'
support/setup_codes.py:            self.environment['CUDA_SDK'] = '/NOCUDACONFIGURED'
support/setup_codes.py:        for x in ['CUDA_TK', 'CUDA_SDK']:
support/setup_codes.py:            cuda_dir = self.environment['CUDA_TK']
support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L'+cuda_dir+'/lib' +  ' -L'+cuda_dir+'/lib64'
support/setup_codes.py:            self.environment['CUDA_LIBS'] = '-lcudart'
support/setup_codes.py:            self.found_cuda = False
support/setup_codes.py:            self.environment_notset['CUDA_SDK'] = '<directory>'
support/setup_codes.py:            self.environment_notset['CUDA_TK'] = '<directory>'
support/setup_codes.py:        cuda_dir = os.path.dirname(os.path.dirname(dir))
support/setup_codes.py:        self.environment['CUDA_LIBDIRS'] = '-L'+cuda_dir+'/lib' + ' -L'+cuda_dir+'/lib64'
support/setup_codes.py:        self.environment['CUDA_LIBS'] = '-lcudart'
support/setup_codes.py:        self.environment['CUDA_TK'] = cuda_dir
support/setup_codes.py:        if not 'CUDA_SDK' in self.environment:
support/setup_codes.py:            self.environment_notset['CUDA_SDK'] = '<directory>'
support/setup_codes.py:        self.found_cuda = True
support/setup_codes.py:            if self.config and hasattr(self.config.cuda, 'sapporo_version'):
support/setup_codes.py:                if self.config.cuda.sapporo_version == '2':
support/setup_codes.py:            if line.startswith('muse_worker_gpu:'):
support/setup_codes.py:                result.append(('muse_worker_gpu', 'GPU',))
support/setup_codes.py:    def is_cuda_needed(self, string):
support/setup_codes.py:            if 'CUDA_TK variable is not set' in line:
support/setup_codes.py:            if 'CUDA_SDK variable is not set' in line:
support/setup_codes.py:        is_cuda_needed = list()
support/setup_codes.py:                elif self.is_cuda_needed(outputlog):
support/setup_codes.py:                    is_cuda_needed.append(x[len(self.lib_dir) + 1:])
support/setup_codes.py:                elif self.is_cuda_needed(outputlog):
support/setup_codes.py:                    is_cuda_needed.append(shortname)
support/setup_codes.py:        if not_build or not_build_special or is_download_needed or is_cuda_needed or are_python_imports_needed:
support/setup_codes.py:            if is_cuda_needed:
support/setup_codes.py:                self.announce("Optional builds failed, need CUDA/GPU libraries:",  level = level)
support/setup_codes.py:                for x in is_cuda_needed:
packages/omuse/support/setup_codes.py:         "build variants of the codes (gpu versions etc)"),
packages/omuse/support/setup_codes.py:        self.found_cuda = False
packages/omuse/support/setup_codes.py:        self.set_cuda_variables()
packages/omuse/support/setup_codes.py:    def set_cuda_variables(self):
packages/omuse/support/setup_codes.py:        if self.config and self.config.cuda.is_enabled:
packages/omuse/support/setup_codes.py:            self.found_cuda = True
packages/omuse/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L'+self.config.cuda.toolkit_path+'/lib' + ' -L'+self.config.cuda.toolkit_path+'/lib64'
packages/omuse/support/setup_codes.py:            self.environment['CUDA_TK'] = self.config.cuda.toolkit_path
packages/omuse/support/setup_codes.py:            self.environment['CUDA_SDK'] = self.config.cuda.sdk_path
packages/omuse/support/setup_codes.py:            if hasattr(self.config.cuda, 'cuda_libs'):
packages/omuse/support/setup_codes.py:                self.environment['CUDA_LIBS'] = self.config.cuda.cuda_libs
packages/omuse/support/setup_codes.py:                raise DistutilsError("configuration is not up to date for cuda, please reconfigure amuse by running 'configure --enable-cuda'")
packages/omuse/support/setup_codes.py:        if self.config and not self.config.cuda.is_enabled:
packages/omuse/support/setup_codes.py:            self.found_cuda = True
packages/omuse/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L/NOCUDACONFIGURED/lib' + ' -LNOCUDACONFIGURED/lib64'
packages/omuse/support/setup_codes.py:            self.environment['CUDA_LIBS'] = '-lnocuda'
packages/omuse/support/setup_codes.py:            self.environment['CUDART_LIBS'] = '-lnocudart'
packages/omuse/support/setup_codes.py:            self.environment['CUDA_TK'] = '/NOCUDACONFIGURED'
packages/omuse/support/setup_codes.py:            self.environment['CUDA_SDK'] = '/NOCUDACONFIGURED'
packages/omuse/support/setup_codes.py:        for x in ['CUDA_TK', 'CUDA_SDK']:
packages/omuse/support/setup_codes.py:            cuda_dir = self.environment['CUDA_TK']
packages/omuse/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L'+cuda_dir+'/lib' +  ' -L'+cuda_dir+'/lib64'
packages/omuse/support/setup_codes.py:            self.environment['CUDA_LIBS'] = '-lcudart'
packages/omuse/support/setup_codes.py:            self.found_cuda = False
packages/omuse/support/setup_codes.py:            self.environment_notset['CUDA_SDK'] = '<directory>'
packages/omuse/support/setup_codes.py:            self.environment_notset['CUDA_TK'] = '<directory>'
packages/omuse/support/setup_codes.py:        cuda_dir = os.path.dirname(os.path.dirname(dir))
packages/omuse/support/setup_codes.py:        self.environment['CUDA_LIBDIRS'] = '-L'+cuda_dir+'/lib' + ' -L'+cuda_dir+'/lib64'
packages/omuse/support/setup_codes.py:        self.environment['CUDA_LIBS'] = '-lcudart'
packages/omuse/support/setup_codes.py:        self.environment['CUDA_TK'] = cuda_dir
packages/omuse/support/setup_codes.py:        if not 'CUDA_SDK' in self.environment:
packages/omuse/support/setup_codes.py:            self.environment_notset['CUDA_SDK'] = '<directory>'
packages/omuse/support/setup_codes.py:        self.found_cuda = True
packages/omuse/support/setup_codes.py:            if self.config and hasattr(self.config.cuda, 'sapporo_version'):
packages/omuse/support/setup_codes.py:                if self.config.cuda.sapporo_version == '2':
packages/omuse/support/setup_codes.py:            if line.startswith('muse_worker_gpu:'):
packages/omuse/support/setup_codes.py:                result.append(('muse_worker_gpu', 'GPU',))
packages/omuse/support/setup_codes.py:    def is_cuda_needed(self, string):
packages/omuse/support/setup_codes.py:            if 'CUDA_TK variable is not set' in line:
packages/omuse/support/setup_codes.py:            if 'CUDA_SDK variable is not set' in line:
packages/omuse/support/setup_codes.py:        is_cuda_needed = list()
packages/omuse/support/setup_codes.py:                elif self.is_cuda_needed(outputlog):
packages/omuse/support/setup_codes.py:                    is_cuda_needed.append(x[len(self.lib_dir) + 1:])
packages/omuse/support/setup_codes.py:                elif self.is_cuda_needed(outputlog):
packages/omuse/support/setup_codes.py:                    is_cuda_needed.append(shortname)
packages/omuse/support/setup_codes.py:        if not_build or not_build_special or is_download_needed or is_cuda_needed or are_python_imports_needed:
packages/omuse/support/setup_codes.py:            if is_cuda_needed:
packages/omuse/support/setup_codes.py:                self.announce("Optional builds failed, need CUDA/GPU libraries:",  level = level)
packages/omuse/support/setup_codes.py:                for x in is_cuda_needed:
packages/omuse-swan/support/setup_codes.py:         "build variants of the codes (gpu versions etc)"),
packages/omuse-swan/support/setup_codes.py:        self.found_cuda = False
packages/omuse-swan/support/setup_codes.py:        self.set_cuda_variables()
packages/omuse-swan/support/setup_codes.py:    def set_cuda_variables(self):
packages/omuse-swan/support/setup_codes.py:        if self.config and self.config.cuda.is_enabled:
packages/omuse-swan/support/setup_codes.py:            self.found_cuda = True
packages/omuse-swan/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L'+self.config.cuda.toolkit_path+'/lib' + ' -L'+self.config.cuda.toolkit_path+'/lib64'
packages/omuse-swan/support/setup_codes.py:            self.environment['CUDA_TK'] = self.config.cuda.toolkit_path
packages/omuse-swan/support/setup_codes.py:            self.environment['CUDA_SDK'] = self.config.cuda.sdk_path
packages/omuse-swan/support/setup_codes.py:            if hasattr(self.config.cuda, 'cuda_libs'):
packages/omuse-swan/support/setup_codes.py:                self.environment['CUDA_LIBS'] = self.config.cuda.cuda_libs
packages/omuse-swan/support/setup_codes.py:                raise DistutilsError("configuration is not up to date for cuda, please reconfigure amuse by running 'configure --enable-cuda'")
packages/omuse-swan/support/setup_codes.py:        if self.config and not self.config.cuda.is_enabled:
packages/omuse-swan/support/setup_codes.py:            self.found_cuda = True
packages/omuse-swan/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L/NOCUDACONFIGURED/lib' + ' -LNOCUDACONFIGURED/lib64'
packages/omuse-swan/support/setup_codes.py:            self.environment['CUDA_LIBS'] = '-lnocuda'
packages/omuse-swan/support/setup_codes.py:            self.environment['CUDART_LIBS'] = '-lnocudart'
packages/omuse-swan/support/setup_codes.py:            self.environment['CUDA_TK'] = '/NOCUDACONFIGURED'
packages/omuse-swan/support/setup_codes.py:            self.environment['CUDA_SDK'] = '/NOCUDACONFIGURED'
packages/omuse-swan/support/setup_codes.py:        for x in ['CUDA_TK', 'CUDA_SDK']:
packages/omuse-swan/support/setup_codes.py:            cuda_dir = self.environment['CUDA_TK']
packages/omuse-swan/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L'+cuda_dir+'/lib' +  ' -L'+cuda_dir+'/lib64'
packages/omuse-swan/support/setup_codes.py:            self.environment['CUDA_LIBS'] = '-lcudart'
packages/omuse-swan/support/setup_codes.py:            self.found_cuda = False
packages/omuse-swan/support/setup_codes.py:            self.environment_notset['CUDA_SDK'] = '<directory>'
packages/omuse-swan/support/setup_codes.py:            self.environment_notset['CUDA_TK'] = '<directory>'
packages/omuse-swan/support/setup_codes.py:        cuda_dir = os.path.dirname(os.path.dirname(dir))
packages/omuse-swan/support/setup_codes.py:        self.environment['CUDA_LIBDIRS'] = '-L'+cuda_dir+'/lib' + ' -L'+cuda_dir+'/lib64'
packages/omuse-swan/support/setup_codes.py:        self.environment['CUDA_LIBS'] = '-lcudart'
packages/omuse-swan/support/setup_codes.py:        self.environment['CUDA_TK'] = cuda_dir
packages/omuse-swan/support/setup_codes.py:        if not 'CUDA_SDK' in self.environment:
packages/omuse-swan/support/setup_codes.py:            self.environment_notset['CUDA_SDK'] = '<directory>'
packages/omuse-swan/support/setup_codes.py:        self.found_cuda = True
packages/omuse-swan/support/setup_codes.py:            if self.config and hasattr(self.config.cuda, 'sapporo_version'):
packages/omuse-swan/support/setup_codes.py:                if self.config.cuda.sapporo_version == '2':
packages/omuse-swan/support/setup_codes.py:            if line.startswith('muse_worker_gpu:'):
packages/omuse-swan/support/setup_codes.py:                result.append(('muse_worker_gpu', 'GPU',))
packages/omuse-swan/support/setup_codes.py:    def is_cuda_needed(self, string):
packages/omuse-swan/support/setup_codes.py:            if 'CUDA_TK variable is not set' in line:
packages/omuse-swan/support/setup_codes.py:            if 'CUDA_SDK variable is not set' in line:
packages/omuse-swan/support/setup_codes.py:        is_cuda_needed = list()
packages/omuse-swan/support/setup_codes.py:                elif self.is_cuda_needed(outputlog):
packages/omuse-swan/support/setup_codes.py:                    is_cuda_needed.append(x[len(self.lib_dir) + 1:])
packages/omuse-swan/support/setup_codes.py:                elif self.is_cuda_needed(outputlog):
packages/omuse-swan/support/setup_codes.py:                    is_cuda_needed.append(shortname)
packages/omuse-swan/support/setup_codes.py:        if not_build or not_build_special or is_download_needed or is_cuda_needed or are_python_imports_needed:
packages/omuse-swan/support/setup_codes.py:            if is_cuda_needed:
packages/omuse-swan/support/setup_codes.py:                self.announce("Optional builds failed, need CUDA/GPU libraries:",  level = level)
packages/omuse-swan/support/setup_codes.py:                for x in is_cuda_needed:
packages/omuse-era5/support/setup_codes.py:         "build variants of the codes (gpu versions etc)"),
packages/omuse-era5/support/setup_codes.py:        self.found_cuda = False
packages/omuse-era5/support/setup_codes.py:        self.set_cuda_variables()
packages/omuse-era5/support/setup_codes.py:    def set_cuda_variables(self):
packages/omuse-era5/support/setup_codes.py:        if self.config and self.config.cuda.is_enabled:
packages/omuse-era5/support/setup_codes.py:            self.found_cuda = True
packages/omuse-era5/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L'+self.config.cuda.toolkit_path+'/lib' + ' -L'+self.config.cuda.toolkit_path+'/lib64'
packages/omuse-era5/support/setup_codes.py:            self.environment['CUDA_TK'] = self.config.cuda.toolkit_path
packages/omuse-era5/support/setup_codes.py:            self.environment['CUDA_SDK'] = self.config.cuda.sdk_path
packages/omuse-era5/support/setup_codes.py:            if hasattr(self.config.cuda, 'cuda_libs'):
packages/omuse-era5/support/setup_codes.py:                self.environment['CUDA_LIBS'] = self.config.cuda.cuda_libs
packages/omuse-era5/support/setup_codes.py:                raise DistutilsError("configuration is not up to date for cuda, please reconfigure amuse by running 'configure --enable-cuda'")
packages/omuse-era5/support/setup_codes.py:        if self.config and not self.config.cuda.is_enabled:
packages/omuse-era5/support/setup_codes.py:            self.found_cuda = True
packages/omuse-era5/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L/NOCUDACONFIGURED/lib' + ' -LNOCUDACONFIGURED/lib64'
packages/omuse-era5/support/setup_codes.py:            self.environment['CUDA_LIBS'] = '-lnocuda'
packages/omuse-era5/support/setup_codes.py:            self.environment['CUDART_LIBS'] = '-lnocudart'
packages/omuse-era5/support/setup_codes.py:            self.environment['CUDA_TK'] = '/NOCUDACONFIGURED'
packages/omuse-era5/support/setup_codes.py:            self.environment['CUDA_SDK'] = '/NOCUDACONFIGURED'
packages/omuse-era5/support/setup_codes.py:        for x in ['CUDA_TK', 'CUDA_SDK']:
packages/omuse-era5/support/setup_codes.py:            cuda_dir = self.environment['CUDA_TK']
packages/omuse-era5/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L'+cuda_dir+'/lib' +  ' -L'+cuda_dir+'/lib64'
packages/omuse-era5/support/setup_codes.py:            self.environment['CUDA_LIBS'] = '-lcudart'
packages/omuse-era5/support/setup_codes.py:            self.found_cuda = False
packages/omuse-era5/support/setup_codes.py:            self.environment_notset['CUDA_SDK'] = '<directory>'
packages/omuse-era5/support/setup_codes.py:            self.environment_notset['CUDA_TK'] = '<directory>'
packages/omuse-era5/support/setup_codes.py:        cuda_dir = os.path.dirname(os.path.dirname(dir))
packages/omuse-era5/support/setup_codes.py:        self.environment['CUDA_LIBDIRS'] = '-L'+cuda_dir+'/lib' + ' -L'+cuda_dir+'/lib64'
packages/omuse-era5/support/setup_codes.py:        self.environment['CUDA_LIBS'] = '-lcudart'
packages/omuse-era5/support/setup_codes.py:        self.environment['CUDA_TK'] = cuda_dir
packages/omuse-era5/support/setup_codes.py:        if not 'CUDA_SDK' in self.environment:
packages/omuse-era5/support/setup_codes.py:            self.environment_notset['CUDA_SDK'] = '<directory>'
packages/omuse-era5/support/setup_codes.py:        self.found_cuda = True
packages/omuse-era5/support/setup_codes.py:            if self.config and hasattr(self.config.cuda, 'sapporo_version'):
packages/omuse-era5/support/setup_codes.py:                if self.config.cuda.sapporo_version == '2':
packages/omuse-era5/support/setup_codes.py:            if line.startswith('muse_worker_gpu:'):
packages/omuse-era5/support/setup_codes.py:                result.append(('muse_worker_gpu', 'GPU',))
packages/omuse-era5/support/setup_codes.py:    def is_cuda_needed(self, string):
packages/omuse-era5/support/setup_codes.py:            if 'CUDA_TK variable is not set' in line:
packages/omuse-era5/support/setup_codes.py:            if 'CUDA_SDK variable is not set' in line:
packages/omuse-era5/support/setup_codes.py:        is_cuda_needed = list()
packages/omuse-era5/support/setup_codes.py:                elif self.is_cuda_needed(outputlog):
packages/omuse-era5/support/setup_codes.py:                    is_cuda_needed.append(x[len(self.lib_dir) + 1:])
packages/omuse-era5/support/setup_codes.py:                elif self.is_cuda_needed(outputlog):
packages/omuse-era5/support/setup_codes.py:                    is_cuda_needed.append(shortname)
packages/omuse-era5/support/setup_codes.py:        if not_build or not_build_special or is_download_needed or is_cuda_needed or are_python_imports_needed:
packages/omuse-era5/support/setup_codes.py:            if is_cuda_needed:
packages/omuse-era5/support/setup_codes.py:                self.announce("Optional builds failed, need CUDA/GPU libraries:",  level = level)
packages/omuse-era5/support/setup_codes.py:                for x in is_cuda_needed:
packages/omuse-qgmodel/support/setup_codes.py:         "build variants of the codes (gpu versions etc)"),
packages/omuse-qgmodel/support/setup_codes.py:        self.found_cuda = False
packages/omuse-qgmodel/support/setup_codes.py:        self.set_cuda_variables()
packages/omuse-qgmodel/support/setup_codes.py:    def set_cuda_variables(self):
packages/omuse-qgmodel/support/setup_codes.py:        if self.config and self.config.cuda.is_enabled:
packages/omuse-qgmodel/support/setup_codes.py:            self.found_cuda = True
packages/omuse-qgmodel/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L'+self.config.cuda.toolkit_path+'/lib' + ' -L'+self.config.cuda.toolkit_path+'/lib64'
packages/omuse-qgmodel/support/setup_codes.py:            self.environment['CUDA_TK'] = self.config.cuda.toolkit_path
packages/omuse-qgmodel/support/setup_codes.py:            self.environment['CUDA_SDK'] = self.config.cuda.sdk_path
packages/omuse-qgmodel/support/setup_codes.py:            if hasattr(self.config.cuda, 'cuda_libs'):
packages/omuse-qgmodel/support/setup_codes.py:                self.environment['CUDA_LIBS'] = self.config.cuda.cuda_libs
packages/omuse-qgmodel/support/setup_codes.py:                raise DistutilsError("configuration is not up to date for cuda, please reconfigure amuse by running 'configure --enable-cuda'")
packages/omuse-qgmodel/support/setup_codes.py:        if self.config and not self.config.cuda.is_enabled:
packages/omuse-qgmodel/support/setup_codes.py:            self.found_cuda = True
packages/omuse-qgmodel/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L/NOCUDACONFIGURED/lib' + ' -LNOCUDACONFIGURED/lib64'
packages/omuse-qgmodel/support/setup_codes.py:            self.environment['CUDA_LIBS'] = '-lnocuda'
packages/omuse-qgmodel/support/setup_codes.py:            self.environment['CUDART_LIBS'] = '-lnocudart'
packages/omuse-qgmodel/support/setup_codes.py:            self.environment['CUDA_TK'] = '/NOCUDACONFIGURED'
packages/omuse-qgmodel/support/setup_codes.py:            self.environment['CUDA_SDK'] = '/NOCUDACONFIGURED'
packages/omuse-qgmodel/support/setup_codes.py:        for x in ['CUDA_TK', 'CUDA_SDK']:
packages/omuse-qgmodel/support/setup_codes.py:            cuda_dir = self.environment['CUDA_TK']
packages/omuse-qgmodel/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L'+cuda_dir+'/lib' +  ' -L'+cuda_dir+'/lib64'
packages/omuse-qgmodel/support/setup_codes.py:            self.environment['CUDA_LIBS'] = '-lcudart'
packages/omuse-qgmodel/support/setup_codes.py:            self.found_cuda = False
packages/omuse-qgmodel/support/setup_codes.py:            self.environment_notset['CUDA_SDK'] = '<directory>'
packages/omuse-qgmodel/support/setup_codes.py:            self.environment_notset['CUDA_TK'] = '<directory>'
packages/omuse-qgmodel/support/setup_codes.py:        cuda_dir = os.path.dirname(os.path.dirname(dir))
packages/omuse-qgmodel/support/setup_codes.py:        self.environment['CUDA_LIBDIRS'] = '-L'+cuda_dir+'/lib' + ' -L'+cuda_dir+'/lib64'
packages/omuse-qgmodel/support/setup_codes.py:        self.environment['CUDA_LIBS'] = '-lcudart'
packages/omuse-qgmodel/support/setup_codes.py:        self.environment['CUDA_TK'] = cuda_dir
packages/omuse-qgmodel/support/setup_codes.py:        if not 'CUDA_SDK' in self.environment:
packages/omuse-qgmodel/support/setup_codes.py:            self.environment_notset['CUDA_SDK'] = '<directory>'
packages/omuse-qgmodel/support/setup_codes.py:        self.found_cuda = True
packages/omuse-qgmodel/support/setup_codes.py:            if self.config and hasattr(self.config.cuda, 'sapporo_version'):
packages/omuse-qgmodel/support/setup_codes.py:                if self.config.cuda.sapporo_version == '2':
packages/omuse-qgmodel/support/setup_codes.py:            if line.startswith('muse_worker_gpu:'):
packages/omuse-qgmodel/support/setup_codes.py:                result.append(('muse_worker_gpu', 'GPU',))
packages/omuse-qgmodel/support/setup_codes.py:    def is_cuda_needed(self, string):
packages/omuse-qgmodel/support/setup_codes.py:            if 'CUDA_TK variable is not set' in line:
packages/omuse-qgmodel/support/setup_codes.py:            if 'CUDA_SDK variable is not set' in line:
packages/omuse-qgmodel/support/setup_codes.py:        is_cuda_needed = list()
packages/omuse-qgmodel/support/setup_codes.py:                elif self.is_cuda_needed(outputlog):
packages/omuse-qgmodel/support/setup_codes.py:                    is_cuda_needed.append(x[len(self.lib_dir) + 1:])
packages/omuse-qgmodel/support/setup_codes.py:                elif self.is_cuda_needed(outputlog):
packages/omuse-qgmodel/support/setup_codes.py:                    is_cuda_needed.append(shortname)
packages/omuse-qgmodel/support/setup_codes.py:        if not_build or not_build_special or is_download_needed or is_cuda_needed or are_python_imports_needed:
packages/omuse-qgmodel/support/setup_codes.py:            if is_cuda_needed:
packages/omuse-qgmodel/support/setup_codes.py:                self.announce("Optional builds failed, need CUDA/GPU libraries:",  level = level)
packages/omuse-qgmodel/support/setup_codes.py:                for x in is_cuda_needed:
packages/omuse-iemic/support/setup_codes.py:         "build variants of the codes (gpu versions etc)"),
packages/omuse-iemic/support/setup_codes.py:        self.found_cuda = False
packages/omuse-iemic/support/setup_codes.py:        self.set_cuda_variables()
packages/omuse-iemic/support/setup_codes.py:    def set_cuda_variables(self):
packages/omuse-iemic/support/setup_codes.py:        if self.config and self.config.cuda.is_enabled:
packages/omuse-iemic/support/setup_codes.py:            self.found_cuda = True
packages/omuse-iemic/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L'+self.config.cuda.toolkit_path+'/lib' + ' -L'+self.config.cuda.toolkit_path+'/lib64'
packages/omuse-iemic/support/setup_codes.py:            self.environment['CUDA_TK'] = self.config.cuda.toolkit_path
packages/omuse-iemic/support/setup_codes.py:            self.environment['CUDA_SDK'] = self.config.cuda.sdk_path
packages/omuse-iemic/support/setup_codes.py:            if hasattr(self.config.cuda, 'cuda_libs'):
packages/omuse-iemic/support/setup_codes.py:                self.environment['CUDA_LIBS'] = self.config.cuda.cuda_libs
packages/omuse-iemic/support/setup_codes.py:                raise DistutilsError("configuration is not up to date for cuda, please reconfigure amuse by running 'configure --enable-cuda'")
packages/omuse-iemic/support/setup_codes.py:        if self.config and not self.config.cuda.is_enabled:
packages/omuse-iemic/support/setup_codes.py:            self.found_cuda = True
packages/omuse-iemic/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L/NOCUDACONFIGURED/lib' + ' -LNOCUDACONFIGURED/lib64'
packages/omuse-iemic/support/setup_codes.py:            self.environment['CUDA_LIBS'] = '-lnocuda'
packages/omuse-iemic/support/setup_codes.py:            self.environment['CUDART_LIBS'] = '-lnocudart'
packages/omuse-iemic/support/setup_codes.py:            self.environment['CUDA_TK'] = '/NOCUDACONFIGURED'
packages/omuse-iemic/support/setup_codes.py:            self.environment['CUDA_SDK'] = '/NOCUDACONFIGURED'
packages/omuse-iemic/support/setup_codes.py:        for x in ['CUDA_TK', 'CUDA_SDK']:
packages/omuse-iemic/support/setup_codes.py:            cuda_dir = self.environment['CUDA_TK']
packages/omuse-iemic/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L'+cuda_dir+'/lib' +  ' -L'+cuda_dir+'/lib64'
packages/omuse-iemic/support/setup_codes.py:            self.environment['CUDA_LIBS'] = '-lcudart'
packages/omuse-iemic/support/setup_codes.py:            self.found_cuda = False
packages/omuse-iemic/support/setup_codes.py:            self.environment_notset['CUDA_SDK'] = '<directory>'
packages/omuse-iemic/support/setup_codes.py:            self.environment_notset['CUDA_TK'] = '<directory>'
packages/omuse-iemic/support/setup_codes.py:        cuda_dir = os.path.dirname(os.path.dirname(dir))
packages/omuse-iemic/support/setup_codes.py:        self.environment['CUDA_LIBDIRS'] = '-L'+cuda_dir+'/lib' + ' -L'+cuda_dir+'/lib64'
packages/omuse-iemic/support/setup_codes.py:        self.environment['CUDA_LIBS'] = '-lcudart'
packages/omuse-iemic/support/setup_codes.py:        self.environment['CUDA_TK'] = cuda_dir
packages/omuse-iemic/support/setup_codes.py:        if not 'CUDA_SDK' in self.environment:
packages/omuse-iemic/support/setup_codes.py:            self.environment_notset['CUDA_SDK'] = '<directory>'
packages/omuse-iemic/support/setup_codes.py:        self.found_cuda = True
packages/omuse-iemic/support/setup_codes.py:            if self.config and hasattr(self.config.cuda, 'sapporo_version'):
packages/omuse-iemic/support/setup_codes.py:                if self.config.cuda.sapporo_version == '2':
packages/omuse-iemic/support/setup_codes.py:            if line.startswith('muse_worker_gpu:'):
packages/omuse-iemic/support/setup_codes.py:                result.append(('muse_worker_gpu', 'GPU',))
packages/omuse-iemic/support/setup_codes.py:    def is_cuda_needed(self, string):
packages/omuse-iemic/support/setup_codes.py:            if 'CUDA_TK variable is not set' in line:
packages/omuse-iemic/support/setup_codes.py:            if 'CUDA_SDK variable is not set' in line:
packages/omuse-iemic/support/setup_codes.py:        is_cuda_needed = list()
packages/omuse-iemic/support/setup_codes.py:                elif self.is_cuda_needed(outputlog):
packages/omuse-iemic/support/setup_codes.py:                    is_cuda_needed.append(x[len(self.lib_dir) + 1:])
packages/omuse-iemic/support/setup_codes.py:                elif self.is_cuda_needed(outputlog):
packages/omuse-iemic/support/setup_codes.py:                    is_cuda_needed.append(shortname)
packages/omuse-iemic/support/setup_codes.py:        if not_build or not_build_special or is_download_needed or is_cuda_needed or are_python_imports_needed:
packages/omuse-iemic/support/setup_codes.py:            if is_cuda_needed:
packages/omuse-iemic/support/setup_codes.py:                self.announce("Optional builds failed, need CUDA/GPU libraries:",  level = level)
packages/omuse-iemic/support/setup_codes.py:                for x in is_cuda_needed:
packages/omuse-framework/support/setup_codes.py:         "build variants of the codes (gpu versions etc)"),
packages/omuse-framework/support/setup_codes.py:        self.found_cuda = False
packages/omuse-framework/support/setup_codes.py:        self.set_cuda_variables()
packages/omuse-framework/support/setup_codes.py:    def set_cuda_variables(self):
packages/omuse-framework/support/setup_codes.py:        if self.config and self.config.cuda.is_enabled:
packages/omuse-framework/support/setup_codes.py:            self.found_cuda = True
packages/omuse-framework/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L'+self.config.cuda.toolkit_path+'/lib' + ' -L'+self.config.cuda.toolkit_path+'/lib64'
packages/omuse-framework/support/setup_codes.py:            self.environment['CUDA_TK'] = self.config.cuda.toolkit_path
packages/omuse-framework/support/setup_codes.py:            self.environment['CUDA_SDK'] = self.config.cuda.sdk_path
packages/omuse-framework/support/setup_codes.py:            if hasattr(self.config.cuda, 'cuda_libs'):
packages/omuse-framework/support/setup_codes.py:                self.environment['CUDA_LIBS'] = self.config.cuda.cuda_libs
packages/omuse-framework/support/setup_codes.py:                raise DistutilsError("configuration is not up to date for cuda, please reconfigure amuse by running 'configure --enable-cuda'")
packages/omuse-framework/support/setup_codes.py:        if self.config and not self.config.cuda.is_enabled:
packages/omuse-framework/support/setup_codes.py:            self.found_cuda = True
packages/omuse-framework/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L/NOCUDACONFIGURED/lib' + ' -LNOCUDACONFIGURED/lib64'
packages/omuse-framework/support/setup_codes.py:            self.environment['CUDA_LIBS'] = '-lnocuda'
packages/omuse-framework/support/setup_codes.py:            self.environment['CUDART_LIBS'] = '-lnocudart'
packages/omuse-framework/support/setup_codes.py:            self.environment['CUDA_TK'] = '/NOCUDACONFIGURED'
packages/omuse-framework/support/setup_codes.py:            self.environment['CUDA_SDK'] = '/NOCUDACONFIGURED'
packages/omuse-framework/support/setup_codes.py:        for x in ['CUDA_TK', 'CUDA_SDK']:
packages/omuse-framework/support/setup_codes.py:            cuda_dir = self.environment['CUDA_TK']
packages/omuse-framework/support/setup_codes.py:            self.environment['CUDA_LIBDIRS'] = '-L'+cuda_dir+'/lib' +  ' -L'+cuda_dir+'/lib64'
packages/omuse-framework/support/setup_codes.py:            self.environment['CUDA_LIBS'] = '-lcudart'
packages/omuse-framework/support/setup_codes.py:            self.found_cuda = False
packages/omuse-framework/support/setup_codes.py:            self.environment_notset['CUDA_SDK'] = '<directory>'
packages/omuse-framework/support/setup_codes.py:            self.environment_notset['CUDA_TK'] = '<directory>'
packages/omuse-framework/support/setup_codes.py:        cuda_dir = os.path.dirname(os.path.dirname(dir))
packages/omuse-framework/support/setup_codes.py:        self.environment['CUDA_LIBDIRS'] = '-L'+cuda_dir+'/lib' + ' -L'+cuda_dir+'/lib64'
packages/omuse-framework/support/setup_codes.py:        self.environment['CUDA_LIBS'] = '-lcudart'
packages/omuse-framework/support/setup_codes.py:        self.environment['CUDA_TK'] = cuda_dir
packages/omuse-framework/support/setup_codes.py:        if not 'CUDA_SDK' in self.environment:
packages/omuse-framework/support/setup_codes.py:            self.environment_notset['CUDA_SDK'] = '<directory>'
packages/omuse-framework/support/setup_codes.py:        self.found_cuda = True
packages/omuse-framework/support/setup_codes.py:            if self.config and hasattr(self.config.cuda, 'sapporo_version'):
packages/omuse-framework/support/setup_codes.py:                if self.config.cuda.sapporo_version == '2':
packages/omuse-framework/support/setup_codes.py:            if line.startswith('muse_worker_gpu:'):
packages/omuse-framework/support/setup_codes.py:                result.append(('muse_worker_gpu', 'GPU',))
packages/omuse-framework/support/setup_codes.py:    def is_cuda_needed(self, string):
packages/omuse-framework/support/setup_codes.py:            if 'CUDA_TK variable is not set' in line:
packages/omuse-framework/support/setup_codes.py:            if 'CUDA_SDK variable is not set' in line:
packages/omuse-framework/support/setup_codes.py:        is_cuda_needed = list()
packages/omuse-framework/support/setup_codes.py:                elif self.is_cuda_needed(outputlog):
packages/omuse-framework/support/setup_codes.py:                    is_cuda_needed.append(x[len(self.lib_dir) + 1:])
packages/omuse-framework/support/setup_codes.py:                elif self.is_cuda_needed(outputlog):
packages/omuse-framework/support/setup_codes.py:                    is_cuda_needed.append(shortname)
packages/omuse-framework/support/setup_codes.py:        if not_build or not_build_special or is_download_needed or is_cuda_needed or are_python_imports_needed:
packages/omuse-framework/support/setup_codes.py:            if is_cuda_needed:
packages/omuse-framework/support/setup_codes.py:                self.announce("Optional builds failed, need CUDA/GPU libraries:",  level = level)
packages/omuse-framework/support/setup_codes.py:                for x in is_cuda_needed:
packages/omuse-framework/src/omuse/community/qgcm/src/q-gcm_utility.F:*     local common block bcudata
packages/omuse-framework/src/omuse/community/qgcm/src/q-gcm_utility.F:      write(*,203) '  bcudata    local memory   (MB) = ',dim2mb*sizarr
packages/omuse-framework/src/omuse/community/pop/pop_in_lowres:&gpu_mod_nml
packages/omuse-framework/src/omuse/community/pop/pop_in_lowres:  use_gpu = .true.
packages/omuse-framework/src/omuse/community/pop/pop_in_highres:&gpu_mod_nml
packages/omuse-framework/src/omuse/community/pop/pop_in_highres:  use_gpu = .false.
packages/omuse-framework/src/omuse/community/pop/pop_in:&gpu_mod_nml
packages/omuse-framework/src/omuse/community/pop/pop_in:  use_gpu = .true.
src/omuse/community/qgcm/src/q-gcm_utility.F:*     local common block bcudata
src/omuse/community/qgcm/src/q-gcm_utility.F:      write(*,203) '  bcudata    local memory   (MB) = ',dim2mb*sizarr
src/omuse/community/pop/pop_in_lowres:&gpu_mod_nml
src/omuse/community/pop/pop_in_lowres:  use_gpu = .true.
src/omuse/community/pop/pop_in_highres:&gpu_mod_nml
src/omuse/community/pop/pop_in_highres:  use_gpu = .false.
src/omuse/community/pop/pop_in:&gpu_mod_nml
src/omuse/community/pop/pop_in:  use_gpu = .true.

```
