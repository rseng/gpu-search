# https://github.com/tobiasrausch/wally

```console
Makefile:	if [ -f src/opencv/CMakeLists.txt ]; then cd src/opencv/ && mkdir build && cd build/ &&  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=${PWD}/src/ocv -D BUILD_SHARED_LIBS=${CVSHARED} -DOPENCV_GENERATE_PKGCONFIG=ON -D BUILD_ZLIB=ON -D BUILD_PNG=ON -D BUILD_OPENJPEG=ON -D WITH_OPENEXR=OFF -D WITH_JPEG=OFF -D WITH_JASPER=OFF -D WITH_TIFF=OFF -D WITH_WEBP=OFF -D WITH_OPENCL=OFF -D WITH_GTK=${CVSHARED} -D WITH_FFMPEG=OFF -D WITH_1394=OFF -D WITH_IPP=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_opencv_apps=OFF .. &&  make -j 4 && make install && cd ../ && rm -rf build/ && cd ../../ && touch .opencv; fi

```
