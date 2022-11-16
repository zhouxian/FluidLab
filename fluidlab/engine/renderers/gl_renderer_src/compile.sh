cd FlexRenderer/bindings
rm -rf build
mkdir build
cd build
cmake -DPYBIND11_PYTHON_VERSION=3.7 -DCMAKE_BUILD_TYPE=Release ..
make -j

cd ../../..
cp FlexRenderer/bindings/build/*.so .