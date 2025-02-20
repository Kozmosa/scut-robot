# 如果不存在就创建build目录
if [ ! -d build ]; then
    mkdir build
fi

if [ ! -d test ]; then
    mkdir test
fi

cd build
rm -rf *
cmake ..
make -j8

mv *.o ../test/
cd ..
cd test
