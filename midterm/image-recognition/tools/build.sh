# 检查工作目录下是否存在src目录
if [ ! -d src ]; then
    echo "src directory not found, please run this script in the root of the project"
    exit 1
fi

# 检查工作目录下是否存在build目录
if [ ! -d build ]; then
    mkdir build
fi

# 进入build目录
cd build

# cmake
cmake ..

# 执行make命令
make -j16

# 拷贝生成的可执行文件
cp *.o ../test/
cd ..

# clear
rm -rf build
mkdir build

echo "Compile success"