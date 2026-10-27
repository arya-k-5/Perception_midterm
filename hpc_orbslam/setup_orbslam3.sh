#!/bin/bash
# Build Pangolin and ORB-SLAM3 (monocular) into a conda env on NYU HPC.
# Run inside a Slurm job on a CPU node: sbatch build_orbslam3.sbatch
# Untested on the cluster when written; the log shows where it stops.
set -euo pipefail

ENV=${ENV:-perception}                       # conda env name; set ENV=orbslam3 for a separate env
WORK=${WORK:-$HOME/orbslam3_build}           # build area (not the work folder)
ORB_REPO=https://github.com/UZ-SLAMLab/ORB_SLAM3.git
PANGOLIN_REPO=https://github.com/stevenlovegrove/Pangolin.git

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
PREFIX="$CONDA_PREFIX"
echo "env: $ENV  prefix: $PREFIX"

# 1. Dependencies (skipped if already present in the env)
conda install -y -c conda-forge --override-channels \
  cmake "eigen=3.4" opencv boost-cpp glew libglvnd epoxy libgl-devel libegl-devel \
  xorg-libx11 xorg-libxext xorg-libxrandr xorg-libxinerama xorg-libxcursor xorg-libxi \
  pkg-config ninja

mkdir -p "$WORK" && cd "$WORK"

# 2. Pangolin (viewer library; ORB-SLAM3 includes its headers even when run headless)
if [ ! -d Pangolin ]; then git clone --depth 1 "$PANGOLIN_REPO"; fi
cmake -S Pangolin -B Pangolin/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DOpenGL_GL_PREFERENCE=GLVND \
  -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_TOOLS=OFF -DBUILD_PANGOLIN_PYTHON=OFF \
  -DBUILD_PANGOLIN_FFMPEG=OFF -DBUILD_PANGOLIN_LIBDC=OFF -DBUILD_PANGOLIN_OPENNI=OFF \
  -DBUILD_PANGOLIN_OPENNI2=OFF -DBUILD_PANGOLIN_REALSENSE=OFF -DBUILD_PANGOLIN_PLEORA=OFF \
  -DBUILD_PANGOLIN_UVC=OFF -DBUILD_PANGOLIN_TELICAM=OFF -DBUILD_PANGOLIN_VIDEO=OFF \
  -DBUILD_PANGOLIN_DEPTHSENSE=OFF -DBUILD_PANGOLIN_OPENEXR=OFF -DBUILD_PANGOLIN_LZ4=OFF \
  -DBUILD_PANGOLIN_ZSTD=OFF -DBUILD_PANGOLIN_LIBPNG=OFF -DBUILD_PANGOLIN_LIBJPEG=OFF \
  -DBUILD_PANGOLIN_TIFF=OFF
cmake --build Pangolin/build -j"$(nproc)"
cmake --install Pangolin/build

# 3. ORB-SLAM3 (fresh clone; two source patches, see README next to this script)
if [ ! -d ORB_SLAM3 ]; then git clone --depth 1 "$ORB_REPO"; fi
cd ORB_SLAM3
sed -i 's/-std=c++11/-std=c++14/g' CMakeLists.txt   # Pangolin's sigslot headers need C++14
# Thirdparty libraries (DBoW2, g2o, Sophus). CMAKE_POLICY_VERSION_MINIMUM: old cmake_minimum_required under CMake 4.
for d in Thirdparty/DBoW2 Thirdparty/g2o Thirdparty/Sophus; do
  mkdir -p "$d/build" && cd "$d/build"
  cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX"
  cmake --build . -j"$(nproc)"
  cd "$OLDPWD"
done
# Vocabulary (ORB words for loop closure / relocalisation)
(cd Vocabulary && tar -xf ORBvoc.txt.tar.gz)
# Main library
mkdir -p build && cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PREFIX"
cmake --build . -j"$(nproc)"
echo "BUILD_OK: $WORK/ORB_SLAM3/lib"
