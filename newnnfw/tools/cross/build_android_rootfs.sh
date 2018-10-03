#!/usr/bin/env bash
set -x

# Different from other architectures, the result does not contain only RootFS.
#   $__Rootfs/bin     : Toolchains
#   $__Rootfs/sysroot : RootFS

usage()
{
    echo "Usage: $0 [BuildArch] [NDKVersion] [APILevel] [ACL]"
    echo "BuildArch  : arm or arm64"
    echo "NDKVersion : r16b or higher (Must start with 'r')"
    echo "APILevel   : 27 or higher"
    echo "ACL        : acl (default), noacl (exclude ACL)"
    exit 1
}

__CrossDir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
__BuildArch=arm64
__NDKVersion=r16b
__APILevel="27"
__ACL=1
__UnprocessedBuildArgs=

for i in "$@" ; do
    lowerI="$(echo $i | awk '{print tolower($0)}')"
    case $lowerI in
        -?|-h|--help)
            usage
            exit 1
            ;;
        arm)
            __BuildArch=arm
            ;;
        arm64)
            __BuildArch=arm64
            ;;
        r16b)
            __NDKVersion=r16b
            ;;
        27)
            __APILevel="27"
            ;;
        acl)
            __ACL=1
            ;;
        noacl)
            __ACL=0
            ;;
        *)
            __UnprocessedBuildArgs="$__UnprocessedBuildArgs $i"
            ;;
    esac
done

__ToolchainDir=${TOOLCHAIN_DIR:-"${__CrossDir}/ndk/${__NDKVersion}"}
__RootfsDir=${ROOTFS_DIR:-"${__CrossDir}/rootfs/${__BuildArch}.android"}

NDK_DIR=android-ndk-${__NDKVersion}
NDK_ZIP=${NDK_DIR}-linux-x86_64.zip

if [[ -e $__RootfsDir ]]; then
  echo "ERROR: $__RootfsDir already exists"
  exit 255
fi
if [[ -e $__RootfsDir.gnustl ]]; then
  echo "ERROR: $__RootfsDir.gnustl already exists"
  exit 255
fi

echo "Downloading Android NDK"
rm -rf "$__ToolchainDir"
mkdir -p "$__ToolchainDir"
wget -nv -nc https://dl.google.com/android/repository/$NDK_ZIP -O $__ToolchainDir/$NDK_ZIP

echo "Unzipping Android NDK"
unzip -qq -o $__ToolchainDir/$NDK_ZIP -d $__ToolchainDir
rm $__ToolchainDir/$NDK_ZIP
mv  $__ToolchainDir/${NDK_DIR} "$__ToolchainDir/ndk"

echo "Generating standalone toolchain and rootfs to $__RootfsDir"

$__ToolchainDir/ndk/build/tools/make-standalone-toolchain.sh --arch=$__BuildArch --platform=android-$__APILevel --install-dir=$__RootfsDir

# ACL build from source needs --stl=gnustl
echo "Generating standalone toolchain and rootfs with to $__RootfsDir.gnustl"
$__ToolchainDir/ndk/build/tools/make-standalone-toolchain.sh --arch=$__BuildArch --platform=android-$__APILevel --install-dir=$__RootfsDir.gnustl --stl=gnustl

# Install boost

# NOTE This only copies headers so header-only libraries will work
echo "Installing boost library (HEADER-ONLY)"

BOOST_VERSION=1_67_0
BOOST_BASENAME=boost_$BOOST_VERSION
wget -nv -nc https://dl.bintray.com/boostorg/release/1.67.0/source/$BOOST_BASENAME.tar.gz -O $__ToolchainDir/$BOOST_BASENAME.tar.gz

tar xzf $__ToolchainDir/$BOOST_BASENAME.tar.gz -C $__ToolchainDir
cp -rv $__ToolchainDir/$BOOST_BASENAME/boost $__RootfsDir/sysroot/usr/include

if [[ "$__ACL" == 1 ]]; then
    echo "Installing arm compute library"

    ACL_VERSION=18.03
    ACL_BASENAME=arm_compute-v$ACL_VERSION-bin-android
    wget -nv -nc https://github.com/ARM-software/ComputeLibrary/releases/download/v$ACL_VERSION/$ACL_BASENAME.tar.gz -O $__ToolchainDir/$ACL_BASENAME.tar.gz

    tar xzf $__ToolchainDir/$ACL_BASENAME.tar.gz -C $__ToolchainDir
    cp -rv $__ToolchainDir/$ACL_BASENAME/arm_compute $__RootfsDir/sysroot/usr/include
    cp -rv $__ToolchainDir/$ACL_BASENAME/include/* $__RootfsDir/sysroot/usr/include
    cp -rv $__ToolchainDir/$ACL_BASENAME/support $__RootfsDir/sysroot/usr/include
    cp -rv $__ToolchainDir/$ACL_BASENAME/util $__RootfsDir/sysroot/usr/include
    cp -rv $__ToolchainDir/$ACL_BASENAME/lib/android-arm64-v8a-cl/* $__RootfsDir/sysroot/usr/lib # TODO hardcoded path "arm64-v8a"
fi
