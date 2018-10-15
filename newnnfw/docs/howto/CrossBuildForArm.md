# Cross building for ARM

## Prepare Ubuntu RootFS

Install required packages

```
sudo apt-get install qemu qemu-user-static binfmt-support debootstrap
```

Use `build_rootfs.sh` script to prepare Root File System. You should have `sudo`

```
sudo ./tools/cross/build_rootfs.sh arm
```
- supports `arm`(default) and `arm64` architecutre for now
- supports `xenial`(default) and `trusty` release

To see the options,
```
./tools/cross/build_rootfs.sh -h
```

RootFS will be prepared at `tools/cross/rootfs/arm` folder.

## Prepare RootFS at alternative folder

Use `ROOTFS_DIR` to a full path to prepare at alternative path.

```
ROOTFS_DIR=/home/user/rootfs/arm-xenial sudo ./tools/cross/build_rootfs.sh arm
```

## Using proxy

If you need to use proxy server while building the rootfs, use `--setproxy` option.

```
# for example,
sudo ./tools/cross/build_rootfs.sh arm --setproxy="1.2.3.4:8080"
# or
sudo ./tools/cross/build_rootfs.sh arm --setproxy="proxy.server.com:8888"
```

This will put `apt` proxy settings in `rootfs/etc/apt/apt.conf.d/90proxy` file
for `http`, `https` and `ftp` protocol.

## Install ARM Cross Toolchain

We recommend you have g++ >= 6 installed on your system because NN generated tests require it.

On Ubuntu 16.04 or older, follow the next steps:

```
cd ~/your/path
wget https://releases.linaro.org/components/toolchain/binaries/7.2-2017.11/arm-linux-gnueabihf/gcc-linaro-7.2.1-2017.11-x86_64_arm-linux-gnueabihf.tar.xz
tar xvf gcc-linaro-7.2.1-2017.11-x86_64_arm-linux-gnueabihf.tar.xz
echo 'PATH=~/your/path/gcc-linaro-7.2.1-2017.11-x86_64_arm-linux-gnueabihf/bin:$PATH' >> ~/.bashrc
```

On Ubuntu 18.04 LTS, you can install using `apt-get`.
Choose g++ version whatever you prefer: 6, 7 or 8.

```
sudo apt-get install g++-{6,7,8}-arm-linux-gnueabihf
```

Make sure you get `libstdc++.so` updated on your target with your new toolchain's corresponding one.

For example, if you installed gcc-linaro-7.2.1-2017.11 above, do

```
wget https://releases.linaro.org/components/toolchain/binaries/7.2-2017.11/arm-linux-gnueabihf/runtime-gcc-linaro-7.2.1-2017.11-arm-linux-gnueabihf.tar.xz
tar xvf runtime-gcc-linaro-7.2.1-2017.11-arm-linux-gnueabihf.tar.xz
```

Then, copy `libstdc++.so.6.0.24` into `/usr/lib/arm-linux-gnueabihf`, and update symbolic links on your device.

## Build and install ARM Compute Library

```
TARGET_ARCH=armv7l make acl
```
Mostly you only need once of ACL build. This will build and install to `Product/(target_arch-os)/out/bin` folder.
- this is required for ARM on Ubuntu

## Build nnfw

Give `TARGET_ARCH` variable to set the target architecture

```
CROSS_BUILD=1 TARGET_ARCH=armv7l make all install
```
- supports `armv7l` and `aarch64` for now

If you used `ROOTFS_DIR` to prepare in alternative folder, you should also give this to makefile.

```
ROOTFS_DIR=ROOTFS_ARM=/path/to/your/rootfs/arm \
CROSS_BUILD=1 TARGET_ARCH=armv7l make all install
```

## Run test

```
 ./tools/test_driver/test_driver.sh --artifactpath=.
```
