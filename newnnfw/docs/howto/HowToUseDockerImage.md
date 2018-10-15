# How to use docker image of nnfw

We have a docker image to build `nnfw` repo.

This docker image is built from https://github.sec.samsung.net/STAR/nnfw/blob/master/docker/Dockerfile and based on Ubuntu 16.04.
And prebuilt docker image is available from Samsung private docker registry.

This document describes how to use prebuilt docker image when developing `nnfw`.

## How to install docker

Follow [Installing Docker](https://docs.docker.com/)

- For Ubuntu, follow [Installing Docker on Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

These are the actual steps to install using apt package manager:
```
$ sudo apt-get install \
     apt-transport-https \
     ca-certificates \
     curl \
     software-properties-common
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo apt-key fingerprint 0EBFCD88
```
```
$ sudo add-apt-repository \
     "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) \
	 stable"
$ sudo apt-get update
```
```
$ sudo apt-get install docker-ce
```

## Configure docker daemon

1. Set HTTP/HTTPS proxy

    * For Ubuntu, follow [Setting HTTP/HTTPS proxy environment variables](https://docs.docker.com/v17.09/engine/admin/systemd/#httphttps-proxy)

If you are behind an HTTP or HTTPS proxy server, you will need to add this configuration in the Docker systemd service file.
These are the actual steps to set an HTTP/HTTPS proxy environment variable:
```
$ sudo mkdir -p /etc/systemd/system/docker.service.d
$ sudo vi etc/systemd/system/docker.service.d/http-proxy.conf
```
```
[Service]
Environment="HTTP_PROXY=http://10.112.1.184:8080/" "HTTPS_PROXY=https://10.112.1.184:8080/" "NO_PROXY=localhost,127.0.0.1"
```
```
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker
$ systemctl show --property=Environment docker
```

2. Edit configuration file of docker daemon

First you have to add Samsung private docker reigstry to your docker daemon.
Depending on your docker daemon installed, there are two ways of configuration.


If there is a `/etc/default/docker`, please edit the file as below.
```
$ sudo vi /etc/default/docker

DOCKER_OPTS="--insecure-registry docker.sec.samsung.net:5000"
```

If there is a `/etc/docker/daemon.json`, please edit the file as below.
```
{
  ...,
  "insecure-registries": [..., "docker.sec.samsung.net:5000"]
}
```

3. Then restart docker daemon as below.

```
$ sudo service docker restart     // Ubuntu 14.04

or

$ sudo systemctl restart docker   // Ubuntu 16.04
```

## Install docker image of `nnfw`

Let's pull docker image for `nnfw` repo and tag it to `nnfw_docker:latest`

```
$ docker pull docker.sec.samsung.net:5000/star/nnfw/nnfw_docker:1.5
$ docker tag docker.sec.samsung.net:5000/star/nnfw/nnfw_docker:1.5 nnfw_docker:latest
```

If you would like to build `nnfw` tizen package using gbs, pull `nnfw_docker_tizen`.
```
$ docker pull docker.sec.samsung.net:5000/star/nnfw/nnfw_docker_tizen:1.2
$ docker tag docker.sec.samsung.net:5000/star/nnfw/nnfw_docker_tizen:1.2 nnfw_docker_tizen:latest
```

## Use docker image to build `nnfw`
Three different targets for `nnfw` can be built using docker image.

1. Build `nnfw` for `x86_64` target
```
$ cd nnfw
$ docker run --rm -v $(pwd):/opt/nnfw -w /opt/nnfw nnfw_docker make install
```
or use `docker_run_test.sh` for convenience as below.
```
$ cd nnfw
$ ./run docker_run_test.sh
```
You can find built artifacts at `nnfw/Product/x86_64-linux.debug`.

2. Cross build `nnfw` for ARM on x86_64 host

You should prepare RootFS, following [Cross Building for ARM](https://github.sec.samsung.net/STAR/nnfw/blob/master/docs/howto/CrossBuildForArm.md) except ACL build and cross build steps. Then execute below commands. If your RootFS directory is different with below directory, change it to correct path and ensure the path is absolute.
```
$ cd nnfw
$ ROOTFS_DIR=$(pwd)/tools/cross/rootfs/arm \
./run docker_build_cross_arm_ubuntu.sh
```
You can find built artifacts at `nnfw/Product/armv7l-linux.debug/`.

3. Build `nnfw` for Tizen ARM package on x86_64 host
```
$ cd nnfw
$ ./run docker_gbs_build.sh
```
You can find built artifacts at `Product/out/rpm`.
