# Building TensorFlow and TOCO from source

You can build TensorFlow and tools including `TOCO` from source.
Please read
[Installing TensorFlow from Sources](https://www.tensorflow.org/install/install_sources)
for full description.

## Install Bazel

Follow [Installing Bazel](https://docs.bazel.build/versions/master/install.html)
- For Ubuntu, follow [Installing Bazel on Ubuntu](https://docs.bazel.build/versions/master/install-ubuntu.html)

These are the actual steps to install using apt package manager:
```
sudo apt-get install openjdk-8-jdk
```
```
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" \
| sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
```
```
sudo apt-get update && sudo apt-get install bazel
```
```
sudo apt-get upgrade bazel
```

## Install python packages

```
sudo apt-get install python-numpy python-dev python-pip python-wheel
```

## Configure

```
cd external/tensorflow
./configure
```

Select options like this page: https://www.tensorflow.org/install/install_sources#ConfigureInstallation

## Build with Bazel

```
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

If you have any problems while building, please fire an issue.

## Uninstall if already installed

You may skip this if you haven't installed
```
pip uninstall /tmp/tensorflow_pkg/tensorflow-1.6.0rc1-cp27-cp27mu-linux_x86_64.whl
```

## Install TensorFlow and tools

```
pip install /tmp/tensorflow_pkg/tensorflow-1.6.0rc1-cp27-cp27mu-linux_x86_64.whl --user
```

You should see installed `toco` at `~/.local/bin` folder.
