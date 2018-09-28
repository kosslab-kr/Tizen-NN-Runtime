#! /bin/bash

rpm -Uvh libncurses-5.9-7.1.armv7l.rpm
rpm -Uvh info-4.13a-1.6.armv7l.rpm
rpm -Uvh binutils-2.25.0-2015.01.1.8.armv7l.rpm
rpm -Uvh libasan-4.9.2-2015.02.1.11.armv7l.rpm
rpm -Uvh libatomic-4.9.2-2015.02.1.11.armv7l.rpm
rpm -Uvh libitm-4.9.2-2015.02.1.11.armv7l.rpm
rpm -Uvh libubsan-4.9.2-2015.02.1.11.armv7l.rpm
rpm -Uvh glibc-headers-2.20-2014.11.1.10.armv7l.rpm glibc-2.20-2014.11.1.10.armv7l.rpm --nodeps --force
rpm -Uvh cpp-4.9.2-2015.02.1.11.armv7l.rpm
rpm -Uvh glibc-devel-2.20-2014.11.1.10.armv7l.rpm
rpm -Uvh gcc-4.9.2-2015.02.1.11.armv7l.rpm
rpm -Uvh libstdc++-devel-4.9.2-2015.02.1.11.armv7l.rpm
rpm -Uvh gcc-c++-4.9.2-2015.02.1.11.armv7l.rpm
