## How to use XU4 with Ubuntu 16.04

Ref: https://wiki.odroid.com/odroid-xu4/odroid-xu4

eMMC card pre-installed Ubuntu 16.04

Preparation for IO via serial cable
- Refer to `minicom` section in xu4_tizen.md
- To find the name of serial device, plug your odroid into your host machine and power it on. Then, run the following on your host:
	```
	$ dmesg | grep tty
	[    0.000000] console [tty0] enabled
	[322282.017985] usb 2-1: cp210x converter now attached to ttyUSB0
	```
- Use `CTRL-a z o` > `Serial port setup` to enter the dialog
- Set configuration `Serial Device` to `/dev/ttyUSB0` for the name of serial device
- Baud should be `115200-8N1`
- Set configuration `Hardware Flow Control` to `No` to enable communication(keyboard typing..)

Connect
- Connect eMMC to bottom of the board
- Connect Serial Console to Host USB
- Connect power and boot

Login with serial console. you can login with `root` or default `odroid` account
- `root` password: `odroid`
- `odroid `password: `odroid`

Set ethernet
`/etc/network/interfaces`
```
# interfaces(5) file used by ifup(8) and ifdown(8)
# Include files from /etc/network/interfaces.d:
source-directory /etc/network/interfaces.d

auto lo eth0
iface lo inet loopback

iface eth0 inet static
	address 10.113.xxx.yyy
	netmask 255.255.255.0
	network 10.113.xxx.0
	broadcast 10.113.xxx.255
	gateway 10.113.xxx.1
	dns-nameservers 10.32.192.11 10.32.193.11 8.8.8.8
```
Change `xxx.yyy` to your IP address.

Reboot and login with SSH

### Add proxy settings

Add `/etc/apt/apt.conf.d/90proxies`
```
Acquire::http::proxy "http://10.112.1.184:8080/";
Acquire::https::proxy "http://10.112.1.184:8080/";
Acquire::ftp::proxy "ftp://10.112.1.184:8080/";
```

Add `/etc/profile.d/proxy.sh`
```
#!/bin/bash

# Proxy
export HTTP_PROXY=http://10.112.1.184:8080/
export HTTPS_PROXY=https://10.112.1.184:8080/
```

### Update and install programs

```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install vim nfs-common
```

### MALI GPU driver

Driver files are pre-installed in eMMC as follows
```
odroid@odroid:/usr/lib/arm-linux-gnueabihf/mali-egl$ ll
total 20136
drwxr-xr-x   2 root root     4096 Aug 20  2017 ./
drwxr-xr-x 106 root root    90112 Mar 26 08:32 ../
-rw-r--r--   1 root root       38 Apr 30  2017 ld.so.conf
-rwxr-xr-x   1 root root     2752 Apr 30  2017 libEGL.so*
lrwxrwxrwx   1 root root        9 Apr 30  2017 libEGL.so.1 -> libEGL.so*
lrwxrwxrwx   1 root root        9 Apr 30  2017 libEGL.so.1.4 -> libEGL.so*
-rwxr-xr-x   1 root root     2752 Apr 30  2017 libGLESv1_CM.so*
lrwxrwxrwx   1 root root       15 Apr 30  2017 libGLESv1_CM.so.1 -> libGLESv1_CM.so*
lrwxrwxrwx   1 root root       15 Apr 30  2017 libGLESv1_CM.so.1.1 -> libGLESv1_CM.so*
-rwxr-xr-x   1 root root     2752 Apr 30  2017 libGLESv2.so*
lrwxrwxrwx   1 root root       12 Apr 30  2017 libGLESv2.so.2 -> libGLESv2.so*
lrwxrwxrwx   1 root root       12 Apr 30  2017 libGLESv2.so.2.0 -> libGLESv2.so*
-rwxr-xr-x   1 root root 20493444 May  8  2017 libmali.so*
-rwxr-xr-x   1 root root     2752 Apr 30  2017 libOpenCL.so*
lrwxrwxrwx   1 root root       12 Apr 30  2017 libOpenCL.so.1 -> libOpenCL.so*
lrwxrwxrwx   1 root root       12 Apr 30  2017 libOpenCL.so.1.1 -> libOpenCL.so*
```
