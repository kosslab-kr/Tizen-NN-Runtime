# About

This will describe how to flash microSD with Tizen-4.0 for ODroid XU4.

Host environment is Ubuntu 16.04

# Download files

## Images

Boot
- https://download.tizen.org/snapshots/tizen/unified/latest/images/standard/tv-boot-armv7l-odroidxu3/
- download the biggest file

Root FS
- https://download.tizen.org/snapshots/tizen/unified/latest/images/standard/tv-wayland-armv7l-odroidu3/
- download the biggest file

U-Boot images
```
wget https://github.com/hardkernel/u-boot/raw/odroidxu3-v2012.07/sd_fuse/hardkernel_1mb_uboot/bl1.bin.hardkernel
wget https://github.com/hardkernel/u-boot/raw/odroidxu3-v2012.07/sd_fuse/hardkernel_1mb_uboot/bl2.bin.hardkernel.1mb_uboot
wget https://github.com/hardkernel/u-boot/raw/odroidxu3-v2012.07/sd_fuse/hardkernel_1mb_uboot/tzsw.bin.hardkernel
```

You also need `u-boot-mmc.bin` that is inside `tizen-unified_20180425.2_tv-boot-armv7l-odroidxu3.tar.gz` file.
```
tar xvf tizen-unified_20180425.2_tv-boot-armv7l-odroidxu3.tar.gz u-boot-mmc.bin
```


## Flashing script

Download `sd_fusing_xu4-u1604.sh` from https://github.sec.samsung.net/RS7-RuntimeNTools/TizenTools/tree/master/sd_fusing_xu4

This file is modified to work on Ubuntu 16.04.

You can download original (What I got in the first place) file as `sd_fusing_xu4.sh`

Make it executable
```
chmod u+x sd_fusing_xu4-u1604.sh
```


## Files

You should see like this
```
-rw-rw-r-- 1 maxwell maxwell     15616 Mar 23 17:11 bl1.bin.hardkernel
-rw-rw-r-- 1 maxwell maxwell     14592 Mar 23 17:10 bl2.bin.hardkernel.1mb_uboot
-rw-rw-r-- 1 maxwell maxwell   9290646 Apr 26 02:35 tizen-unified_20180425.2_tv-boot-armv7l-odroidxu3.tar.gz
-rw-rw-r-- 1 maxwell maxwell 346530499 Apr 26 02:59 tizen-unified_20180425.2_tv-wayland-armv7l-odroidu3.tar.gz
-rw-rw-r-- 1 maxwell maxwell    262144 Mar 23 17:11 tzsw.bin.hardkernel
-rwxr-xr-x 1 maxwell maxwell   1048576 Apr 26 02:35 u-boot-mmc.bin*
```


# Flash

Host environment
- Ubuntu 16.04
- microSD connected through USB Reader as `/dev/sdd` file.

## Flash boot files

Give `--format` if it's a new flash memory.
```
sudo ./sd_fusing_xu4-u1604.sh --format \
-d /dev/sdd \
-b bl1.bin.hardkernel bl2.bin.hardkernel.1mb_uboot tzsw.bin.hardkernel u-boot-mmc.bin
```
Change `/dev/sdd` to your configuration.

You will be asked to confirm format when used `--format`. Please type `y` to continue.
```
/dev/sdd will be formatted, Is it OK? [y/n]
y
```

You can omit `--format` from the second time and followings.
```
sudo ./sd_fusing_xu4-u1604.sh \
-d /dev/sdd \
-b bl1.bin.hardkernel bl2.bin.hardkernel.1mb_uboot tzsw.bin.hardkernel u-boot-mmc.bin
```
`--format` option will, 1) delete current partition 2) create new partition table, 3) format each partitions.

- If you meet `./sd_fusing_xu4-u1604.sh: line 147: pv: command not found` message and want to remove this message, install pv package by `sudo apt-get install pv`

## Flash image files
```
sudo ./sd_fusing_xu4-u1604.sh -d /dev/sdd \
-b tizen-unified_20180425.2_tv-boot-armv7l-odroidxu3.tar.gz \
tizen-unified_20180425.2_tv-wayland-armv7l-odroidu3.tar.gz
```

# Boot with Tizen 4.0

Follow the steps

Step 1.
- Take out eMMC memory card if you have any

Step 2. 
- Plug-In microSD with Tizen 4.0

Step 3. Set boot switch
- Refer https://wiki.odroid.com/odroid-xu4/hardware/hardware
- Set `Boot mode selector` switch on the bottom of the board to `uSD`

Step 4. Connect Serial Console port with USB of Host computer
- Install `minicom`
```
sudo apt-get install minicom
```
- Add yourself to the group `dialout`
   - `sudo vi /etc/group`
- Use serial terminal program like `minicom` (note that `/dev/ttyUSB1` might be different in your environment.)
```
minicom --baudrate 115200 --device /dev/ttyUSB1
```
- Use `CTRL-a z o` > `Serial port setup` to enter the dialog
- Baud should be `115200-8N1`
- Set configuration `Hardware Flow Control` to `No` to enable communication(keyboard typing..)
- `Save setup as dfl` in configuration
- If you are connecting from Windows or Mac my need to install the driver
   - https://www.silabs.com/products/development-tools/software/usb-to-uart-bridge-vcp-drivers
   - Use `PuTTY` for Windows.

Step 5. Connect Power
- You should see the boot logs...

Step 6. Login root
- login `root` pwd `tizen`

# After boot

## Slow down the fan speed

If the fan noise is disturbing, you can slow down a little.

```
echo "100" > /sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1
```
This will slow down the speed to 100. Range is from 0 to 255. "0" to make it stop. "255" for maximum speed.

This value resets after reboot so may have to set the value every time you reboot.

## Expand root file system

Default Root FS is 3G but the size shows about the size of the image file, about 700MB.

There would be not enough space to install files. To overcome this do the following in Tizen root shell.

```
mount -o remount,rw /
resize2fs /dev/mmcblk0p2
sync
```
And reboot
```
reboot
```

`df` before and after
```
Filesystem           1K-blocks      Used Available Use% Mounted on
/dev/root               754716    721228      8764  99% /
```
to
```
Filesystem           1K-blocks      Used Available Use% Mounted on
/dev/root              3031952    724724   2282504  25% /
```


## Wide console

```
stty cols 200
```

## Setting IP Address of Target Device

Use `connmanctl`

**CAUTION** PLEASE DO THIS IN YOUR TARGET DEVICE. RUNNING THIS IN YOUR HOST MAY DAMAGE.

Step 1. Get the service name
- You first need to connect Ethernet cable.
```
connmanctl services
```
Will drop something like this
```
*AR Wired                ethernet_1a43230d5dfa_cable
```

Step 2. Use `config` to set the IP address
```
connmanctl config ethernet_1a43230d5dfa_cable --ipv4 manual 10.113.XXX.YYY 255.255.255.0 10.113.XXX.1
connmanctl config ethernet_1a43230d5dfa_cable --nameservers 10.32.192.11 10.32.193.11
```
where `XXX.YYY` is your address for the target board.

Setting for proxy can be done with connmanctl but don't know how to check.
```
connmanctl config ethernet_1a43230d5dfa_cable --proxy manual http://10.112.1.184:8080/
```
You can use environment variable but still don't know how to check.


This information remains after reboot.

# Connecting with SDB

Default Tizen image has running SDBD in the device with default port (26101).

In your Linux or Windows with `sdb` command,
```
sdb connect 10.113.XXX.YYY
```
Result will be something like
```
* Server is not running. Start it now on port 26099 *
* Server has started successfully *
connecting to 10.113.xxx.yyy:26101 ...
connected to 10.113.xxx.yyy:26101
```
With `sdb devices`,
```
sdb devices
List of devices attached 
10.113.xxx.yyy:26101  	device    	xu3
```
It comes up with `xu3` as our `xu4` also uses same image `xu3` image.


# Known issue
- `ls -al` of root folder shows strange output.

# Reference
- https://wiki.tizen.org/Quick_guide_for_odroidxu4
- and the mail got from "김석원님"
- https://magazine.odroid.com/wp-content/uploads/odroid-xu4-user-manual.pdf
   - https://magazine.odroid.com/odroid-xu4
