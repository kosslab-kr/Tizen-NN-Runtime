## Odroid-XU4 부팅하기

#### 필요한 파일 준비

1. 프리 부트로더 파일

   ```
   $ wget https://github.com/hardkernel/u-boot/raw/odroidxu3-v2012.07/sd_fuse/hardkernel_1mb_uboot/bl1.bin.hardkernel
   $ wget https://github.com/hardkernel/u-boot/raw/odroidxu3-v2012.07/sd_fuse/hardkernel_1mb_uboot/bl2.bin.hardkernel.1mb_uboot
   $ wget https://github.com/hardkernel/u-boot/raw/odroidxu3-v2012.07/sd_fuse/hardkernel_1mb_uboot/tzsw.bin.hardkernel
   ```

2. 최신 u-boot, 커널 다운로드

   ```
   http://download.tizen.org/snapshots/tizen/tv/latest/images/arm-wayland/tv-boot-armv7l-odroidxu3/
   ```

3. 다운로드한 파일 압축 풀기, u-boot 파일 이름 변경

   ```
   $ tar xvf tizen-tv_xxxxxxxx.x_tv-boot-armv7l-odroidxu3.tar.gz
   ```

4. "sd_fusing_xu4.sh" 파일에 다음의 내용을 저장

   ```
   #!/bin/bash
   
   declare FORMAT=""
   declare DEVICE=""
   
   # Binaires array for fusing
   declare -a FUSING_BINARY_ARRAY
   declare -i FUSING_BINARY_NUM=0
   
   declare CONV_ASCII=""
   declare -i FUS_ENTRY_NUM=0
   
   # binary name | part number | offset | bs
   declare -a PART_TABLE=(
   	"bl1.bin.hardkernel"		""	1	512
   	"bl2.bin.hardkernel.1mb_uboot"	""	31	512
   	"u-boot-mmc.bin"		""	63	512
   	"tzsw.bin.hardkernel"		""	2111	512
   	"params.bin"			""	6272	512
   	"boot.img"			1	0	512
   	"ramdisk.img"			1	0	512
   	"rootfs.img"			2	0	4M
   	"system-data.img"		3	0	4M
   	"user.img"			5	0	4M
   	"modules.img"			6	0	512
   	)
   
   declare -r -i PART_TABLE_COL=4
   declare -r -i PART_TABLE_ROW=${#PART_TABLE[*]}/${PART_TABLE_COL}
   
   # partition table support
   function get_index_use_name () {
   	local -r binary_name=$1
   
   	for ((idx=0;idx<$PART_TABLE_ROW;idx++)); do
   		if [ ${PART_TABLE[idx * ${PART_TABLE_COL} + 0]} == $binary_name ]; then
   			return $idx
   		fi
   	done
   
   	# return out of bound index
   	return $idx
   }
   
   # fusing feature
   function convert_num_to_ascii () {
   	local number=$1
   
   	CONV_ASCII=$(printf \\$(printf '%03o' $number))
   }
   
   function print_message () {
   	local color=$1
   	local message=$2
   
   	tput setaf $color
   	tput bold
   	echo ""
   	echo $message
   	tput sgr 0
   }
   
   function add_fusing_entry () {
   	local name=$1
   	local offset=$2
   	local size=$3
   
   	FUS_ENTRY_NUM=$((FUS_ENTRY_NUM + 1))
   
   	echo -n "$name" > entry_name
   	cat entry_name /dev/zero | head -c 32 >> entry
   
   	echo -n "" > entry_offset
   	for ((i=0; i < 4; i++))
   	do
   		declare -i var;
   		var=$(( ($offset >> (i*8)) & 0xFF ))
   		convert_num_to_ascii $var
   		echo -n $CONV_ASCII > tmp
   		cat tmp /dev/zero | head -c 1 >> entry_offset
   	done
   	cat entry_offset /dev/zero | head -c 4 >> entry
   
   	echo -n "" > entry_size
   	for ((i=0; i < 4; i++))
   	do
   		declare -i var;
   		var=$(( ($size >> (i*8)) & 0xFF ))
   		convert_num_to_ascii $var
   		echo -n $CONV_ASCII > tmp
   		cat tmp /dev/zero | head -c 1 >> entry_size
   	done
   	cat entry_size /dev/zero | head -c 4 >> entry
   
   	rm tmp
   	rm entry_name
   	rm entry_offset
   	rm entry_size
   }
   
   function fusing_image () {
   	local -r fusing_img=$1
   
   	# get binary info using basename
   	get_index_use_name $(basename $fusing_img)
   	local -r -i part_idx=$?
   
   	if [ $part_idx -ne $PART_TABLE_ROW ];then
   		local -r device=$DEVICE${PART_TABLE[${part_idx} * ${PART_TABLE_COL} + 1]}
   		local -r seek=${PART_TABLE[${part_idx} * ${PART_TABLE_COL} + 2]}
   		local -r bs=${PART_TABLE[${part_idx} * ${PART_TABLE_COL} + 3]}
   	else
   		echo "Not supported binary: $fusing_img"
   		return
   	fi
   
   	local -r input_size=`du -b $fusing_img | awk '{print $1}'`
   
   	print_message 2 "[Fusing $1]"
   
   	if [ "$(basename $fusing_img)" == "ramdisk.img" ]; then
   		umount $device
   		mkdir mnt_tmp
   		mount -t vfat $device ./mnt_tmp
   		cp -f $fusing_img ./mnt_tmp
   		sync
   		umount ./mnt_tmp
   		rmdir mnt_tmp
   		echo "fusing $fusing_img is done."
   	else
   		dd if=$fusing_img | pv -s $input_size | dd of=$device seek=$seek bs=$bs
   	fi
   
   	if [ $(basename $fusing_img) == "u-boot-mmc.bin" ];then
   		add_fusing_entry "u-boot" $seek 2048
   	fi
   }
   
   function fuse_image_tarball () {
   	local -r filepath=$1
   	local -r temp_dir="tar_tmp"
   
   	mkdir -p $temp_dir
   	tar xvf $filepath -C $temp_dir
   	cd $temp_dir
   
   	for file in *
   	do
   		fusing_image $file
   	done
   
   	cd ..
   	rm -rf $temp_dir
   }
   
   function check_binary_format () {
   	local -r binary=$1
   
   	case "$binary" in
   	*.tar | *.tar.gz)
   		fuse_image_tarball $binary
   		eval sync
   		;;
   	*)
   		fusing_image $binary
   		eval sync
   		;;
   	esac
   }
   
   function fuse_image () {
   	if [ "$FUSING_BINARY_NUM" == 0 ]; then
   		return
   	fi
   
   	# NOTE: to ensure ramdisk booting, ramdisk image should be copied after
   	# boot image is flashed into boot partition.
   	#
   	# This code guarantees that ramdisk image is flashed in the end of binaries.
   	local -i tmpval=$FUSING_BINARY_NUM-1
   	for ((fuse_idx = 0 ; fuse_idx < $FUSING_BINARY_NUM ; fuse_idx++))
   	do
   		local filename=${FUSING_BINARY_ARRAY[fuse_idx]}
   		local tmpname=""
   
   		case "$filename" in
   		*.tar | *.tar.gz)
   			if [ $fuse_idx -lt $tmpval ]; then
   				local tar_contents=`tar tvf $filename | awk 'BEGIN {FS=" "} {print $6}'`
   
   				for content in $tar_contents
   				do
   					if [ "$content" == "ramdisk.img" ]; then
   						tmpname=$filename
   						filename=${FUSING_BINARY_ARRAY[$tmpval]}
   						FUSING_BINARY_ARRAY[$tmpval]=$tmpname
   						break
   					fi
   				done
   			fi
   			check_binary_format $filename
   			;;
   		*)
   			if [ $fuse_idx -lt $tmpval ]; then
   				if [ "$filename" == "ramdisk.img" ]; then
   					tmpname=$filename
   					filename=${FUSING_BINARY_ARRAY[$tmpval]}
   					FUSING_BINARY_ARRAY[$tmpval]=$tmpname
   				fi
   			fi
   			check_binary_format $filename
   			;;
   		esac
   	done
   	echo ""
   }
   
   # partition format
   function mkpart_3 () {
   	# NOTE: if your sfdisk version is less than 2.26.0, then you should use following sfdisk command:
   	# sfdisk --in-order --Linux --unit M $DISK <<-__EOF__
   	#
   	# NOTE: sfdisk 2.26 doesn't support units other than sectors and marks --unit option as deprecated.
   	# The input data needs to contain multipliers (MiB) instead.
   	local version=`sfdisk -v | awk '{print $4}'`
   	local major=${version%%.*}
   	local version=${version:`expr index $version .`}
   	local minor=${version%%.*}
   	local sfdisk_new=0
   
   	if [ $major -gt 2 ];  then
   		sfdisk_new=1
   	else
   		if [ $major -eq 2 -a $minor -ge 26 ];  then
   			sfdisk_new=1
   		fi
   	fi
   
   	local -r DISK=$DEVICE
   	local -r SIZE=`sfdisk -s $DISK`
   	local -r SIZE_MB=$((SIZE >> 10))
   
   	local -r BOOT_SZ=64
   	local -r ROOTFS_SZ=3072
   	local -r DATA_SZ=512
   	local -r MODULE_SZ=32
   	if [ $sfdisk_new == 1 ]; then
   		local -r EXTEND_SZ=8
   	else
   		local -r EXTEND_SZ=4
   	fi
   
   	let "USER_SZ = $SIZE_MB - $BOOT_SZ - $ROOTFS_SZ - $DATA_SZ - $MODULE_SZ - $EXTEND_SZ"
   
   	local -r BOOT=boot
   	local -r ROOTFS=rootfs
   	local -r SYSTEMDATA=system-data
   	local -r USER=user
   	local -r MODULE=modules
   
   	if [[ $USER_SZ -le 100 ]]
   	then
   		echo "We recommend to use more than 4GB disk"
   		exit 0
   	fi
   
   	echo "========================================"
   	echo "Label          dev           size"
   	echo "========================================"
   	echo $BOOT"		" $DISK"1  	" $BOOT_SZ "MB"
   	echo $ROOTFS"		" $DISK"2  	" $ROOTFS_SZ "MB"
   	echo $SYSTEMDATA"	" $DISK"3  	" $DATA_SZ "MB"
   	echo "[Extend]""	" $DISK"4"
   	echo " "$USER"		" $DISK"5  	" $USER_SZ "MB"
   	echo " "$MODULE"		" $DISK"6  	" $MODULE_SZ "MB"
   
   	local MOUNT_LIST=`mount | grep $DISK | awk '{print $1}'`
   	for mnt in $MOUNT_LIST
   	do
   		umount $mnt
   	done
   
   	echo "Remove partition table..."
   	dd if=/dev/zero of=$DISK bs=512 count=16 conv=notrunc
   
   	if [ $sfdisk_new == 1 ]; then
   		sfdisk $DISK <<-__EOF__
   		4MiB,${BOOT_SZ}MiB,0xE,*
   		8MiB,${ROOTFS_SZ}MiB,,-
   		8MiB,${DATA_SZ}MiB,,-
   		8MiB,,E,-
   		,${USER_SZ}MiB,,-
   		,${MODULE_SZ}MiB,,-
   		__EOF__
   	else
   		sfdisk --in-order --Linux --unit M $DISK <<-__EOF__
   		4,$BOOT_SZ,0xE,*
   		,$ROOTFS_SZ,,-
   		,$DATA_SZ,,-
   		,,E,-
   		,$USER_SZ,,-
   		,$MODULE_SZ,,-
   		__EOF__
   	fi
   
   	mkfs.vfat -F 16 ${DISK}1 -n $BOOT
   	mkfs.ext4 -q ${DISK}2 -L $ROOTFS -F
   	mkfs.ext4 -q ${DISK}3 -L $SYSTEMDATA -F
   	mkfs.ext4 -q ${DISK}5 -L $USER -F
   	mkfs.ext4 -q ${DISK}6 -L $MODULE -F
   }
   
   function show_usage () {
   	echo "- Usage:"
   	echo "	sudo ./sd_fusing_xu4.sh -d <device> [-b <path> <path> ..] [--format]"
   }
   
   function check_partition_format () {
   	if [ "$FORMAT" != "2" ]; then
   		echo "-----------------------"
   		echo "Skip $DEVICE format"
   		echo "-----------------------"
   		return 0
   	fi
   
   	echo "-------------------------------"
   	echo "Start $DEVICE format"
   	echo ""
   	mkpart_3
   	echo "End $DEVICE format"
   	echo "-------------------------------"
   	echo ""
   }
   
   function check_args () {
   	if [ "$DEVICE" == "" ]; then
   		echo "$(tput setaf 1)$(tput bold)- Device node is empty!"
   		show_usage
   		tput sgr 0
   		exit 0
   	fi
   
   	if [ "$DEVICE" != "" ]; then
   		echo "Device: $DEVICE"
   	fi
   
   	if [ "$FUSING_BINARY_NUM" != 0 ]; then
   		echo "Fusing binary: "
   		for ((bid = 0 ; bid < $FUSING_BINARY_NUM ; bid++))
   		do
   			echo "  ${FUSING_BINARY_ARRAY[bid]}"
   		done
   		echo ""
   	fi
   
   	if [ "$FORMAT" == "1" ]; then
   		echo ""
   		echo "$(tput setaf 3)$(tput bold)$DEVICE will be formatted, Is it OK? [y/n]"
   		tput sgr 0
   		read input
   		if [ "$input" == "y" ] || [ "$input" == "Y" ]; then
   			FORMAT=2
   		else
   			FORMAT=0
   		fi
   	fi
   }
   
   function print_logo () {
   	echo ""
   	echo "[Odroid-XU3/4 downloader]"
   	echo "This version also supports Tizen 4.0."
   	echo ""
   }
   
   print_logo
   
   function add_fusing_binary() {
   	local declare binary_name=$1
   	FUSING_BINARY_ARRAY[$FUSING_BINARY_NUM]=$binary_name
   
   	FUSING_BINARY_NUM=$((FUSING_BINARY_NUM + 1))
   }
   
   declare -i binary_option=0
   
   while test $# -ne 0; do
   	option=$1
   	shift
   
   	case $option in
   	--f | --format)
   		FORMAT="1"
   		binary_option=0
   		;;
   	-d)
   		DEVICE=$1
   		binary_option=0
   		shift
   		;;
   	-b)
   		add_fusing_binary $1
   		binary_option=1
   		shift
   		;;
   	*)
   		if [ $binary_option == 1 ];then
   			add_fusing_binary $option
   		else
   			echo "Unkown command: $option"
   			exit
   		fi
   		;;
   	esac
   done
   
   check_args
   check_partition_format
   fuse_image
   ```

5. 위 파일을 실행가능하도록 수정

   ```
   $ chmod u+x sd_fusing_xu4.sh
   ```

6. pv tools 설치(fusing script를 사용하기 위해 필요)

   ``` 
   $ sudo apt-get install pv
   ```


#### Micro SD로 부팅하기

1. 데스크탑에 Micro SD 연결 후 device node 확인

```
Desktop$ sudo fdisk -l

..........
Partition table entries are not in disk order
Disk /dev/sdb: 32.0 GB, 32010928128 bytes
64 heads, 32 sectors/track, 30528 cylinders, total 62521344 sectors
Units = sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disk identifier: 0x00000000
Device Boot Start End Blocks Id System
/dev/sdb1 * 8192 139263 65536 e W95 FAT16 (LBA)
..........
```

2. Micro SD 포맷

※ /dev/sdX 를 적절한 장치 이름으로 변경 e.g. /dev/sdb

```
$ sudo dd if=/dev/zero of=/dev/sdX bs=1024MB count=8
```

3. Micro SD 파티션 나누기

```
$ sudo ./sd_fusing_xu4.sh -d /dev/sdX --format
```

4. Micro SD에 u-boot, boot 이미지, 커널 이미지 넣기

```
$ sudo ./sd_fusing_xu4.sh -d /dev/sdX -b tizen-tv_xxxxxxxx.x_tv-boot-armv7l-odroidxu3.tar.gz tizen-tv_xxxxxxxx.x_tv-wayland-armv7l-odroidu3.tar.gz
```

5. 프리 부트로더 넣기

```
$ sudo ./sd_fusing_xu4.sh -d /dev/sdX -b bl1.bin.hardkernel bl2.bin.hardkernel.1mb_uboot tzsw.bin.hardkernel
```

3. 오드로이드에 Micro SD 카드 연결, 부트 모드를 SD로 선택

4. 데스크탑과 Serial Console port 연결

   - `minicom` 설치

   ```
   $ sudo apt-get install minicom
   ```

   - `dialout` 그룹에 자신을 추가

   ```
   $ sudo vi /etc/group
   ```

   - `minicom` 사용하기 (`/dev/ttyUSB1` 은 달라질 수 있음)

   ```
   $ minicom --baudrate 115200 --device /dev/ttyUSB1
   ```

   - `CTRL-a z o` > `Serial port setup`을 사용해 dialog에 진입
   - Baud는 `115200-8N1`이어야 함
   - communication을 가능하게 하기 위해 configuration`Hardware Flow Control`을 `No`로

   - Mac이나 Windows에서 연결하는 경우 드라이버를 설치해야 함

     https://www.silabs.com/products/development-tools/software/usb-to-uart-bridge-vcp-drivers

    - Windows는 putty 사용

5. 전원 연결

6. root로 로그인

   - pwd `tizen`

#### 부트 후

- 팬 스피드 낮추기(0~255)

```
$ echo "100" > /sys/devices/platform/pwm-fan/hwmon/hwmon0/pwm1
```

​	재부팅 때마다 초기화 되므로 다시 설정해야 함

- Root file system 확장하기

  Root FS는 3G로 초기화. 파일을 설치하기에 충분하지 않을 수 있음.

  이를 해결하기 위해서는 Tizen root shell에서

  ```
  mount -o remount,rw /
  resize2fs /dev/mmcblk0p2
  sync
  ```

  재부팅

  ```
  reboot
  ```

  `df`로 확인

- Wide console

```
stty cols 200
```

- Target Device의 IP 주소 설정하기

  `connmanctl` 사용(Target Device에서)

  Step 1. 서비스 네임 받기

  - 이더넷 케이블 연결

  ```
  $ connmanctl services
  ```

  - 아래와 같은 것이 출력

  ```
  *AR Wired			ethernet_1a43230d5dfa_cable
  ```

  Step 2. IP 주소를 설정하기 위해 `config` 사용

  ```
  $ connmanctl config ethernet_1a43230d5dfa_cable --ipv4 manual 10.113.XXX.YYY 255.255.255.0 10.113.XXX.1
  
  $ connmanctl config ethernet_1a43230d5dfa_cable --nameservers 10.32.192.11 10.32.193.11
  ```

  XXX.YYY는 target board에 대한 자신의 주소



#### SDB로 연결하기

```
$ sdb connect 10.113.XXX.YYY
```

아래와 같은 것이 출력

```
\* Server is not running. Start it now on port 26099 *
\* Server has started successfully *
connecting to 10.113.xxx.yyy:26101 ...
connected to 10.113.xxx.yyy:26101
```

`sdb devices`

```
$ sdb devices

List of devices attached 

10.113.xxx.yyy:26101    device      xu3
```

xu3 이미지와 같은 것을 사용했으므로 xu3이라 출력됨