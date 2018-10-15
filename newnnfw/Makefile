TARGET_ARCH?=$(shell uname -p)
BUILD_TYPE?=Debug
CROSS_BUILD?=0
HOST_OS?=linux
TARGET_OS?=linux
NPROCS:=1
PARALLEL_BUILD?=1
## TODO: fix obs build break
OBS_BUILD?=OFF
COVERAGE_BUILD?=0
BENCHMARK_ACL_BUILD?=0
OPTIONS?=
UPDATE_MODULE?=1

# make TARGET and TYPE to lowercase
TARGET_ARCH_LC=$(shell echo $(TARGET_ARCH) | tr A-Z a-z)
BUILD_TYPE_LC=$(shell echo $(BUILD_TYPE) | tr A-Z a-z)
# we need base name 'arm` for all arm arch
TARGET_ARCH_BASE=$(TARGET_ARCH_LC)
ifneq (,$(findstring arm64,$(TARGET_ARCH_BASE)))
	# arm64 as target-arch comes from Android
	TARGET_ARCH_BASE=arm64
	ifdef ROOTFS_DIR
		ROOTFS_ARM64=$(ROOTFS_DIR)
		export ROOTFS_ARM64
	endif
	# For now Android is the only option for arm64
	TARGET_OS:=android
else ifneq (,$(findstring arm,$(TARGET_ARCH_BASE)))
	TARGET_ARCH_BASE=arm
	ifdef ROOTFS_DIR
		ROOTFS_ARM=$(ROOTFS_DIR)
		export ROOTFS_ARM
	endif
else ifneq (,$(findstring aarch64,$(TARGET_ARCH_BASE)))
	# aarch64 as target-arch comes from all except for Android
	TARGET_ARCH_BASE=aarch64
	ifdef ROOTFS_DIR
		ROOTFS_ARM64=$(ROOTFS_DIR)
		export ROOTFS_ARM64
	endif
endif
# Todo: we may set CROSS_BUILD=1 when ROOTFS_DIR is given
# the toolchain file, only for cross build
ifeq ($(CROSS_BUILD),1)
	TOOLCHAIN_FILE=cmake/config/config_$(TARGET_ARCH_LC)-$(TARGET_OS).cmake
	OPTION_TOOLCHAIN=-DCMAKE_TOOLCHAIN_FILE=$(TOOLCHAIN_FILE)
else
	OPTION_TOOLCHAIN=
endif

ifeq ($(COVERAGE_BUILD),1)
	OPTIONS+= -DCOVERAGE_BUILD=1
else
	OPTIONS+= -DCOVERAGE_BUILD=0
endif

ifeq ($(BENCHMARK_ACL_BUILD),1)
	OPTIONS+= -DBUILD_BENCHMARK_ACL=1
endif

ifeq ($(PARALLEL_BUILD),1)
	# Get number of processors (linux only for now)
	ifeq ($(HOST_OS),linux)
		NPROCS:=$(shell grep -c ^processor /proc/cpuinfo)
	endif
endif

WORKHOME=Product
WORKFOLDER=$(TARGET_ARCH_LC)-$(TARGET_OS).$(BUILD_TYPE_LC)
WORKDIR=$(WORKHOME)/$(WORKFOLDER)

BUILD_ROOT=$(WORKDIR)/obj
INSTALL_ROOT=$(WORKDIR)/out

BUILD_FOLDER=$(WORKFOLDER)/obj
INSTALL_FOLDER=$(WORKFOLDER)/out
BUILD_ALIAS=$(WORKHOME)/obj
INSTALL_ALIAS=$(WORKHOME)/out

INSTALL_PATH?=$(CURDIR)/$(WORKDIR)/out

TIMESTAMP_CONFIGURE=$(WORKDIR)/CONFIGURE
TIMESTAMP_BUILD=$(WORKDIR)/BUILD
TIMESTAMP_INSTALL=$(WORKDIR)/INSTALL

#
# for Build Arm Compute Library with SCONS
#
ACL_FOLDER=externals/acl
ACL_COMMAND=scons -j${NPROCS} neon=1 opencl=1 examples=0 embed_kernels=1 os=$(TARGET_OS)
ifeq ($(TARGET_ARCH_LC),armv7l)
	ACL_COMMAND+= arch=armv7a
	ACL_BUILD_OUT=armv7a-$(TARGET_OS)
else ifeq ($(TARGET_ARCH_LC),aarch64)
	ACL_COMMAND+= arch=arm64-v8a
	ACL_BUILD_OUT=arm64-v8a-$(TARGET_OS)
else ifeq ($(TARGET_ARCH_BASE),arm64)
	ACL_COMMAND+= arch=arm64-v8a
	ACL_BUILD_OUT=arm64-v8a-$(TARGET_OS)
endif
ifeq ($(TARGET_OS),android)
	ACL_COMMAND+= Werror=0
	ANDROID_GNUSTL_PATH=$(ROOTFS_ARM64)/bin:$(ROOTFS_ARM64)/aarch64-linux-android/bin:$$PATH
else
	ACL_COMMAND+= Werror=1
endif
ifeq ($(BUILD_TYPE_LC),debug)
	ACL_COMMAND+= debug=1 asserts=1
endif
ACL_FOLDER_NAME=$(ACL_BUILD_OUT).$(BUILD_TYPE_LC)
ACL_COMMAND+= build_dir=$(ACL_FOLDER_NAME)
ACL_FOLDER_BUILD=$(ACL_FOLDER)/build/$(ACL_FOLDER_NAME)

ifeq ($(OBS_BUILD),ON)
	UPDATE_MODULE=0
endif

all: build

###
### Command (public)
###
configure: configure_internal

build: build_internal

install: $(TIMESTAMP_INSTALL)

clean:
	rm -rf $(WORKDIR)

acl: internal_acl_build internal_acl_install

check: check_internal

runtime: runtime_build_internal

testbuild: test_build_internal

tflite: tflite_build_internal

###
### Command (internal)
###
configure_internal:
ifeq ($(UPDATE_MODULE),1)
	git submodule update --init --recursive
endif

	mkdir -p $(BUILD_ROOT)
	cmake -B$(CURDIR)/$(BUILD_ROOT) -H$(CURDIR) \
		-DCMAKE_INSTALL_PREFIX=$(INSTALL_PATH) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE_LC) -DTARGET_ARCH=$(TARGET_ARCH_LC) \
		-DHOST_OS=$(HOST_OS) \
		-DTARGET_OS=$(TARGET_OS) \
		-DOBS_BUILD=$(OBS_BUILD) \
		$(OPTION_TOOLCHAIN) \
		$(OPTIONS)
	touch $(TIMESTAMP_CONFIGURE)

build_internal: $(BUILD_ROOT)
	cd $(BUILD_ROOT) && make -j $(NPROCS) all
	rm -rf $(BUILD_ALIAS)
	ln -s $(BUILD_FOLDER) $(BUILD_ALIAS)
	touch $(TIMESTAMP_BUILD)

install_internal:
	cd $(BUILD_ROOT) && make install
	rm -rf $(INSTALL_ALIAS)
	ln -s $(INSTALL_FOLDER) $(INSTALL_ALIAS)
	touch $(TIMESTAMP_INSTALL)

internal_acl_build:
ifeq ($(UPDATE_MODULE),1)
	git submodule update --init --recursive
endif

ifeq ($(TARGET_OS),android)
	cd $(ACL_FOLDER) && CXX=clang++ CC=clang PATH=$(ANDROID_GNUSTL_PATH) $(ACL_COMMAND)
else
	cd $(ACL_FOLDER) && $(ACL_COMMAND)
endif

internal_acl_install:
	@mkdir -vp $(INSTALL_PATH)/lib
	@cp -v $(ACL_FOLDER_BUILD)/libarm_compute_core.so $(INSTALL_PATH)/lib/.
	@cp -v $(ACL_FOLDER_BUILD)/libarm_compute_graph.so $(INSTALL_PATH)/lib/.
	@cp -v $(ACL_FOLDER_BUILD)/libarm_compute.so $(INSTALL_PATH)/lib/.

external_acl:
ifneq ($(EXT_ACL_FOLDER),"")
	@mkdir -vp $(INSTALL_PATH)/lib
	@cp -v $(EXT_ACL_FOLDER)/libarm_compute_core.so $(INSTALL_PATH)/lib/.
	@cp -v $(EXT_ACL_FOLDER)/libarm_compute_graph.so $(INSTALL_PATH)/lib/.
	@cp -v $(EXT_ACL_FOLDER)/libarm_compute.so $(INSTALL_PATH)/lib/.
endif

ifeq ($(CROSS_BUILD),1)
check_internal: test_internal_cross
else
check_internal: test_internal
endif

# TODO: Enable tests on cross build
test_internal_cross:
	@echo "Can't do tests on cross build"

test_internal:
	tools/test_driver/test_driver.sh .

build_test_suite: install_internal
	@echo "packaging test suite"
	@rm -rf $(INSTALL_ROOT)/test-suite.tar.gz
	@tar -zcf test-suite.tar.gz tests/ tools/test_driver $(INSTALL_ALIAS) --dereference
	@mv test-suite.tar.gz $(INSTALL_ROOT)/.

build_coverage_suite: install_internal
	@echo "packaging test-coverage suite"
	@rm -rf $(INSTALL_ROOT)/coverage-suite.tar.gz
	@find Product -name "*.gcno" > include_lists.txt
	@tar -zcf coverage-suite.tar.gz tests/ tools/test_driver $(INSTALL_ALIAS) --dereference -T include_lists.txt
	@rm -rf include_lists.txt
	@mv coverage-suite.tar.gz $(INSTALL_ROOT)/.

runtime_build_internal: $(BUILD_ROOT)
	cd $(BUILD_ROOT) && make -j $(NPROCS) nnapi_pure_arm_compute
	rm -rf $(BUILD_ALIAS)
	ln -s $(BUILD_FOLDER) $(BUILD_ALIAS)
	touch $(TIMESTAMP_BUILD)

test_build_internal: $(BUILD_ROOT)
	# Build test
	cd $(BUILD_ROOT) && make -j $(NPROCS) nnapi_test
	# Build unittest
	cd $(BUILD_ROOT) && make -j $(NPROCS) runtime_run_android_nn_test
	rm -rf $(BUILD_ALIAS)
	ln -s $(BUILD_FOLDER) $(BUILD_ALIAS)
	touch $(TIMESTAMP_BUILD)

tflite_build_internal: $(BUILD_ROOT)
	# Build test
	cd $(BUILD_ROOT) && make -j $(NPROCS) tensorflow-lite gtest_main
	rm -rf $(BUILD_ALIAS)
	ln -s $(BUILD_FOLDER) $(BUILD_ALIAS)
	touch $(TIMESTAMP_BUILD)

###
### Timestamps
###
$(WORKDIR):
	mkdir -p $@

$(BUILD_ROOT): $(WORKDIR)
	make configure_internal

$(TIMESTAMP_CONFIGURE):
	make configure_internal

$(TIMESTAMP_BUILD): $(TIMESTAMP_CONFIGURE)
	make build_internal

$(TIMESTAMP_INSTALL): $(TIMESTAMP_BUILD)
	make install_internal
