#
# Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

# define default
default: all

# define package name
PACKAGE_NAME := $(notdir $(shell pwd))

# define library prerequisites list
libllmod := src
make_dir := make
EXE_SOURCES = $(libllmod)

# define target_architecture
export TARGET_AARCH_VARS:= -march=x86-64

# define target name
export TARGET = linux-x86_64

# specify compiler
export CXX := clang++-16

.PHONY: all $(EXE_SOURCES) x86 all-android
all: $(EXE_SOURCES) x86 all-android

# Combined Targets
clean: clean_x86 clean_android

x86:
	$(call build_if_exists,$(libllmod),-$(MAKE) -f $(make_dir)/Makefile.linux-x86_64)
	$(call build_if_exists,test,$(CXX) -g -O0 -lllmod -I api -I $(QNN_SDK_ROOT)/include -L bin/x86_64-linux-clang test/simple_app.cpp -o bin/x86_64-linux-clang/test)


clean_x86:
	@rm -rf bin/x86_64-linux-clang/libllmod.so bin/x86_64-linux-clang/test obj/x86_64-linux-clang

# Android Targets

all-android: aarch64-android arm-android

aarch64-android: check_ndk
	$(call build_if_exists,$(libllmod),$(ANDROID_NDK_ROOT)/ndk-build APP_ALLOW_MISSING_DEPS=true APP_ABI="arm64-v8a" NDK_PROJECT_PATH=./ NDK_APPLICATION_MK=$(make_dir)/Application.mk APP_BUILD_SCRIPT=$(make_dir)/Android.mk)
	mkdir -p bin/aarch64-android
	mv libs/arm64-v8a/* bin/aarch64-android
	rm -rf libs
	cp ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV75Stub.so ${QNN_SDK_ROOT}/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so bin/aarch64-android/

arm-android: check_ndk
	$(call build_if_exists,$(libllmod),$(ANDROID_NDK_ROOT)/ndk-build APP_ALLOW_MISSING_DEPS=true APP_ABI="armeabi-v7a" NDK_PROJECT_PATH=./ NDK_APPLICATION_MK=$(make_dir)/Application.mk APP_BUILD_SCRIPT=$(make_dir)/Android.mk)
	mkdir -p bin/arm-android
	mv libs/armeabi-v7a/* bin/arm-android
	rm -rf libs

clean_android: check_ndk clean_arm-android clean_aarch64-android

clean_arm-android:
	@rm -rf bin/arm-android/libllmod.so bin/arm-android/test
	@rm -rf obj/local/armeabi-v7a

clean_aarch64-android:
	@rm -rf bin/aarch64-android/libllmod.so bin/aarch64-android/test
	@rm -rf obj/local/arm64-v8a


# utilities
# Syntax: $(call build_if_exists <dir>,<cmd>)
build_if_exists = $(if $(wildcard $(1)),$(2),$(warning WARNING: $(1) does not exist. Skipping Compilation))

check_ndk:
ifeq ($(ANDROID_NDK_ROOT),)
	$(error ERROR: ANDROID_NDK_ROOT not set, skipping compilation for Android platform(s).)
endif
