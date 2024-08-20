# ==============================================================================
#
#  Copyright (c) 2020, 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ===============================================================

LOCAL_PATH := $(call my-dir)
SUPPORTED_TARGET_ABI := arm64-v8a armeabi-v7a x86 x86_64

#============================ Verify Target Info and Application Variables =========================================
ifneq ($(filter $(TARGET_ARCH_ABI),$(SUPPORTED_TARGET_ABI)),)
    ifneq ($(APP_STL), c++_shared)
        $(error Unsupported APP_STL: "$(APP_STL)")
    endif
else
    $(error Unsupported TARGET_ARCH_ABI: '$(TARGET_ARCH_ABI)')
endif

#============================ Define Common Variables ===============================================================
# Include paths
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../api/
PACKAGE_C_INCLUDES += -I $(QNN_SDK_ROOT)/include/QNN
PACKAGE_C_INCLUDES += -I $(QNN_SDK_ROOT)/share/QNN/converter/jni

#========================== Define OpPackage Library Build Variables =============================================
include $(CLEAR_VARS)
LOCAL_C_INCLUDES               := $(PACKAGE_C_INCLUDES)
MY_SRC_FILES                   := $(wildcard $(LOCAL_PATH)/../src/*.cc) $(wildcard $(LOCAL_PATH)/../src/*.cpp) 
LOCAL_MODULE                   := llmod
LOCAL_SRC_FILES                := $(subst make/,,$(MY_SRC_FILES))
LOCAL_LDLIBS                   := -lGLESv2 -lEGL
include $(BUILD_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_C_INCLUDES               := -I $(LOCAL_PATH)/../api/
MY_SRC_FILES                   := $(LOCAL_PATH)/../test/simple_app.cpp
LOCAL_MODULE                   := simple_app
LOCAL_SRC_FILES                := $(subst make/,,$(MY_SRC_FILES))
LOCAL_LDLIBS                   := -lGLESv2 -lEGL
LOCAL_SHARED_LIBRARIES         := llmod 
include $(BUILD_EXECUTABLE)
