# ==============================================================================
#
#  Copyright (c) 2020, 2022-2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ===============================================================

APP_ABI      := arm64-v8a armeabi-v7a
APP_STL      := c++_shared
APP_PLATFORM := android-24

ifdef LIBLLMOD_DEBUG
APP_OPTIM 	 := debug
APP_CPPFLAGS += -std=c++20 -O0 -g -DLIBLLMOD_DEBUG=1 -Wall -Werror -fexceptions -DNOTHREADS
else
APP_CPPFLAGS += -std=c++20 -O3 -Wall -Werror -fexceptions -fvisibility=hidden -DNOTHREADS -DLIBLLMOD_API="__attribute__((visibility(\"default\")))"
endif
APP_LDFLAGS  += -lc -lm -ldl -llog
