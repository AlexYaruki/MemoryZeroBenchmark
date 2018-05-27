LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := bench
LOCAL_SRC_FILES := ../main.cpp
LOCAL_CPPFLAGS := -O0
include $(BUILD_EXECUTABLE)
