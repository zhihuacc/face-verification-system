/*
 * timer.cpp
 *
 *  Created on: 30 Apr, 2013
 *      Author: harvey
 */

#include "my_timer.h"

void MyTimer::start() {
    clock_gettime(CLOCK_MONOTONIC, &start_time_xyz);
}

void MyTimer::stop() {
    clock_gettime(CLOCK_MONOTONIC, &stop_time_);
}

struct timespec MyTimer::diff() {
    struct timespec elapse;

    if (stop_time_.tv_nsec - start_time_xyz.tv_nsec < 0) {
    	elapse.tv_sec = stop_time_.tv_sec - start_time_xyz.tv_sec - 1;
    	elapse.tv_nsec = 1000000000 + stop_time_.tv_nsec - start_time_xyz.tv_nsec;
    } else {
    	elapse.tv_sec = stop_time_.tv_sec - start_time_xyz.tv_sec;
    	elapse.tv_nsec = stop_time_.tv_nsec - start_time_xyz.tv_nsec;
    }

    return elapse;
}


