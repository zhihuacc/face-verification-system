/*
 * timer.h
 *
 *  Created on: 30 Apr, 2013
 *      Author: harvey
 */

#ifndef TIMER_H_
#define TIMER_H_

#include <time.h>

class MyTimer {
public:
	void start();
	void stop();
	struct timespec diff();
private:
	struct timespec start_time_xyz;
	struct timespec stop_time_;
};


#endif /* TIMER_H_ */
