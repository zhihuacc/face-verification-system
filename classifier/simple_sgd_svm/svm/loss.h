/*
 * loss.h
 *
 *  Created on: 29 Apr, 2013
 *      Author: harvey
 */

#ifndef LOSS_H_
#define LOSS_H_


class HingeLoss {
public:
   static double loss(double wx, double y);
   static double dloss(double wx, double y);
};



#endif /* LOSS_H_ */
