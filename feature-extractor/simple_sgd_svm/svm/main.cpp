/*
 * main.cpp
 *
 *  Created on: 27 Apr, 2013
 *      Author: harvey
 */

#include <string>
#include <cstdlib>

//#include "sgd_svm.h"

//int
//parse(int argc, const char **argv, string &training_file_name, string &test_file_name, double &lambda, int &t, double &batch_size_ratio, double &avg_start, int &max_train)
//{
//    for (int i=1; i<argc; i++) {
//        const char *arg = argv[i];
//        if (arg[0] != '-')
//        {
//            if (training_file_name.empty())
//              training_file_name = arg;
//            else if (test_file_name.empty())
//        	  test_file_name = arg;
//        } else {
//            while (arg[0] == '-')
//              arg += 1;
//            string opt = arg;
//            if (opt == "lambda" && i+1<argc) {
//                lambda = atof(argv[++i]);
//            } else if (opt == "t" && i + 1 < argc) {
//            	t = atoi(argv[++i]);
//            } else if (opt == "avg" && i + 1 < argc) {
//            	avg_start = atof(argv[++i]);
//            } else if (opt == "batch" && i + 1 < argc) {
//            	batch_size_ratio = atof(argv[++i]);
//            } else if (opt == "maxtrain" && i + 1 < argc) {
//            	max_train = atoi(argv[++i]);
//            } else {
//            	cout << "Unrecognized options" << endl;
//            }
//        }
//    }
//
//    if (training_file_name.empty()) {
//    	cout << "need a training file." << endl;
//    	return -1;
//    }
//
////    cout << "Learning Parameters:" << endl;
////    cout << "    Lamda: " << lambda << endl;
////    cout << "    ETA0: "  << eta0 << endl;
////    cout << "    Iterations: " << t << endl;
////    cout << "    Averaging start ratio: " << avg_start << endl;
////    cout << "    Max training rows:" << max_train << endl;
////    cout << "    Batch size ratio: " << batch_size_ratio << endl;
//
//    return 0;
//}

//int main(int argc, const char **argv) {
//    string training_file_name;
//    string test_file_name;
//    double lambda = 1;
//    int t = 100000;
//    double avg_start_ratio = 1.0;
//    double batch_size_ratio = 0.00001;
//    int max_train = 100000;
//
//    int ret = parse(argc, argv, training_file_name, test_file_name, lambda, t, batch_size_ratio, avg_start_ratio, max_train);
//    if (ret != 0) {
//    	return -1;
//    }
//
//    SgdSvm svm(lambda, t, batch_size_ratio, avg_start_ratio);
//
//    svm.LoadTrainingSampleFile(training_file_name, max_train);
//    svm.LoadTestSampleFile(test_file_name, -1);
//    svm.PrintLearningParameters();
//    svm.shuffle();
//    svm.Learn();
//    svm.Test();
//}

// T vs Accuracy
//int main(int argc, const char **argv) {
////    SgdSvm svm;
////    svm.LoadTrainingSampleFile("/home/harvey/factory/sgd/sgd-2.0/data/pascal/alpha03.txt", 100000);
////    svm.LoadTestSampleFile("/home/harvey/factory/sgd/sgd-2.0/data/pascal/alpha04.txt", 100000);
////    svm.set_lambda(0.0001);
////    svm.set_batch_size(1);
////    svm.set_num_iter_to_average(1);
////
////    svm.set_T(800000);
////    svm.Learn();
////    svm.Test();
//
//
////    for (int i = 1; i <= 30; i++) {
////    	svm.set_num_iter_to_average(i);
////      svm.reset();
////      cout << endl << "Experiment " << i << " ********************************* " << endl;
////      svm.Learn();
////      svm.Test();
////    }
//}

// Fixed T, lambda vs Accuracy
//int main(int argc, const char **argv) {
//	    SgdSvm svm;
//	    svm.LoadTrainingSampleFile("/home/harvey/factory/sgd/sgd-2.0/data/pascal/alpha03.txt", 100000);
//	    svm.LoadTestSampleFile("/home/harvey/factory/sgd/sgd-2.0/data/pascal/alpha04.txt", 100000);
//	    //svm.set_lambda(0.00001);
//	    svm.set_T(800000);
//	    svm.set_batch_size_ratio(0.00001);
//	    svm.set_averaging_start_ratio(1.0);
//
//	    for (int i = 3; i <= 7; i++) {
//	    	int base = 1;
//	    	for (int j = 0; j < i; j++) {
//	    		base *= 10;
//	    	}
//	    	double lambda = 1.0 / base;
//	    	svm.set_lambda(lambda);
//	        svm.reset();
//	        cout << endl << "Experiment " << i << " ********************************* " << endl;
//	        svm.PrintLearningParameters();
//	        svm.Learn();
//	        svm.Test();
//	    }
//}

// //Fixed T, Batch-size vs Accuracy
//int main(int argc, const char **argv) {
//	    SgdSvm svm;
//	    svm.LoadTrainingSampleFile("/home/harvey/factory/sgd/sgd-2.0/data/pascal/alpha03.txt", -1);
//	    svm.LoadTestSampleFile("/home/harvey/factory/sgd/sgd-2.0/data/pascal/alpha04.txt", -1);
//	    svm.set_lambda(0.0001);
//	    svm.set_T(300000);
//	    svm.set_num_iter_to_average(3);
//
//	    double batch_size[] = {1, 2, 4, 16, 32, 64, 128};
//	    for (int i = 0; i < 7; i++) {
//        svm.set_batch_size(batch_size[i]);
//        svm.reset();
//        cout << endl << "Experiment " << i << " ********************************* " << endl;
//        svm.Learn();
//        svm.Test();
//	    }
//}

// Fixed batch size, T vs Accuracy
//int main(int argc, const char **argv) {
//	    SgdSvm svm;
//	    svm.LoadTrainingSampleFile("/home/harvey/factory/sgd/sgd-2.0/data/pascal/alpha03.txt", 100000);
//	    svm.LoadTestSampleFile("/home/harvey/factory/sgd/sgd-2.0/data/pascal/alpha04.txt", 100000);
//	    svm.set_lambda(0.00001);
//	    //svm.set_T(100000);
//	    svm.set_batch_size_ratio(128.0 / 100000);
//	    svm.set_averaging_start_ratio(1.0);
//
//	    //double batch_size[] = {1.0 / 100000, 16.0 / 100000, 64.0 / 100000, 128.0 / 100000, 256.0 / 100000, 512.0 / 100000, 1024.0 / 100000};
//	    int iter[] = {25000, 50000, 100000, 200000, 400000};
//	    for (int i = 0; i < 5; i++) {
//            svm.set_T(iter[i]);
//	        svm.reset();
//	        cout << endl << "Experiment " << i << " ********************************* " << endl;
//	        svm.PrintLearningParameters();
//	        svm.Learn();
//	        svm.Test();
//	    }
//}


