#ifndef MY_SVM_H
#define MY_SVM_H
#include <opencv2/ml/ml.hpp>
using namespace cv;

class MySVM : public CvSVM
{
public:
    double * get_alpha_vector()
    {
        return this->decision_func->alpha;
    }

    float get_rho()
    {
        return this->decision_func->rho;
    }
};

#endif
