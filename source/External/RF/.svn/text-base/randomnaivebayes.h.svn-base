#ifndef RANDOM_NAIVE_BAYES_
#define RANDOM_NAIVE_BAYES_

#include "hyperparameters.h"
#include <string>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include "naivebayes.h"

using namespace boost::numeric::ublas;

class RandomNaiveBayes
{

public:
    RandomNaiveBayes(const HyperParameters &hp);
    ~RandomNaiveBayes();

    void train(const matrix<float>& data, const std::vector<int>& labels, bool use_gpu = false);
    void eval(const matrix<float>& data, const std::vector<int>& labels);

protected:
    matrix<float> m_confidences;
    std::vector<int> m_predictions;
    HyperParameters m_hp;
    std::vector<NaiveBayes::Ptr> m_naiveBayesClassifiers;


    void initialize(const int numSamples);

    void trainByCPU(const matrix<float>& data, const std::vector<int>& labels);

    double computeError(const std::vector<int>& labels);


};



#endif // RANDOM_NAIVE_BAYES_
