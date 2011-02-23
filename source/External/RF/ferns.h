#ifndef FERNS_
#define FERNS_

#include "hyperparameters.h"
#include <string>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include "seminaivebayes.h"

using namespace boost::numeric::ublas;

class Ferns
{

public:
    Ferns(const HyperParameters &hp);
    ~Ferns();

    void train(const matrix<float>& data, const std::vector<int>& labels, bool use_gpu = false);
    void eval(const matrix<float>& data, const std::vector<int>& labels);

private:
    matrix<float> m_confidences;
    std::vector<int> m_predictions;
    HyperParameters m_hp;
    std::vector<SemiNaiveBayes::Ptr> m_naiveBayesClassifiers;


    void initialize(const int numSamples);

    void trainByCPU(const matrix<float>& data, const std::vector<int>& labels);

    double computeError(const std::vector<int>& labels);


};

#endif // FERNS_
