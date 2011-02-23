#ifndef DAS_RANDOM_NAIVE_BAYES_
#define DAS_RANDOM_NAIVE_BAYES_

#include "hyperparameters.h"
#include <string>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <libxml/tree.h>
#include <libxml/parser.h>
#include "naivebayes.h"
#include "randomnaivebayes.h"
#include "similarity.h"

using namespace boost::numeric::ublas;

class DASRandomNaiveBayes : public RandomNaiveBayes
{

public:
    DASRandomNaiveBayes(const HyperParameters &hp);
    ~DASRandomNaiveBayes();

    void train(const matrix<float>& dataTr, const std::vector<int>& labelsTr, const matrix<float>& dataTs,
               const std::vector<int>& labelsTs,bool use_gpu = false);




private:
    Similarity m_sim;
    matrix<float> m_simConf;

    void trainByCPU(const matrix<float>& dataTr, const std::vector<int>& labelsTr,
                    const matrix<float>& dataTs, const std::vector<int>& labelsTs);

    void calcLabelsAndWeights(const matrix<float>& RFConf, const std::vector<int>& RFPre,
                              std::vector<std::vector<int> >& tmpLabels, std::vector<std::vector<double> >& tmpWeights, const int nEpoch,
                              const std::vector<int>& labels, double& numberOfSwitchs,
                              std::vector<int>& classifiersToBeTrained);

    double temperature(const int nEpoch);
    bool shouldISwitch(const int nEpoch);
    bool shouldITrain(const int nEpoch);
    double computeOOBE(const std::vector<int>& labels);

};



#endif // RANDOM_NAIVE_BAYES_

