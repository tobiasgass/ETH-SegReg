#ifndef NAIVE_BAYES_
#define NAIVE_BAYES_

#include "hyperparameters.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/shared_ptr.hpp>
#include "utilities.h"

using namespace boost::numeric::ublas;
using std::cout;
using std::endl;

enum FeatureType{FEATURE_GAUSSIAN, FEATURE_HISTOGRAM};

class NaiveBayes
{
public:
    typedef boost::shared_ptr<NaiveBayes> Ptr;
    NaiveBayes(const HyperParameters &hp);
    ~NaiveBayes();

    virtual void train(const matrix<float>& data, const std::vector<int>& labels,
               matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
               std::vector<int>& forestOutOfBagVoteNum) = 0;

    virtual void train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
               matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
               std::vector<int>& forestOutOfBagVoteNum, bool init = true) = 0;


    virtual void retrain(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
               matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
               std::vector<int>& forestOutOfBagVoteNum, bool init = true) = 0;

    virtual void eval(const matrix<float>& data, const std::vector<int> labels) = 0;
    virtual void eval(const matrix<float>& data, matrix<float>& confidences) = 0;

    double computeError(const std::vector<int>& labels);
    inline std::vector<int> getPredictions() const { return m_predictions; };
    inline std::vector<int> getInBagSamples() const { return m_inBagSamples; };
    inline std::vector<int> getOutOfBagSamples() const { return m_outOfBagSamples; };
    inline matrix<float> getConfidences() const { return m_confidences; };
protected:
    FeatureType m_featureType;
    HyperParameters m_hp;
    matrix<float> m_confidences;
    std::vector<int> m_predictions;
    std::vector<int> m_inBagSamples;
    std::vector<int> m_outOfBagSamples;

    void initialize(const int numSamples); // Create the confidence matrices and prediction vectors
    void subSample(const int numSamples);    // Create bags
    void finalize(const matrix<float>& data,
                  matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                  std::vector<int>& forestOutOfBagVoteNum);

    void clean();

};

#endif //NAIVE_BAYES_

