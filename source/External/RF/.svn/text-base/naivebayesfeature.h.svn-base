#ifndef NAIVE_BAYES_GAUSSIAN_
#define NAIVE_BAYES_GAUSSIAN_

#include "hyperparameters.h"
#include "naivebayes.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/shared_ptr.hpp>
#include "utilities.h"

using namespace boost::numeric::ublas;
using std::cout;
using std::endl;

struct Feature
{
    void eval(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences);
    void evalGaussian(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences);
    void evalHistogram(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences);
    int index;
    std::vector<double> mean;
    std::vector<double> variance;
    std::vector<std::vector<float> > histogram;
    float numBins;
    float w;
    float min;
    float max;
    float threshold; // for the binomial case
    FeatureType type;
 };

class NaiveBayesFeature : public NaiveBayes
{
public:
    typedef boost::shared_ptr<NaiveBayesFeature> Ptr;
    NaiveBayesFeature(const HyperParameters &hp);

    virtual void train(const matrix<float>& data, const std::vector<int>& labels,
               matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
               std::vector<int>& forestOutOfBagVoteNum);
    virtual void train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
               matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
               std::vector<int>& forestOutOfBagVoteNum, bool init = true);
    virtual void retrain(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
               matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
               std::vector<int>& forestOutOfBagVoteNum, bool init = true);
    virtual void eval(const matrix<float>& data, const std::vector<int> labels);
    virtual void eval(const matrix<float>& data, matrix<float>& confidences);

private:
    std::vector<Feature> m_features;

    void evalOutOfBagSamples(const matrix<float>& data);
    void calcMeanAndVariance(const matrix<float>& data, const std::vector<int>& labels, Feature& f);
    void calcHistogram(const matrix<float>& data, const std::vector<int>& labels, Feature& f);
    // Weighted Statistics
    void calcMeanAndVariance(const matrix<float>& data, const std::vector<int>& labels, Feature& f,const std::vector<double>& weights);
    void calcHistogram(const matrix<float>& data, const std::vector<int>& labels, Feature& f,const std::vector<double>& weights);
};

#endif //NAIVE_BAYES_GAUSSIAN_


