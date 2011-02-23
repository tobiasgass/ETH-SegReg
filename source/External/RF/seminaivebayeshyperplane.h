#ifndef SEMI_NAIVE_BAYES_HYPERPLANE_
#define SEMI_NAIVE_BAYES_HYPERPLANE_

#include "hyperparameters.h"
#include "naivebayes.h"
#include "seminaivebayes.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/shared_ptr.hpp>
#include "utilities.h"

using namespace boost::numeric::ublas;
using std::cout;
using std::endl;

struct SemiFeatureHyperplane
{
  void eval(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences);
  void evalGaussian(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences);
  void evalHistogram(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences);
  std::vector<int> indeces;
  std::vector<float> weights;
  std::vector<double> mean;
  std::vector<double> variance;

  // For Histogramming
  std::vector<std::vector<float> > histogram;
  float numBins;
  float w;
  float min;
  float max;
  float threshold; // for the binomial case
  FeatureType type;

};

class SemiNaiveBayesHyperplane : public SemiNaiveBayes
{
 public:
  typedef boost::shared_ptr<SemiNaiveBayesHyperplane> Ptr;
  SemiNaiveBayesHyperplane(const HyperParameters &hp);

  virtual void train(const matrix<float>& data, const std::vector<int>& labels,
                     matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                     std::vector<int>& forestOutOfBagVoteNum);
  virtual void train(const matrix<float>& data, const std::vector<int>& labels,
                     const std::vector<double>& weights, matrix<float>& forestConfidences,
                     matrix<float>& forestOutOfBagConfidences, std::vector<int>& forestOutOfBagVoteNum);
  virtual void eval(const matrix<float>& data, const std::vector<int> labels);
  virtual void eval(const matrix<float>& data, matrix<float>& confidences);

  //double computeError(const std::vector<int>& labels);
 private:
  std::vector<SemiFeatureHyperplane> m_features;

  void evalOutOfBagSamples(const matrix<float>& data);

  void calcMeanAndVariance(const matrix<float>& data, const std::vector<int>& labels, SemiFeatureHyperplane& f);
  void calcHistogram(const matrix<float>& data, const std::vector<int>& labels, SemiFeatureHyperplane& f);

  // Weighted Statistics
  void calcMeanAndVariance(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights, SemiFeatureHyperplane& f);
  void calcHistogram(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights, SemiFeatureHyperplane& f);
};

#endif // SEMI_NAIVE_BAYES_HYPERPLANE_


