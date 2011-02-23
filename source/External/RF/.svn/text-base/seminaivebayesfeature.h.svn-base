#ifndef SEMI_NAIVE_BAYES_GAUSSIAN_
#define SEMI_NAIVE_BAYES_GAUSSIAN_

#include "hyperparameters.h"
#include "naivebayes.h"
#include "seminaivebayes.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/shared_ptr.hpp>
#include "utilities.h"
#include "naivebayes.h"

using namespace boost::numeric::ublas;
using std::cout;
using std::endl;

struct SemiFeature
{
  void eval(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences);
  void evalGaussian(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences);
  void evalHistogram(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences);
  std::vector<int> indeces;

  std::vector<double> mean;
  std::vector<double> variance;
  
  std::vector<std::vector<float> > histogram;
  float numBins;
  std::vector<float> w;
  std::vector<float> min;
  std::vector<float> max;
  std::vector<float> threshold; // for the binomial case
  FeatureType type;
};

class SemiNaiveBayesFeature : public SemiNaiveBayes
{
 public:
  typedef boost::shared_ptr<SemiNaiveBayesFeature> Ptr;
  SemiNaiveBayesFeature(const HyperParameters &hp);

  virtual void train(const matrix<float>& data, const std::vector<int>& labels,
                     matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                     std::vector<int>& forestOutOfBagVoteNum);
  virtual void train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
                     matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                     std::vector<int>& forestOutOfBagVoteNum);
  virtual void eval(const matrix<float>& data, const std::vector<int> labels);
  virtual void eval(const matrix<float>& data, matrix<float>& confidences);

 private:
  std::vector<SemiFeature> m_features;

  void evalOutOfBagSamples(const matrix<float>& data);
  void calcMeanAndVariance(const matrix<float>& data, const std::vector<int>& labels, SemiFeature& f);
  void calcHistogram(const matrix<float>& data, const std::vector<int>& labels, SemiFeature& f);
  // Weighted Statistics
  void calcMeanAndVariance(const matrix<float>& data, const std::vector<int>& labels, SemiFeature& f,const std::vector<double>& weights);
  void calcHistogram(const matrix<float>& data, const std::vector<int>& labels, SemiFeature& f,const std::vector<double>& weights);
};

#endif // SEMI_NAIVE_BAYES_GAUSSIAN_


