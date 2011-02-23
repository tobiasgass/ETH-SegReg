#include "seminaivebayeshyperplane.h"
#include "utilities.h"
#include <boost/foreach.hpp>

using namespace std;

SemiNaiveBayesHyperplane::SemiNaiveBayesHyperplane(const HyperParameters &hp) : SemiNaiveBayes( hp )
{
  m_featureType = (hp.naiveBayesFeatureType == "Gaussian") ? FEATURE_GAUSSIAN : FEATURE_HISTOGRAM;
}

void SemiNaiveBayesHyperplane::calcMeanAndVariance(const matrix<float>& data, const std::vector<int>& labels, SemiFeatureHyperplane& f)
{
  std::vector<double> mean(m_hp.numClasses,0.0);
  std::vector<double> variance(m_hp.numClasses,0.0);
  std::vector<int> numCounts(m_hp.numClasses,0);

  BOOST_FOREACH(int sample, m_inBagSamples)
    {
      double projectionVals = 0.0;
      int featureCounter = 0;
      BOOST_FOREACH(int feat, f.indeces)
        {
          projectionVals += data(sample,feat)*f.weights[featureCounter];
          featureCounter++;
        }

      mean[labels[sample]] += projectionVals;
      numCounts[labels[sample]]++;
      variance[labels[sample]] += pow(projectionVals,2.0);
    }

  for (int c = 0; c < m_hp.numClasses; c++)
    {
      mean[c] /= numCounts[c];  // E(X)
      variance[c] /= numCounts[c]; // E(X^2)
      variance[c] -= pow(mean[c], 2.0);
    }

  f.mean = mean;
  f.variance = variance;
}

// Weighted continuous Statistics
void SemiNaiveBayesHyperplane::calcMeanAndVariance(const matrix<float>& data, const std::vector<int>& labels,
                                                   const std::vector<double>& weights, SemiFeatureHyperplane& f)
{
  std::vector<double> mean(m_hp.numClasses,0.0);
  std::vector<double> variance(m_hp.numClasses,0.0);
  std::vector<double> numCounts(m_hp.numClasses,0.0);

  BOOST_FOREACH(int sample, m_inBagSamples)
    {
      double projectionVals = 0.0;
      int featureCounter = 0;
      BOOST_FOREACH(int feat, f.indeces)
        {
          projectionVals += data(sample,feat)*f.weights[featureCounter];
          featureCounter++;
        }

      mean[labels[sample]] += projectionVals*weights[sample];
      numCounts[labels[sample]] += weights[sample];
      variance[labels[sample]] += pow(projectionVals*weights[sample],2.0);
    }

  for (int c = 0; c < m_hp.numClasses; c++)
    {
      mean[c] /= numCounts[c];  // E(X)
      variance[c] /= numCounts[c]; // E(X^2)
      variance[c] -= pow(mean[c], 2.0);
    }

  f.mean = mean;
  f.variance = variance;
}


void SemiNaiveBayesHyperplane::calcHistogram(const matrix<float>& data, const std::vector<int>& labels, SemiFeatureHyperplane& f)
{
  std::vector<float> projectionVals(data.size1(),0.0);
  int featureCounter = 0;
  BOOST_FOREACH(int feat, f.indeces)
    {
      projectionVals[m_inBagSamples[0]] += data(m_inBagSamples[0],feat)*f.weights[featureCounter];
      featureCounter++;
    }

  // find min and max values
  f.min = projectionVals[m_inBagSamples[0]];
  f.max = projectionVals[m_inBagSamples[0]];

  BOOST_FOREACH(int sample, m_inBagSamples)
    {
      featureCounter = 0;
      BOOST_FOREACH(int feat, f.indeces)
        {
          projectionVals[sample] += data(sample,feat)*f.weights[featureCounter];
          featureCounter++;
        }

      f.min = (projectionVals[sample] < f.min) ? projectionVals[sample] : f.min;
      f.max = (projectionVals[sample] > f.max) ? projectionVals[sample] : f.max;
    }
  f.numBins = (float)m_hp.numHistogramBins;

  std::vector<float> classHistogram((int) f.numBins, 0.0);
  std::vector<std::vector<float> > histogram(m_hp.numClasses,classHistogram);
  std::vector<int> classCounter(m_hp.numClasses,0);

  if ( f.numBins > 2.0)
    {
      f.w = abs(f.min - f.max)/f.numBins;

      BOOST_FOREACH(int sample, m_inBagSamples)
        {
          int binIndex = (int) ((projectionVals[sample] - f.min)/f.w);
          binIndex -= (binIndex == f.numBins) ? 1 : 0;
          histogram[labels[sample]][binIndex]++;
          classCounter[labels[sample]]++;
        }
    }
  else // binomial
    {
      const float numSteps = 20;
      const float stepSize = abs(f.min - f.max)/numSteps;
      float bestEntropy = 1e10;
      float bestThreshold = 0.0;
      std::vector<int> decision(data.size1());
      // Find the best threshold
      for (double threshold = f.min ; threshold < f.max ; threshold += stepSize) {
        int oneCount = 0, zeroCount = 0;
        std::vector<double> zeroClassCount(m_hp.numClasses, 0.0), oneClassCount(m_hp.numClasses, 0.0);

        BOOST_FOREACH(int sample, m_inBagSamples)
          {
            if (projectionVals[sample] > threshold) {
              decision[sample] = 1;
              oneClassCount[labels[sample]]++;
              oneCount++;
            }
            else {
              decision[sample] = 0;
              zeroClassCount[labels[sample]]++;
              zeroCount++;
            }
          }

        // Calc entropy
        double oneEntropy = 0, zeroEntropy = 0;
        for (int nClass = 0; nClass < m_hp.numClasses; nClass++) {
          oneClassCount[nClass] /= oneCount;
          zeroClassCount[nClass] /= zeroCount;

          if (oneClassCount[nClass]) {
            oneEntropy -= oneClassCount[nClass]*log(oneClassCount[nClass]);
          }
          if (zeroClassCount[nClass]) {
            zeroEntropy -= zeroClassCount[nClass]*log(zeroClassCount[nClass]);
          }
        }

        // Total Entropy
        double entropy = (zeroEntropy*zeroCount + oneEntropy*oneCount)/(zeroCount + oneCount);
        if (entropy < bestEntropy) {
          bestEntropy = entropy;
          bestThreshold = threshold;
        }
      }
      f.threshold = bestThreshold;

      BOOST_FOREACH(int sample, m_inBagSamples)
        {
          int binIndex = (projectionVals[sample] > f.threshold) ? 1 : 0;
          histogram[labels[sample]][binIndex]++;
          classCounter[labels[sample]]++;
        }
    }

  std::vector<std::vector<float> >::iterator it(histogram.begin());
  std::vector<std::vector<float> >::iterator end(histogram.end());
  for (int c = 0;it != end; it++, c++)
    {
      for (int bin = 0; bin < f.numBins; bin++)
        {
          (*it)[bin] /= (classCounter[c] + 1e-10);
        }
    }
  f.histogram = histogram;
}

// Weighted Discrete Statistics
void SemiNaiveBayesHyperplane::calcHistogram(const matrix<float>& data, const std::vector<int>& labels,
                                             const std::vector<double>& weights, SemiFeatureHyperplane& f)
{
  std::vector<float> projectionVals(data.size1(),0.0);
  int featureCounter = 0;
  BOOST_FOREACH(int feat, f.indeces)
    {
      projectionVals[m_inBagSamples[0]] += data(m_inBagSamples[0],feat)*f.weights[featureCounter];
      featureCounter++;
    }

  // find min and max values
  f.min = projectionVals[m_inBagSamples[0]];
  f.max = projectionVals[m_inBagSamples[0]];

  BOOST_FOREACH(int sample, m_inBagSamples)
    {
      featureCounter = 0;
      BOOST_FOREACH(int feat, f.indeces)
        {
          projectionVals[sample] += data(sample,feat)*f.weights[featureCounter];
          featureCounter++;
        }

      f.min = (projectionVals[sample] < f.min) ? projectionVals[sample] : f.min;
      f.max = (projectionVals[sample] > f.max) ? projectionVals[sample] : f.max;
    }
  f.numBins = (float)m_hp.numHistogramBins;

  std::vector<float> classHistogram((int) f.numBins, 0.0);
  std::vector<std::vector<float> > histogram(m_hp.numClasses,classHistogram);
  std::vector<double> classCounter(m_hp.numClasses, 0.0);

  if ( f.numBins > 2.0)
    {
      f.w = abs(f.min - f.max)/f.numBins;

      BOOST_FOREACH(int sample, m_inBagSamples)
        {
          int binIndex = (int) ((projectionVals[sample] - f.min)/f.w);
          binIndex -= (binIndex == f.numBins) ? 1 : 0;
          histogram[labels[sample]][binIndex] += weights[sample];
          classCounter[labels[sample]] += weights[sample];
        }
    }
  else // binomial
    {
      const float numSteps = 20;
      const float stepSize = abs(f.min - f.max)/numSteps;
      float bestEntropy = 1e10;
      float bestThreshold = 0.0;
      std::vector<int> decision(data.size1());
      // Find the best threshold
      for (double threshold = f.min ; threshold < f.max ; threshold += stepSize) {
        double oneCount = 0, zeroCount = 0;
        std::vector<double> zeroClassCount(m_hp.numClasses, 0.0), oneClassCount(m_hp.numClasses, 0.0);

        BOOST_FOREACH(int sample, m_inBagSamples)
          {
            if (projectionVals[sample] > threshold) {
              decision[sample] = 1;
              oneClassCount[labels[sample]] += weights[sample];
              oneCount += weights[sample];
            }
            else {
              decision[sample] = 0;
              zeroClassCount[labels[sample]] += weights[sample];
              zeroCount += weights[sample];
            }
          }

        // Calc entropy
        double oneEntropy = 0, zeroEntropy = 0;
        for (int nClass = 0; nClass < m_hp.numClasses; nClass++) {
          oneClassCount[nClass] /= oneCount;
          zeroClassCount[nClass] /= zeroCount;

          if (oneClassCount[nClass]) {
            oneEntropy -= oneClassCount[nClass]*log(oneClassCount[nClass]);
          }
          if (zeroClassCount[nClass]) {
            zeroEntropy -= zeroClassCount[nClass]*log(zeroClassCount[nClass]);
          }
        }

        // Total Entropy
        double entropy = (zeroEntropy*zeroCount + oneEntropy*oneCount)/(zeroCount + oneCount);
        if (entropy < bestEntropy) {
          bestEntropy = entropy;
          bestThreshold = threshold;
        }
      }
      f.threshold = bestThreshold;

      BOOST_FOREACH(int sample, m_inBagSamples)
        {
          int binIndex = (projectionVals[sample] > f.threshold) ? 1 : 0;
          histogram[labels[sample]][binIndex] += weights[sample];
          classCounter[labels[sample]] += weights[sample];
        }
    }

  std::vector<std::vector<float> >::iterator it(histogram.begin());
  std::vector<std::vector<float> >::iterator end(histogram.end());
  for (int c = 0;it != end; it++, c++)
    {
      for (int bin = 0; bin < f.numBins; bin++)
        {
          (*it)[bin] /= (classCounter[c] + 1e-10);
        }
    }
  f.histogram = histogram;
}


void SemiNaiveBayesHyperplane::train(const matrix<float>& data, const std::vector<int>& labels,
                                     matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                                     std::vector<int>& forestOutOfBagVoteNum)
{
  // Initialize
  initialize(m_hp.numLabeled);

  // Random Subsamples data according to bagratio
  subSample(m_hp.numLabeled);

  // Train the Naive Bayes Classifiers

  for (int n = 0; n < m_hp.numRandomFeatures;n++)
    {
      SemiFeatureHyperplane f;
      f.type = m_featureType;
      f.indeces = randPerm(data.size2(),m_hp.numProjFeatures);
      matrix<float> tmpConfidences(data.size1(),m_hp.numClasses);
      double bestError = 1.0;

      std::vector<float> bestWeights(m_hp.numProjFeatures);
      f.weights.resize(m_hp.numProjFeatures);

      if (m_featureType == FEATURE_GAUSSIAN)
        {
          std::vector<double> bestMean;
          std::vector<double> bestVariance;
          for (int t = 0; t < m_hp.numTries;t++)
            {
              fillWithRandomNumbers(f.weights);
              calcMeanAndVariance(data,labels,f);

              for (int sample = 0; sample < (int)data.size1();sample++)
                {
                  for (int c = 0; c < m_hp.numClasses; c++)
                    {
                      tmpConfidences(sample,c) = 1.0;
                    }
                }
              // calc error
              double error = 0;
              for (int sample = 0; sample < (int)data.size1();sample++)
                {
                  //evaluate
                  f.eval(data,sample,tmpConfidences);
                  double bestConf = 0.0;
                  int bestClass = -1;
                  for (int c = 0; c < m_hp.numClasses; c++)
                    {
                      if (tmpConfidences(sample,c) > bestConf)
                        {
                          bestConf = tmpConfidences(sample,c);
                          bestClass = c;
                        }
                    }
                  error += (bestClass != labels[sample]) ? 1.0 : 0.0;
                }

              error /= data.size1();

              if (error < bestError)
                {
                  bestWeights = f.weights;
                  bestError = error;
                  bestMean = f.mean;
                  bestVariance = f.variance;
                }

            }
          f.weights = bestWeights;
          f.mean = bestMean;
          f.variance = bestVariance;
        }
      else   // Use Histogramming
        {
          std::vector<std::vector<float> > bestHistogram;
          float bestW = 0.0;
          float bestMin = 0.0;
          float bestMax = 0.0;
          for (int t = 0; t < m_hp.numRandomFeatures;t++)
            {
              fillWithRandomNumbers(f.weights);
              calcHistogram(data,labels,f);

              for (int sample = 0; sample < (int)data.size1();sample++)
                {
                  for (int c = 0; c < m_hp.numClasses; c++)
                    {
                      tmpConfidences(sample,c) = 1.0;
                    }
                }
              // calc error
              double error = 0;
              for (int sample = 0; sample < (int)data.size1();sample++)
                {
                  //evaluate
                  f.eval(data,sample,tmpConfidences);
                  double bestConf = 0.0;
                  int bestClass = -1;
                  for (int c = 0; c < m_hp.numClasses; c++)
                    {
                      if (tmpConfidences(sample,c) > bestConf)
                        {
                          bestConf = tmpConfidences(sample,c);
                          bestClass = c;
                        }
                    }
                  error += (bestClass != labels[sample]) ? 1.0 : 0.0;
                }

              error /= data.size1();

              if (error < bestError)
                {
                  bestWeights = f.weights;
                  bestError = error;
                  bestW = f.w;
                  bestHistogram = f.histogram;
                  bestMin = f.min;
                  bestMax = f.max;
                }

            }
          f.weights = bestWeights;
          bestW = f.w;
          f.histogram = bestHistogram;
          f.min = bestMin;
          f.max = bestMax;
        }
      m_features.push_back(f);

    }


  bool verbose = m_hp.verbose;
  m_hp.verbose = false;
  eval(data,labels);
  m_hp.verbose = verbose;

  finalize(data, forestConfidences, forestOutOfBagConfidences, forestOutOfBagVoteNum);
}

// Weighted Training
void SemiNaiveBayesHyperplane::train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
                                     matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                                     std::vector<int>& forestOutOfBagVoteNum)
{
  // Initialize
  initialize(m_hp.numLabeled);

  // Random Subsamples data according to bagratio
  subSample(m_hp.numLabeled);

  // Train the Naive Bayes Classifiers

  for (int n = 0; n < m_hp.numRandomFeatures;n++)
    {
      SemiFeatureHyperplane f;
      f.type = m_featureType;
      f.indeces = randPerm(data.size2(),m_hp.numProjFeatures);
      matrix<float> tmpConfidences(data.size1(),m_hp.numClasses);
      double bestError = 1.0;

      std::vector<float> bestWeights(m_hp.numProjFeatures);
      f.weights.resize(m_hp.numProjFeatures);

      if (m_featureType == FEATURE_GAUSSIAN)
        {
          std::vector<double> bestMean;
          std::vector<double> bestVariance;
          for (int t = 0; t < m_hp.numTries;t++)
            {
              fillWithRandomNumbers(f.weights);
              calcMeanAndVariance(data,labels,weights, f);

              for (int sample = 0; sample < (int)data.size1();sample++)
                {
                  for (int c = 0; c < m_hp.numClasses; c++)
                    {
                      tmpConfidences(sample,c) = 1.0;
                    }
                }
              // calc error
              double error = 0;
              for (int sample = 0; sample < (int)data.size1();sample++)
                {
                  //evaluate
                  f.eval(data,sample,tmpConfidences);
                  double bestConf = 0.0;
                  int bestClass = -1;
                  for (int c = 0; c < m_hp.numClasses; c++)
                    {
                      if (tmpConfidences(sample,c) > bestConf)
                        {
                          bestConf = tmpConfidences(sample,c);
                          bestClass = c;
                        }
                    }
                  error += (bestClass != labels[sample]) ? 1.0 : 0.0;
                }

              error /= data.size1();

              if (error < bestError)
                {
                  bestWeights = f.weights;
                  bestError = error;
                  bestMean = f.mean;
                  bestVariance = f.variance;
                }

            }
          f.weights = bestWeights;
          f.mean = bestMean;
          f.variance = bestVariance;
        }
      else   // Use Histogramming
        {
          std::vector<std::vector<float> > bestHistogram;
          float bestW = 0.0;
          float bestMin = 0.0;
          float bestMax = 0.0;
          for (int t = 0; t < m_hp.numRandomFeatures;t++)
            {
              fillWithRandomNumbers(f.weights);
              calcHistogram(data,labels,weights, f);

              for (int sample = 0; sample < (int)data.size1();sample++)
                {
                  for (int c = 0; c < m_hp.numClasses; c++)
                    {
                      tmpConfidences(sample,c) = 1.0;
                    }
                }
              // calc error
              double error = 0;
              for (int sample = 0; sample < (int)data.size1();sample++)
                {
                  //evaluate
                  f.eval(data,sample,tmpConfidences);
                  double bestConf = 0.0;
                  int bestClass = -1;
                  for (int c = 0; c < m_hp.numClasses; c++)
                    {
                      if (tmpConfidences(sample,c) > bestConf)
                        {
                          bestConf = tmpConfidences(sample,c);
                          bestClass = c;
                        }
                    }
                  error += (bestClass != labels[sample]) ? 1.0 : 0.0;
                }

              error /= data.size1();

              if (error < bestError)
                {
                  bestWeights = f.weights;
                  bestError = error;
                  bestW = f.w;
                  bestHistogram = f.histogram;
                  bestMin = f.min;
                  bestMax = f.max;
                }

            }
          f.weights = bestWeights;
          bestW = f.w;
          f.histogram = bestHistogram;
          f.min = bestMin;
          f.max = bestMax;
        }
      m_features.push_back(f);

    }


  bool verbose = m_hp.verbose;
  m_hp.verbose = false;
  eval(data,labels);
  m_hp.verbose = verbose;

  finalize(data, forestConfidences, forestOutOfBagConfidences, forestOutOfBagVoteNum);
}

void SemiNaiveBayesHyperplane::evalOutOfBagSamples(const matrix<float>& data)
{
}

void SemiNaiveBayesHyperplane::eval(const matrix<float>& data, const std::vector<int> labels)
{
  m_confidences.resize(data.size1(), m_hp.numClasses);
  // init to one (due to multiplication)
  for (int sample = 0; sample < (int)data.size1(); sample++)
    {
      for ( int c = 0; c < m_hp.numClasses; c++)
        {
          m_confidences(sample,c) = 1.0;
        }
    }

  // Fill confidence matrix
  std::vector<SemiFeatureHyperplane>::iterator it(m_features.begin());
  std::vector<SemiFeatureHyperplane>::iterator end(m_features.end());
  for (int sample = 0; sample < (int)data.size1(); sample++)
    {
      it = m_features.begin();
      while (it != end)
        {
          it->eval(data,sample,m_confidences);
          ++it;
        }
      double max = 0.0;
      for ( int c = 0; c < m_hp.numClasses; c++)
        {
          if (m_confidences(sample,c) > max)
            {
              m_predictions[sample] = c;
              max = m_confidences(sample,c);
            }
        }
    }

  double error = computeError(labels);
  if (m_hp.verbose)
    {
      cout << "Test Error: " << error << endl;
    }

}

void SemiNaiveBayesHyperplane::eval(const matrix<float>& data, matrix<float>& confidences)
{
  m_confidences.resize(data.size1(), m_hp.numClasses);
  // init to one (due to multiplication)
  for (int sample = 0; sample < (int)data.size1(); sample++)
    {
      for ( int c = 0; c < m_hp.numClasses; c++)
        {
          m_confidences(sample,c) = 1.0;
        }
    }

  // Fill confidence matrix
  std::vector<SemiFeatureHyperplane>::iterator it(m_features.begin());
  std::vector<SemiFeatureHyperplane>::iterator end(m_features.end());
  for (int sample = 0; sample < (int)data.size1(); sample++)
    {
      it = m_features.begin();
      while (it != end)
        {
          it->eval(data,sample,m_confidences);
          ++it;
        }
      for ( int c = 0; c < m_hp.numClasses; c++)
        {
          confidences(sample,c) += m_confidences(sample,c);
        }
    }

}

void SemiFeatureHyperplane::eval(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences)
{
  if ( type == FEATURE_GAUSSIAN )
    {
      evalGaussian(data,sampleIndex,confidences);
    }
  else
    {
      evalHistogram(data,sampleIndex,confidences);
    }
}

void SemiFeatureHyperplane::evalGaussian(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences)
{
  double projectionVals = 0.0;
  int featureCounter = 0;
  BOOST_FOREACH(int feat, indeces)
    {
      projectionVals += data(sampleIndex,feat)*weights[featureCounter];
      featureCounter++;
    }
  for (int c = 0; c < (int)mean.size(); c++)
    {
      confidences(sampleIndex,c) *= 1.0/variance[c] * exp(-pow(projectionVals - mean[c],2.0)/(2.0*pow(variance[c],2.0)));
    }
}

void SemiFeatureHyperplane::evalHistogram(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences)
{
  double projectionVals = 0.0;
  int featureCounter = 0;
  BOOST_FOREACH(int feat, indeces)
    {
      projectionVals += data(sampleIndex,feat)*weights[featureCounter];
      featureCounter++;
    }

  int binIndex;
  if (numBins > 2) {
    binIndex = (int) ((projectionVals - min)/w);
    binIndex -= (binIndex == numBins) ? 1 : 0;
    binIndex = (binIndex < 0.0) ? 0 : binIndex;
  }
  else {
    binIndex = (projectionVals > threshold) ? 1 : 0;
  }

  for (int c = 0; c < (int)histogram.size(); c++)
    {
      confidences(sampleIndex,c) *= histogram[c][binIndex];
    }
}


