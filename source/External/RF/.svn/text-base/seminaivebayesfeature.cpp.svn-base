#include "seminaivebayesfeature.h"
#include "utilities.h"
#include <boost/foreach.hpp>

using namespace std;

SemiNaiveBayesFeature::SemiNaiveBayesFeature(const HyperParameters &hp) : SemiNaiveBayes( hp ) {
  m_featureType = (hp.naiveBayesFeatureType == "Gaussian") ? FEATURE_GAUSSIAN : FEATURE_HISTOGRAM;
}

void SemiNaiveBayesFeature::calcMeanAndVariance(const matrix<float>& data, const std::vector<int>& labels, SemiFeature& f) {
  cout << "SemiNaiveBayesFeature for Gaussians is not implemented yet!!!" << endl;
  exit(1);
}

// Weighted Statistics
void SemiNaiveBayesFeature::calcMeanAndVariance(const matrix<float>& data, const std::vector<int>& labels, SemiFeature& f,
                                                const std::vector<double>& weights) {
  cout << "SemiNaiveBayesFeature for Gaussians is not implemented yet!!!" << endl;
  exit(1);
}

void SemiNaiveBayesFeature::calcHistogram(const matrix<float>& data, const std::vector<int>& labels, SemiFeature& f) {
  // find min and max values
  int numF = (int) f.indeces.size();
  for (int nF = 0; nF < numF; nF++) {
    f.min.push_back(data(m_inBagSamples[0],f.indeces[nF]));
    f.max.push_back(data(m_inBagSamples[0],f.indeces[nF]));

    BOOST_FOREACH(int sample, m_inBagSamples) {
      f.min[nF] = (data(sample,f.indeces[nF]) < f.min[nF]) ? data(sample,f.indeces[nF]) : f.min[nF];
      f.max[nF] = (data(sample,f.indeces[nF]) > f.max[nF]) ? data(sample,f.indeces[nF]) : f.max[nF];
    }
  }
  f.numBins = (float) m_hp.numHistogramBins;

  std::vector<float> classHistogram((int) pow(f.numBins, numF), 0.0);
  std::vector<std::vector<float> > histogram(m_hp.numClasses, classHistogram);
  std::vector<int> classCounter(m_hp.numClasses,0);
  if (f.numBins > 2.0) {
    for (int nF = 0; nF < numF; nF++) {
      f.w.push_back(abs(f.min[nF] - f.max[nF])/f.numBins);
    }

    BOOST_FOREACH(int sample, m_inBagSamples) {
      int binIndex = 0;
      for (int nF = 0; nF < numF; nF++) {
        binIndex += (int) ((data(sample, f.indeces[nF]) - f.min[nF])/f.w[nF]) + nF*numF;
      }
      binIndex -= (binIndex == pow(f.numBins, numF)) ? 1 : 0;
      histogram[labels[sample]][binIndex]++;
      classCounter[labels[sample]]++;
    }
  } else {
    if (numF > 2) {
      cout << "Exhaustive search for the bins is not possible for more than 2 features!!!" << endl;
      exit(1);
    }
    const float numSteps = 20;
    std::vector<float> stepSize(numF);
    for (int nF = 0; nF < numF; nF++) {
      stepSize[nF] = abs(f.min[nF] - f.max[nF])/numSteps;
    }
    float bestEntropy = 1e10;
    std::vector<float> bestThreshold(numF, 0.0);
    int decision = 0;
    // Find the best threshold
    for (double threshold0 = f.min[0] ; threshold0 < f.max[0] ; threshold0 += stepSize[0]) {
      for (double threshold1 = f.min[1] ; threshold1 < f.max[1] ; threshold1 += stepSize[1]) {
        std::vector<int> count(4, 0);
        std::vector<double> tmpCount(m_hp.numClasses, 0.0);
        std::vector<std::vector<double> > classCount(4, tmpCount);

        BOOST_FOREACH(int sample, m_inBagSamples) {
          if (data(sample, f.indeces[0]) < threshold0 && data(sample, f.indeces[1]) < threshold1) { //00
            decision = 0;
          } else if (data(sample, f.indeces[0]) < threshold0 && data(sample, f.indeces[1]) > threshold1) { //01
            decision = 1;
          } else if (data(sample, f.indeces[0]) > threshold0 && data(sample, f.indeces[1]) < threshold1) { //10
            decision = 2;
          } else if (data(sample, f.indeces[0]) > threshold0 && data(sample, f.indeces[1]) > threshold1) { //11
            decision = 3;
          }

          classCount[decision][labels[sample]]++;
          count[decision]++;
        }

        // Calc entropy
        int totalSampleNum = 0;
        double totalEntropy = 0;
        std::vector<double> entropy(4, 0.0);
        for (int nB = 0; nB < 4; nB++) {
          for (int nClass = 0; nClass < m_hp.numClasses; nClass++) {
            classCount[nB][nClass] /= count[nB];
            classCount[nB][nClass] /= count[nB];
            
            if (classCount[nB][nClass]) {
              entropy[nB] -= classCount[nB][nClass]*log(classCount[nB][nClass]);
            }
          }

          totalEntropy += count[nB]*entropy[nB];
          totalSampleNum += count[nB];
        }

        // Total Entropy
        totalEntropy /= totalSampleNum;
        if (totalEntropy < bestEntropy) {
          bestEntropy = totalEntropy;
          bestThreshold[0] = threshold0;
          bestThreshold[1] = threshold1;
        }
      }
    }
    f.threshold = bestThreshold;

    BOOST_FOREACH(int sample, m_inBagSamples) {
      int tmpIndex = 0;
      if (data(sample, f.indeces[0]) < f.threshold[0] && data(sample, f.indeces[1]) < f.threshold[1]) { //00
        tmpIndex = 0;
      } else if (data(sample, f.indeces[0]) < f.threshold[0] && data(sample, f.indeces[1]) > f.threshold[1]) { //01
        tmpIndex = 1;
      } else if (data(sample, f.indeces[0]) > f.threshold[0] && data(sample, f.indeces[1]) < f.threshold[1]) { //10
        tmpIndex = 2;
      } else if (data(sample, f.indeces[0]) > f.threshold[0] && data(sample, f.indeces[1]) > f.threshold[1]) { //11
        tmpIndex = 3;
      }

      histogram[labels[sample]][tmpIndex]++;
      classCounter[labels[sample]]++;
    }
  }

  std::vector<std::vector<float> >::iterator it(histogram.begin());
  std::vector<std::vector<float> >::iterator end(histogram.end());
  for (int c = 0;it != end; it++, c++) {
    for (int bin = 0; bin < (int) pow(f.numBins, numF); bin++) {
      (*it)[bin] /= (classCounter[c] + 1e-10);
    }
  }
  f.histogram = histogram;
}

// Weighted Discrete Statistics
void SemiNaiveBayesFeature::calcHistogram(const matrix<float>& data, const std::vector<int>& labels, SemiFeature& f,
                                          const std::vector<double>& weights) {

}


void SemiNaiveBayesFeature::train(const matrix<float>& data, const std::vector<int>& labels,
                                  matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                                  std::vector<int>& forestOutOfBagVoteNum) {
  // Initialize
  initialize(m_hp.numLabeled);

  // Random Subsamples data according to bagratio
  subSample(m_hp.numLabeled);

  // Train the Naive Bayes Classifiers
  SemiFeature f;
  f.type = m_featureType;
  if (m_featureType == FEATURE_GAUSSIAN) {
    cout << "SemiNaiveBayesFeature for Gaussians is not implemented yet!!!" << endl;
    exit(1);
  } else { // FEATURE_HISTOGRAM
    f.indeces = randPerm(data.size2(), m_hp.numRandomFeatures);
    calcHistogram(data,labels,f);
    m_features.push_back(f);
  }

  eval(data,labels);

  finalize(data, forestConfidences, forestOutOfBagConfidences, forestOutOfBagVoteNum);
}

// Weighted Training
void SemiNaiveBayesFeature::train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
                                  matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                                  std::vector<int>& forestOutOfBagVoteNum) {
  // Initialize
  initialize(m_hp.numLabeled);

  // Random Subsamples data according to bagratio
  subSample(m_hp.numLabeled);

  // Train the Naive Bayes Classifiers
  SemiFeature f;
  f.type = m_featureType;
  if (m_featureType == FEATURE_GAUSSIAN) {
    cout << "SemiNaiveBayesFeature for Gaussians is not implemented yet!!!" << endl;
    exit(1);
  } else { // FEATURE_HISTOGRAM
    f.indeces = randPerm(data.size2(),m_hp.numRandomFeatures);
    calcHistogram(data,labels,f,weights);
    m_features.push_back(f);
  }

  eval(data,labels);

  finalize(data, forestConfidences, forestOutOfBagConfidences, forestOutOfBagVoteNum);
}


void SemiNaiveBayesFeature::evalOutOfBagSamples(const matrix<float>& data) {

}

void SemiNaiveBayesFeature::eval(const matrix<float>& data, const std::vector<int> labels) {
  m_confidences.resize(data.size1(), m_hp.numClasses);
  // init to one (due to multiplication)
  for (int sample = 0; sample < (int)data.size1(); sample++) {
    for ( int c = 0; c < m_hp.numClasses; c++) {
      m_confidences(sample,c) = 1.0;
    }
  }

  // Fill confidence matrix
  std::vector<SemiFeature>::iterator it(m_features.begin());
  std::vector<SemiFeature>::iterator end(m_features.end());
  for (int sample = 0; sample < (int)data.size1(); sample++) {
    it = m_features.begin();
    while (it != end) {
      it->eval(data,sample,m_confidences);
      ++it;
    }
    double max = 0.0;
    for ( int c = 0; c < m_hp.numClasses; c++) {
      if (m_confidences(sample,c) > max) {
        m_predictions[sample] = c;
        max = m_confidences(sample,c);
      }
    }
  }

  double error = computeError(labels);
  if (m_hp.verbose) {
    cout << "Error: " << error << endl;
  }

}

void SemiNaiveBayesFeature::eval(const matrix<float>& data, matrix<float>& confidences) {
  m_confidences.resize(data.size1(), m_hp.numClasses);
  // init to one (due to multiplication)
  for (int sample = 0; sample < (int)data.size1(); sample++) {
    for ( int c = 0; c < m_hp.numClasses; c++) {
      m_confidences(sample,c) = 1.0;
    }
  }

  // Fill confidence matrix
  std::vector<SemiFeature>::iterator it(m_features.begin());
  std::vector<SemiFeature>::iterator end(m_features.end());
  for (int sample = 0; sample < (int)data.size1(); sample++) {
    it = m_features.begin();
    while (it != end) {
      it->eval(data,sample,m_confidences);
      ++it;
    }
    for ( int c = 0; c < m_hp.numClasses; c++) {
      confidences(sample,c) += m_confidences(sample,c);
    }
  }

}

void SemiFeature::eval(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences) {
  if ( type == FEATURE_GAUSSIAN ) {
    cout << "SemiNaiveBayesFeature for Gaussians is not implemented yet!!!" << endl;
    exit(1);
  } else {
    evalHistogram(data,sampleIndex,confidences);
  }
}

void SemiFeature::evalGaussian(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences) {
  cout << "SemiNaiveBayesFeature for Gaussians is not implemented yet!!!" << endl;
  exit(1);
}

void SemiFeature::evalHistogram(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences) {
  int tmpIndex;
  if (numBins > 2) {
    tmpIndex = 0;
    for (int nF = 0; nF < (int) indeces.size(); nF++) {
      tmpIndex += (int) ((data(sampleIndex, indeces[nF]) - min[nF])/w[nF]) + nF*indeces.size();
    }
  }
  else {
    if (data(sampleIndex, indeces[0]) < threshold[0] && data(sampleIndex, indeces[1]) < threshold[1]) { //00
      tmpIndex = 0;
    } else if (data(sampleIndex, indeces[0]) < threshold[0] && data(sampleIndex, indeces[1]) > threshold[1]) { //01
      tmpIndex = 1;
    } else if (data(sampleIndex, indeces[0]) > threshold[0] && data(sampleIndex, indeces[1]) < threshold[1]) { //10
      tmpIndex = 2;
    } else if (data(sampleIndex, indeces[0]) > threshold[0] && data(sampleIndex, indeces[1]) > threshold[1]) { //11
      tmpIndex = 3;
    }
  }

  for (int c = 0; c < (int)histogram.size(); c++) {
    confidences(sampleIndex,c) *= histogram[c][tmpIndex];
  }
}
