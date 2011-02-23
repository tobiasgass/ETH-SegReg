#include "naivebayesfeature.h"
#include "utilities.h"
#include <boost/foreach.hpp>

using namespace std;

NaiveBayesFeature::NaiveBayesFeature(const HyperParameters &hp) : NaiveBayes( hp ) {
    m_featureType = (hp.naiveBayesFeatureType == "Gaussian") ? FEATURE_GAUSSIAN : FEATURE_HISTOGRAM;
}

void NaiveBayesFeature::calcMeanAndVariance(const matrix<float>& data, const std::vector<int>& labels, Feature& f) {
    std::vector<double> mean(m_hp.numClasses,0.0);
    std::vector<double> variance(m_hp.numClasses,0.0);
    std::vector<int> numCounts(m_hp.numClasses,0);

    BOOST_FOREACH(int sample, m_inBagSamples) {
        mean[labels[sample]] += data(sample,f.index);
        numCounts[labels[sample]]++;
        variance[labels[sample]] += pow((double)data(sample,f.index),2.0);
    }

    for (int c = 0; c < m_hp.numClasses; c++) {
        mean[c] /= numCounts[c];  // mu
        variance[c] /= numCounts[c]; // E(X^2)
        variance[c] -= pow(mean[c], 2.0);
    }

    f.mean = mean;
    f.variance = variance;
}

// Weighted Statistics
void NaiveBayesFeature::calcMeanAndVariance(const matrix<float>& data, const std::vector<int>& labels, Feature& f,
        const std::vector<double>& weights) {
    std::vector<double> mean(m_hp.numClasses,0.0);
    std::vector<double> variance(m_hp.numClasses,0.0);
    std::vector<double> numCounts(m_hp.numClasses,0.0);

    BOOST_FOREACH(int sample, m_inBagSamples) {
        mean[labels[sample]] += data(sample,f.index)*weights[sample];
        numCounts[labels[sample]] += weights[sample];
        variance[labels[sample]] += pow((double)data(sample,f.index)*weights[sample],2.0);
    }

    for (int c = 0; c < m_hp.numClasses; c++) {
        mean[c] /= numCounts[c];  // mu
        variance[c] /= numCounts[c]; // E(X^2)
        variance[c] -= pow(mean[c], 2.0);
    }

    f.mean = mean;
    f.variance = variance;
}

void NaiveBayesFeature::calcHistogram(const matrix<float>& data, const std::vector<int>& labels, Feature& f) {
    // find min and max values
    f.min = data(m_inBagSamples[0],f.index);
    f.max = data(m_inBagSamples[0],f.index);

    BOOST_FOREACH(int sample, m_inBagSamples) {
        f.min = (data(sample,f.index) < f.min) ? data(sample,f.index) : f.min;
        f.max = (data(sample,f.index) > f.max) ? data(sample,f.index) : f.max;
    }
    f.numBins = (float)m_hp.numHistogramBins;

    std::vector<float> classHistogram((int) f.numBins, 0.0);
    std::vector<std::vector<float> > histogram(m_hp.numClasses,classHistogram);
    std::vector<int> classCounter(m_hp.numClasses,0);

    if ( f.numBins > 2.0) {
        f.w = abs(f.min - f.max)/f.numBins;

        BOOST_FOREACH(int sample, m_inBagSamples) {
            int binIndex = (int) ((data(sample, f.index) - f.min)/f.w);
            binIndex -= (binIndex == f.numBins) ? 1 : 0;
            histogram[labels[sample]][binIndex]++;
            classCounter[labels[sample]]++;
        }
    } else {
        const float numSteps = 20;
        const float stepSize = abs(f.min - f.max)/numSteps;
        float bestEntropy = 1e10;
        float bestThreshold = 0.0;
        std::vector<int> decision(data.size1());
        // Find the best threshold
        for (double threshold = f.min ; threshold < f.max ; threshold += stepSize) {
            int oneCount = 0, zeroCount = 0;
            std::vector<double> zeroClassCount(m_hp.numClasses, 0.0), oneClassCount(m_hp.numClasses, 0.0);

            BOOST_FOREACH(int sample, m_inBagSamples) {
                if (data(sample, f.index) > threshold) {
                    decision[sample] = 1;
                    oneClassCount[labels[sample]]++;
                    oneCount++;
                } else {
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

        BOOST_FOREACH(int sample, m_inBagSamples) {
            int binIndex = (data(sample, f.index) > f.threshold) ? 1 : 0;
            histogram[labels[sample]][binIndex]++;
            classCounter[labels[sample]]++;
        }
    }

    std::vector<std::vector<float> >::iterator it(histogram.begin());
    std::vector<std::vector<float> >::iterator end(histogram.end());
    for (int c = 0;it != end; it++, c++) {
        for (int bin = 0; bin < f.numBins; bin++) {
            (*it)[bin] /= (classCounter[c] + 1e-10);
        }
    }
    f.histogram = histogram;
}

// Weighted Discrete Statistics
void NaiveBayesFeature::calcHistogram(const matrix<float>& data, const std::vector<int>& labels, Feature& f,
                                      const std::vector<double>& weights) {
    // find min and max values
    f.min = data(m_inBagSamples[0],f.index);
    f.max = data(m_inBagSamples[0],f.index);

    BOOST_FOREACH(int sample, m_inBagSamples) {
        f.min = (data(sample,f.index) < f.min) ? data(sample,f.index) : f.min;
        f.max = (data(sample,f.index) > f.max) ? data(sample,f.index) : f.max;
    }
    f.numBins = (float)m_hp.numHistogramBins;

    std::vector<float> classHistogram((int) f.numBins, 0.0);
    std::vector<std::vector<float> > histogram(m_hp.numClasses,classHistogram);
    std::vector<double> classCounter(m_hp.numClasses,0.0);

    if ( f.numBins > 2.0) {
        f.w = abs(f.min - f.max)/f.numBins;

        BOOST_FOREACH(int sample, m_inBagSamples) {
            int binIndex = (int) ((data(sample,f.index) - f.min)/f.w);
            binIndex -= (binIndex == f.numBins) ? 1 : 0;
            histogram[labels[sample]][binIndex] += weights[sample];
            classCounter[labels[sample]] += weights[sample];
        }
    } else {
        const float numSteps = 20;
        const float stepSize = abs(f.min - f.max)/numSteps;
        float bestEntropy = 1e10;
        float bestThreshold = 0.0;
        std::vector<int> decision(data.size1());
        // Find the best threshold
        for (double threshold = f.min ; threshold < f.max ; threshold += stepSize) {
            double oneCount = 0, zeroCount = 0;
            std::vector<double> zeroClassCount(m_hp.numClasses, 0.0), oneClassCount(m_hp.numClasses, 0.0);

            BOOST_FOREACH(int sample, m_inBagSamples) {
                if (data(sample, f.index) > threshold) {
                    decision[sample] = 1;
                    oneClassCount[labels[sample]] += weights[sample];
                    oneCount += weights[sample];
                } else {
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

        BOOST_FOREACH(int sample, m_inBagSamples) {
            int binIndex = (data(sample, f.index) > f.threshold) ? 1 : 0;
            histogram[labels[sample]][binIndex] += weights[sample];
            classCounter[labels[sample]] += weights[sample];
        }
    }

    // Normalise Bins
    std::vector<std::vector<float> >::iterator it(histogram.begin());
    std::vector<std::vector<float> >::iterator end(histogram.end());
    for (int c = 0;it != end; it++, c++) {
        for (int bin = 0; bin < f.numBins; bin++) {
            (*it)[bin] /= (classCounter[c] + 1e-10);
        }
    }
    f.histogram = histogram;
}


void NaiveBayesFeature::train(const matrix<float>& data, const std::vector<int>& labels,
                              matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                              std::vector<int>& forestOutOfBagVoteNum) {
    // Initialize
    initialize(m_hp.numLabeled);

    // Random Subsamples data according to bagratio
    subSample(m_hp.numLabeled);

    // Train the Naive Bayes Classifiers
    std::vector<int> randomFeatures = randPerm(data.size2(),m_hp.numRandomFeatures);

    Feature f;
    f.type = m_featureType;
    if (m_featureType == FEATURE_GAUSSIAN) {
        BOOST_FOREACH(int n, randomFeatures) {
            f.index = n;
            calcMeanAndVariance(data,labels,f);
            m_features.push_back(f);
        }
    } else { // FEATURE_HISTOGRAM
        BOOST_FOREACH(int n, randomFeatures) {
            f.index = n;
            calcHistogram(data,labels,f);
            m_features.push_back(f);
        }
    }

    eval(data,labels);

    finalize(data, forestConfidences, forestOutOfBagConfidences, forestOutOfBagVoteNum);
}

// Weighted Training
void NaiveBayesFeature::train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
                              matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                              std::vector<int>& forestOutOfBagVoteNum, bool init)
{
    if (init)
    {
        // Initialize
        initialize(m_hp.numLabeled);
        // Random Subsamples data according to bagratio
        subSample(m_hp.numLabeled);
    }
    // Train the Naive Bayes Classifiers
    std::vector<int> randomFeatures = randPerm(data.size2(),m_hp.numRandomFeatures);

    Feature f;
    f.type = m_featureType;
    if (m_featureType == FEATURE_GAUSSIAN)
    {
        BOOST_FOREACH(int n, randomFeatures)
        {
            f.index = n;
            calcMeanAndVariance(data,labels,f);
            m_features.push_back(f);
        }
    }
    else   // FEATURE_HISTOGRAM
    {
        BOOST_FOREACH(int n, randomFeatures)
        {
            f.index = n;
            calcHistogram(data,labels,f,weights);
            m_features.push_back(f);
        }
    }

    eval(data,labels);

    finalize(data, forestConfidences, forestOutOfBagConfidences, forestOutOfBagVoteNum);
}

// Weighted Training
void NaiveBayesFeature::retrain(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
                              matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                              std::vector<int>& forestOutOfBagVoteNum, bool init)
{
    if (init)
    {
        // Initialize
        initialize(data.size1());
        // Random Subsamples data according to bagratio
        subSample(data.size1());
    }
    // Train the Naive Bayes Classifiers
    std::vector<int> randomFeatures = randPerm(data.size2(),m_hp.numRandomFeatures);

    Feature f;
    f.type = m_featureType;
    if (m_featureType == FEATURE_GAUSSIAN) {
        BOOST_FOREACH(int n, randomFeatures) {
            f.index = n;
            calcMeanAndVariance(data,labels,f,weights);
            m_features.push_back(f);
        }
    } else { // FEATURE_HISTOGRAM
        BOOST_FOREACH(int n, randomFeatures) {
            f.index = n;
            calcHistogram(data,labels,f,weights);
            m_features.push_back(f);
        }
    }

    eval(data,labels);

    finalize(data, forestConfidences, forestOutOfBagConfidences, forestOutOfBagVoteNum);
}


void NaiveBayesFeature::evalOutOfBagSamples(const matrix<float>& data) {

}

void NaiveBayesFeature::eval(const matrix<float>& data, const std::vector<int> labels) {
    m_confidences.resize(data.size1(), m_hp.numClasses);
    // init to one (due to multiplication)
    for (int sample = 0; sample < (int)data.size1(); sample++) {
        for ( int c = 0; c < m_hp.numClasses; c++) {
            m_confidences(sample,c) = 1.0;
        }
    }

    // Fill confidence matrix
    std::vector<Feature>::iterator it(m_features.begin());
    std::vector<Feature>::iterator end(m_features.end());
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

void NaiveBayesFeature::eval(const matrix<float>& data, matrix<float>& confidences) {
    m_confidences.resize(data.size1(), m_hp.numClasses);
    // init to one (due to multiplication)
    for (int sample = 0; sample < (int)data.size1(); sample++) {
        for ( int c = 0; c < m_hp.numClasses; c++) {
            m_confidences(sample,c) = 1.0;
        }
    }

    // Fill confidence matrix
    std::vector<Feature>::iterator it(m_features.begin());
    std::vector<Feature>::iterator end(m_features.end());
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

void Feature::eval(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences) {
    if ( type == FEATURE_GAUSSIAN ) {
        evalGaussian(data,sampleIndex,confidences);
    } else {
        evalHistogram(data,sampleIndex,confidences);
    }
}

void Feature::evalGaussian(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences) {
    for (int c = 0; c < (int)mean.size(); c++) {
        confidences(sampleIndex,c) *= 1.0/variance[c] * exp(-pow(data(sampleIndex,index) - mean[c],2.0)/(2.0*pow(variance[c],2.0)));
    }
}

void Feature::evalHistogram(const matrix<float>& data, const int sampleIndex, matrix<float>& confidences) {
  int binIndex;
  if (numBins > 2) {
    binIndex = (int) ((data(sampleIndex,index) - min)/w);
    binIndex -= (binIndex == numBins) ? 1 : 0;
    binIndex = (binIndex < 0.0) ? 0 : binIndex;
  }
  else {
    binIndex = (data(sampleIndex, index) > threshold) ? 1 : 0;
  }
  for (int c = 0; c < (int)histogram.size(); c++) {
    confidences(sampleIndex,c) *= histogram[c][binIndex];
  }
}
