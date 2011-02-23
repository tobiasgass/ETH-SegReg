#include "naivebayes.h"
#include "utilities.h"
#include <boost/foreach.hpp>

using namespace std;

NaiveBayes::NaiveBayes(const HyperParameters &hp) : m_hp( hp )
{

}


NaiveBayes::~NaiveBayes()
{
}


void NaiveBayes::initialize(const int numSamples)
{
    m_confidences.resize(numSamples, m_hp.numClasses);
    m_predictions.resize(numSamples);

    m_inBagSamples.clear();
    m_outOfBagSamples.clear();
}


void NaiveBayes::subSample(const int numSamples)
{
    if (m_hp.useSubSamplingWithReplacement)
    {
        m_inBagSamples = subSampleWithReplacement(numSamples);
        m_outOfBagSamples = setDiff(m_inBagSamples, numSamples);
    }
    else
    {
        subSampleWithoutReplacement(numSamples, static_cast<int>(floor(numSamples * m_hp.bagRatio)),
                                    m_inBagSamples, m_outOfBagSamples);
    }
}

void NaiveBayes::finalize(const matrix<float>& data,
                    matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                    std::vector<int>& forestOutOfBagVoteNum)
{

    // Fill the confidence of the forest
    for (int nSamp = 0; nSamp < m_hp.numLabeled; nSamp++) {
        if (m_hp.useSoftVoting) {
            for (int nClass = 0; nClass < m_hp.numClasses; nClass++) {
                forestConfidences(nSamp,nClass) += m_confidences(nSamp,nClass);
            }
        }
        else {
            forestConfidences(nSamp, m_predictions[nSamp])++;
        }
    }

     // Fill the out of bag confidences and vote count
    BOOST_FOREACH(int n, m_outOfBagSamples) {
        forestOutOfBagVoteNum[n]++;

        if (m_hp.useSoftVoting) {
            for (int nClass = 0; nClass < m_hp.numClasses; nClass++) {
                forestOutOfBagConfidences(n,nClass) += m_confidences(n,nClass);
            }
        }
        else {
            forestOutOfBagConfidences(n, m_predictions[n])++;
        }
    }

    // Clean up
    //clean();
}

void NaiveBayes::clean() {
  m_confidences.clear();
}

double NaiveBayes::computeError(const std::vector<int>& labels)
{
    int error = 0;
    for(int sample = 0; sample < (int)labels.size(); sample++) {
        error += (m_predictions[sample] != labels[sample]) ? 1 : 0;
    }
    return static_cast<double>(error)/static_cast<double>(labels.size());
}


