#include "onlinetree.h"
#include "onlinenodegini.h"
#include "onlinenodehyperplane.h"
#include <cmath>
#include "utilities.h"
#include <boost/foreach.hpp>

OnlineTree::OnlineTree(const HyperParameters &hp, int numFeatures) : m_hp( &hp )
{
    if (hp.useRandProj)
    {
        m_rootNode = OnlineNode::Ptr(new OnlineNodeHyperplane(hp,0,numFeatures));
    }
    else
    {
        m_rootNode = OnlineNode::Ptr(new OnlineNodeGini(hp,0,numFeatures));
    }
}

OnlineTree::~OnlineTree()
{
}

void OnlineTree::finalize(const matrix<float>& data,
                          matrix<float>& forestConfidences, matrix<float>& forestOutOfBagConfidences,
                          std::vector<int>& forestOutOfBagVoteNum)
{
    // Fill the confidence of the forest
    for (int nSamp = 0; nSamp < m_hp->numLabeled; nSamp++)
    {
        if (m_hp->useSoftVoting)
        {
            for (int nClass = 0; nClass < m_hp->numClasses; nClass++)
            {
                forestConfidences(nSamp,nClass) += m_confidences(nSamp,nClass);
            }
        }
        else
        {
            forestConfidences(nSamp, m_predictions[nSamp])++;
        }
    }

    // Fill the out of bag confidences and vote count
    BOOST_FOREACH(int n, m_outOfBagSamples)
    {
        forestOutOfBagVoteNum[n]++;

        if (m_hp->useSoftVoting)
        {
            for (int nClass = 0; nClass < m_hp->numClasses; nClass++)
            {
                forestOutOfBagConfidences(n,nClass) += m_confidences(n,nClass);
            }
        }
        else
        {
            forestOutOfBagConfidences(n, m_predictions[n])++;
        }
    }

    // Clean up
    clean();
}

void OnlineTree::clean()
{
    m_confidences.clear();
}

void OnlineTree::train(const std::vector<float>& sample, const int label)
{
    // Initialize
    //initialize(m_hp->numLabeled);

    // Random Subsamples data according to bagratio
    //subSample(m_hp->numLabeled);

    // Train the root Node
    m_rootNode->train(sample,label);
    //evalOutOfBagSamples(data);

    if (m_hp->verbose)
    {
        cout << "Trained a tree with " << m_rootNode->numNodes() << " nodes." << endl;
        // cout << "Training error = " << computeError(labels) << ", in bag = ";
        // cout << computeError(labels, m_inBagSamples) << ", out of bag = ";
        // cout << computeError(labels, m_outOfBagSamples) <<  endl;
    }
}

void OnlineTree::evalOutOfBagSamples(const matrix<float>& data)
{
    m_rootNode->eval(data, m_outOfBagSamples, m_confidences, m_predictions);
}

double OnlineTree::eval(const std::vector<float>& sample, const int label)
{
    // Initialize
    return m_rootNode->eval(sample, label);

    //if (m_hp->verbose) {
    // cout << "Test error: " << computeError(labels) << endl;
    //}
}

void OnlineTree::initialize(const int numSamples)
{
    m_confidences.resize(numSamples, m_hp->numClasses);
    m_predictions.resize(numSamples);

    m_inBagSamples.clear();
    m_outOfBagSamples.clear();
}

void OnlineTree::subSample(const int numSamples)
{
    if (m_hp->useSubSamplingWithReplacement)
    {
        m_inBagSamples = subSampleWithReplacement(numSamples);
        m_outOfBagSamples = setDiff(m_inBagSamples, numSamples);
    }
    else
    {
        subSampleWithoutReplacement(numSamples, static_cast<int>(floor(numSamples * m_hp->bagRatio)),
                                    m_inBagSamples, m_outOfBagSamples);
    }
}

void OnlineTree::subSample(const int numSamples, const std::vector<double>& weights)
{
    std::vector<int> tmpInBag;
    if (m_hp->useSubSamplingWithReplacement)
    {
        tmpInBag = subSampleWithReplacement(numSamples);
    }
    else
    {
        subSampleWithoutReplacement(numSamples, static_cast<int>(floor(numSamples * m_hp->bagRatio)),
                                    tmpInBag, m_outOfBagSamples);
    }

    // Weight Trimming
    BOOST_FOREACH(int n, tmpInBag)
    {
        if (weights[n] > 1e-4)
        {
            m_inBagSamples.push_back(n);
        }
    }
    m_outOfBagSamples = setDiff(m_inBagSamples, numSamples);
}

double OnlineTree::computeError(const std::vector<int>& labels, const std::vector<int>& sampleIndeces)
{
    double error = 0.0;
    int sampleCount = 0;
    BOOST_FOREACH(int n, sampleIndeces)
    {
        if (sampleCount < (int) labels.size())
        {
            error += (m_predictions[n] != labels[n]) ? 1.0 : 0.0;
        }
        else
        {
            break;
        }

        sampleCount++;
    }
    return error/(double) labels.size();
}

double OnlineTree::computeError(const std::vector<int>& labels)
{
    double error = 0.0;
    std::vector<int>::const_iterator itr(m_predictions.begin());
    std::vector<int>::const_iterator labelItr(labels.begin()), labelEnd(labels.end());
    for (; labelItr != labelEnd; itr++, labelItr++)
    {
        error += (*itr != *labelItr) ? 1.0 : 0.0;
    }
    return error/(double) labels.size();
}

/*
std::vector<int> OnlineTree::getNodeLabels() {
  std::set<int> nodeLabels;
  m_rootNode->getLabels(nodeLabels);

  std::vector<int> out;
  std::set<int>::const_iterator itr = nodeLabels.begin(), end = nodeLabels.end();
  for (; itr != end; itr++) {
    out.push_back(*itr);
  }

  return out;
}
*/
