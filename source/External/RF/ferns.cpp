#include "ferns.h"
#include "seminaivebayeshyperplane.h"
#include "seminaivebayesfeature.h"
#include <boost/foreach.hpp>

Ferns::Ferns(const HyperParameters &hp) : m_hp( hp )
{
}

Ferns::~Ferns()
{
}

void Ferns::train(const matrix<float>& data, const std::vector<int>& labels, bool use_gpu)
{
    // Initialize
    initialize(data.size1());

    if (m_hp.useGPU || use_gpu)
    {
        std::vector<double> weights(labels.size(),1);
        //trainByGPU(data,labels, weights);
        std::cout << "ERROR: Not implemented yet! Jakob, do it!" << std::endl;
    }
    else
        trainByCPU(data,labels);

}

void Ferns::initialize(const int numSamples)
{
    m_confidences.resize(numSamples, m_hp.numClasses);
    m_predictions.resize(numSamples);

    for (int n = 0; n < numSamples; n++)
    {
        for (int m = 0; m < m_hp.numClasses; m++)
        {
            m_confidences(n, m) = 0.0;
        }
    }
}


void Ferns::trainByCPU(const matrix<float>& data, const std::vector<int>& labels)
{
    std::vector<int> outOfBagVoteCount(data.size1(), 0);
    matrix<float> outOfBagConfidences(data.size1(),m_hp.numClasses);
    for ( unsigned int i = 0; i < data.size1(); i++)
    {
        for ( int j = 0; j < m_hp.numClasses; j++)
        {
            outOfBagConfidences(i,j) = 0.0;
        }
    }

    HyperParameters tmpHP = m_hp;
    tmpHP.verbose = false;
    m_naiveBayesClassifiers.clear(); // be sure that they are empty
    for (int i = 0; i < m_hp.numTrees; i++)
    {
        SemiNaiveBayes::Ptr naiveClassifier;
        if (m_hp.useRandProj) {
            naiveClassifier = SemiNaiveBayes::Ptr(new SemiNaiveBayesHyperplane(tmpHP));
        }
        else {
            naiveClassifier = SemiNaiveBayes::Ptr(new SemiNaiveBayesFeature(tmpHP));
        }
        naiveClassifier->train(data,labels, m_confidences, outOfBagConfidences, outOfBagVoteCount);
        m_naiveBayesClassifiers.push_back(naiveClassifier);

        //double error = naiveClassifier->computeError(labels);
        //cout << "Error: " << error << endl;
        cout << "." << flush;
    }

    cout << endl;
    eval(data,labels);
}

void Ferns::eval(const matrix<float>& data, const std::vector<int>& labels)
{
    m_confidences.resize(data.size1(), m_hp.numClasses);
    m_predictions.resize(data.size1());
    // init to one (due to multiplication)
    for (int sample = 0; sample < (int)data.size1(); sample++)
    {
        for ( int c = 0; c < m_hp.numClasses; c++)
        {
            m_confidences(sample,c) = 0.0;
        }
    }


    BOOST_FOREACH(SemiNaiveBayes::Ptr nbc, m_naiveBayesClassifiers)
    {
        nbc->eval(data, m_confidences);
    }


    for (int sample = 0; sample < (int)m_confidences.size1();sample++)
    {
        double maxClass = 0.0;
        for ( int c = 0; c < m_hp.numClasses; c++)
        {
            if (m_confidences(sample,c) > maxClass)
            {
                m_predictions[sample] = c;
                maxClass = m_confidences(sample,c);
            }
        }
    }

    double error = computeError(labels);
    cout << "Overall Error: " << error << endl;

}


double Ferns::computeError(const std::vector<int>& labels)
{
    int error = 0;
    for (int sample = 0; sample < (int)labels.size(); sample++)
    {
        error += (m_predictions[sample] != labels[sample]) ? 1 : 0;
    }
    return static_cast<double>(error)/static_cast<double>(labels.size());
}
