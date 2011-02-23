#ifndef ONLINE_NODE_HYPERPLANE_
#define ONLINE_NODE_HYPERPLANE_

#include "onlinenode.h"
#include "hyperparameters.h"
#include "utilities.h"
#include <boost/shared_ptr.hpp>
#include <string>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>

using namespace boost::numeric::ublas;
using namespace std;

// The struct holds the statistics for one threshold for one feature
// Both threshold and feature are kept fixed

class OnlineNodeHyperplane : public OnlineNode
{
 public:
    typedef boost::shared_ptr<OnlineNodeHyperplane> Ptr;

    OnlineNodeHyperplane(const HyperParameters &hp, int depth, int numFeatures);
    OnlineNodeHyperplane(const HyperParameters &hp, int depth, int numFeatures, std::vector<int>& labelStatistics);

    virtual void train(const std::vector<float>& sample, const int label);

    virtual double eval(const std::vector<float>& sample, const int label);

    virtual void eval(const matrix<float>& data, const std::vector<int>& sampleIndeces,
                      matrix<float>& confidences, std::vector<int>& predictions);

    virtual std::vector<int> bestFeature() { return std::vector<int>(1,m_bestFeature); };
    virtual std::vector<float> bestWeight() { return std::vector<float>(1,1.0f); };
    virtual float bestThreshold() { return m_bestThreshold; };

 private:
    void updateHypotheses(const std::vector<float>& sample, const int label);

    void evalNode(const matrix<float>& data, const std::vector<int>& inBagSamples,
                  std::vector<int>& leftNodeSamples, std::vector<int>& rightNodeSamples);

    std::pair<float, float> calcGiniAndThreshold(const std::vector<int>& labels,
                                                  const std::vector<std::pair<float, int> >& responses);
    void cleanUpAndFreeze();
    double m_gini;
    int m_bestFeature;
    float m_bestThreshold;
    std::vector<int> m_bestIndeces;
    std::vector<float> m_bestWeights;
    std::vector<Feature> m_features;
};


#endif // ONLINE_NODE_HYPERPLANE_


