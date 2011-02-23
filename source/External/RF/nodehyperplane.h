#include "node.h"
#include "hyperparameters.h"
#include "utilities.h"
#include <boost/shared_ptr.hpp>
#include <string>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <libxml/tree.h>
#include <libxml/parser.h>

#ifndef NODE_HYPER_PLANE_
#define NODE_HYPER_PLANE_

using namespace boost::numeric::ublas;
using namespace std;

const char* const NODE_HYPER_PLANE = "nodeHyperPlane";

class NodeHyperPlane : public Node
{
 public:
    typedef boost::shared_ptr<NodeHyperPlane> Ptr;

    NodeHyperPlane(const HyperParameters &hp, int depth);
    NodeHyperPlane(const HyperParameters &hp, int depth, int reset);
    NodeHyperPlane(const HyperParameters &hp, int reset, const xmlNodePtr nodeNode);
    virtual xmlNodePtr save() const;

    virtual NODE_TRAIN_STATUS train(const matrix<float>& data, const std::vector<int>& labels, std::vector<int>& inBagSamples,
                                    matrix<float>& confidences, std::vector<int>& predictions);
    virtual NODE_TRAIN_STATUS train(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
                                    std::vector<int>& inBagSamples, matrix<float>& confidences, std::vector<int>& predictions);

    virtual NODE_TRAIN_STATUS trainLU(const matrix<float>& data, const std::vector<int>& labels, std::vector<int>& inBagSamples,
                                      matrix<float>& confidences, std::vector<int>& predictions);

    virtual void refine(const matrix<float>& data, const std::vector<int>& labels,
                                    std::vector<int>& samples, matrix<float>& confidences, std::vector<int>& predictions);

    virtual void refine(const matrix<float>& data, const std::vector<int>& labels, const std::vector<double>& weights,
                                    std::vector<int>& samples,matrix<float>& confidences, std::vector<int>& predictions);

    virtual void eval(const matrix<float>& data, const std::vector<int>& sampleIndeces,
                      matrix<float>& confidences, std::vector<int>& predictions);

    virtual void getPath(const matrix<float>& data, const std::vector<int>& sampleIndeces, std::vector<std::vector<int> >& path);

    virtual std::vector<int> bestFeature() { return m_bestFeatures; };
    virtual std::vector<float> bestWeight() { return m_bestWeights; };
    virtual float bestThreshold() { return m_bestThreshold; };

private:
    void findHypotheses(const matrix<float>& data, const std::vector<int>& labels,
                        const std::vector<int>& inBagSamples, const std::vector<int>& randFeatures, int numRandTries);
    void findHypothesesLU(const matrix<float>& data, const std::vector<int>& labels,
                          const std::vector<int>& inBagSamples, const std::vector<int>& randFeatures, int numRandTries);
    void findHypotheses(const matrix<float>& data, const std::vector<int>& labels,
                        const std::vector<double>& weights,
                        const std::vector<int>& inBagSamples, const std::vector<int>& randFeatures, int numRandTries);

    bool clusterOrGini();

    void evalNode(const matrix<float>& data, const std::vector<int>& inBagSamples,
                  std::vector<int>& leftNodeSamples, std::vector<int>& rightNodeSamples);

    std::pair<float, float> calcGiniAndThreshold(const std::vector<int>& labels,
                                                 const std::vector<std::pair<float, int> >& responses);
    std::pair<float, float> calcGiniAndThreshold(const std::vector<int>& labels, const std::vector<double>& weights,
                                                 const std::vector<std::pair<float, int> >& responsess, const bool useUnlabeledData = true);

    std::pair<float, float> calcClusterScoreAndThreshold(const matrix<float>& data, const std::vector<int>& inBagSamples,
                                                          const std::vector<double>& weights,
                                                         const std::vector<std::pair<float, int> >& responses);
    std::pair<float, float> calcClusterScoreAndThreshold(const matrix<float>& data, const std::vector<int>& inBagSamples,
                                                          const std::vector<std::pair<float, int> >& responses);

    std::pair<float, float> calcInfoGainAndThreshold(const std::vector<int>& labels,
                                                      const std::vector<std::pair<float, int> >& responses);
    std::pair<float, float> calcInfoGainAndThreshold(const std::vector<int>& labels, const std::vector<double>& weights,
                                                      const std::vector<std::pair<float, int> >& responses);

    xmlNodePtr saveFeature() const;
    double m_gini;
    std::vector<int> m_bestFeatures;
    std::vector<float> m_bestWeights;
    float m_bestThreshold;
};


#endif // NODE_HYPERPLANE_
