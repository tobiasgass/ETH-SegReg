#include "node.h"
#include "hyperparameters.h"
#include "utilities.h"
#include <boost/shared_ptr.hpp>
#include <string>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <libxml/tree.h>
#include <libxml/parser.h>
#ifndef NODE_INFO_GAIN_
#define NODE_INFO_GAIN_

using namespace boost::numeric::ublas;
using namespace std;

const char* const NODE_INFO_GAIN = "nodeInfoGain";

class NodeInfoGain : public Node
{
 public:
    typedef boost::shared_ptr<NodeInfoGain> Ptr;

    NodeInfoGain(const HyperParameters &hp, int depth);
    NodeInfoGain(const HyperParameters &hp, int depth, int reset);
    NodeInfoGain(const HyperParameters &hp, int reset, const xmlNodePtr nodeNode);

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

    virtual std::vector<int> bestFeature() { return std::vector<int>(1,m_bestFeature); };
    virtual std::vector<float> bestWeight() { return std::vector<float>(1,1.0f); };
    virtual float bestThreshold() { return m_bestThreshold; };

private:
    void findHypotheses(const matrix<float>& data, const std::vector<int>& labels,
                        const std::vector<int>& inBagSamples, const std::vector<int>& randFeatures);
    void findHypothesesLU(const matrix<float>& data, const std::vector<int>& labels,
                          const std::vector<int>& inBagSamples, const std::vector<int>& randFeatures, int numRandTries);
    void findHypotheses(const matrix<float>& data, const std::vector<int>& labels,
                        const std::vector<double>& weights,
                        const std::vector<int>& inBagSamples, const std::vector<int>& randFeatures);

    void evalNode(const matrix<float>& data, const std::vector<int>& inBagSamples,
                  std::vector<int>& leftNodeSamples, std::vector<int>& rightNodeSamples);

    std::pair<float, float> calcInfoGainAndThreshold(const std::vector<int>& labels,
                                                      const std::vector<std::pair<float, int> >& responses);
    std::pair<float, float> calcInfoGainAndThreshold(const std::vector<int>& labels, const std::vector<double>& weights,
                                                      const std::vector<std::pair<float, int> >& responses);
    std::pair<float, float> calcClusterScoreAndThreshold(const matrix<float>& data, const std::vector<int>& inBagSamples,
                                                          const std::vector<std::pair<float, int> >& responses);

    xmlNodePtr saveFeature() const;
    double m_infoGain;
    int m_bestFeature;
    float m_bestThreshold;
};


#endif // NODE_INFO_GAIN_
