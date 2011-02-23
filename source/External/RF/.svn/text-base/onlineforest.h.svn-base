#ifndef ONLINE_FOREST_H_
#define ONLINE_FOREST_H_

#include "onlinetree.h"
#include "data.h"
#include <iostream>
#include "hyperparameters.h"
#include <string>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <libxml/tree.h>
#include <libxml/parser.h>
using namespace boost::numeric::ublas;

#ifdef WIN32
#pragma warning( disable : 4290 )
//warning C4290:  C++ exception specification ignored except to indicate a function
//                is not __declspec(nothrow)
#endif

class OnlineForest
{
public:
    OnlineForest(const HyperParameters &hp);
    OnlineForest();
    ~OnlineForest();

    void train(const matrix<float>& data, const std::vector<int>& labels, bool use_gpu = false);

    void eval(const matrix<float>& data, const std::vector<int>& labels, bool use_gpu = false);

    std::vector<int> getPredictions() const { return m_predictions; };
    matrix<float> getConfidences() const { return m_confidences; };

    void save(const std::string &name = "default");
    void load(const std::string &name = "default");

    double oobe() const;

 protected:
    std::vector<OnlineTree> m_trees;

    const HyperParameters *m_hp;

    matrix<float> m_confidences;
    std::vector<int> m_predictions;

    void writeError(const std::string& dataFileName, double error);

    void initialize(const int numSamples);

    void trainByCPU(const matrix<float>& data, const std::vector<int>& labels);
    void evalByCPU(const matrix<float>& data, const std::vector<int>& labels);

    double computeError(const std::vector<int>& labels);
    double computeError(const std::vector<int>& labels, const matrix<float>& confidences,
                        const std::vector<int>& voteNum);

};

#endif /* ONLINE_FOREST_H_ */

