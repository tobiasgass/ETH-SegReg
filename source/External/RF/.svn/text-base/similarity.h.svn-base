#ifndef SIMILARITY_H_
#define SIMILARITY_H_

#include "forest.h"
#include "tree.h"
#include "data.h"
#include <iostream>
#include "hyperparameters.h"
#include <string>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <libxml/tree.h>
#include <libxml/parser.h>
using namespace boost::numeric::ublas;

#ifdef USE_CUDA
#include "cudatemplates/devicememorypitched.hpp"
#include "cudatemplates/devicememorylinear.hpp"
//#include "cudatemplates/hostmemoryreference.hpp"
#include "cudatemplates/hostmemoryheap.hpp"
//#include "cudatemplates/copy.hpp"
#include "cuda/icg_rf_interface.cuh"
#endif

#ifdef WIN32
#pragma warning( disable : 4290 )
//warning C4290:  C++ exception specification ignored except to indicate a function
//                is not __declspec(nothrow)
#endif

class Similarity
{
public:
    Similarity(const HyperParameters &hp);
    Similarity();
    ~Similarity();

    void train(const matrix<float>& data, const std::vector<int>& labels);
    void train(const matrix<float>& data, const std::vector<int>& labels, const double posW);    
    matrix<float> getSimConf(const matrix<float>& data);
    matrix<float> getSimConf(const matrix<float>& data, const std::vector<int>& labels);
    matrix<float> getSimConfChi2(const matrix<float>& data, const std::vector<int>& labels);
private:
    HyperParameters m_hp;
    std::vector<int> m_labels;
    std::vector<Tree> m_trees;
};

#endif /* SIMILARITY_H_ */
