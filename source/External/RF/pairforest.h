#ifndef PAIR_FOREST_H_
#define PAIR_FOREST_H_

#include "pairtree.h"
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
#include "cudatemplates/hostmemoryheap.hpp"
#include "cuda/icg_rf_interface.cuh"
#endif

#ifdef WIN32
#pragma warning( disable : 4290 )
//warning C4290:  C++ exception specification ignored except to indicate a function
//                is not __declspec(nothrow)
#endif

class PairForest
{
public:
    PairForest(const HyperParameters &hp);
    PairForest(){};
    ~PairForest();

    void train(const std::vector<Pair>& pairs);
    void test(const std::vector<Pair>& pairs);
    std::vector<int> getPath(const std::vector<Pair>& pairs, const int treeIndex);

 private:
    HyperParameters m_hp;
    std::vector<PairTree> m_trees;
};

#endif /* PAIR_FOREST_H_ */

