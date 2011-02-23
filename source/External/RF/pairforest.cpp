#include "pairforest.h"
#include <boost/foreach.hpp>
#include <omp.h>

PairForest::PairForest(const HyperParameters &hp) : m_hp(hp)
{
}

PairForest::~PairForest()
{
}

void PairForest::train(const std::vector<Pair>& pairs)
{
    cout << "Train pair forest!" << endl;
    HyperParameters tmpHP = m_hp;
    tmpHP.verbose = false;

    omp_set_num_threads(8);

    int numLeafs = 0;

    #pragma omp parallel for
    for (int i = 0; i < m_hp.numTrees; i++)
    {
        if (m_hp.verbose && !(10*i%m_hp.numTrees))
        {
            cout << 100*i/m_hp.numTrees << "% " << flush;
        }

        PairTree t(tmpHP);
        t.train(pairs);
        numLeafs += t.numLeafs();
        m_trees.push_back(t);
    }

    cout << endl << "\tThe forest has " << numLeafs << " leafs." << endl;

    //test(pairs);

}

void PairForest::test(const std::vector<Pair>& pairs)
{

    std::vector<int> path1;
    std::vector<int> path2;

    std::vector<Pair>::const_iterator pairIt( pairs.begin() );
    std::vector<Pair>::const_iterator pairEnd( pairs.end() );
    std::vector<PairTree>::iterator treeIt( m_trees.begin() );
    std::vector<PairTree>::iterator treeEnd( m_trees.end() );

    int numWrongPos = 0;
    int numWrongNeg = 0;
    int totalPosPairs = 0;
    int totalNegPairs = 0;

    while ( pairIt != pairEnd )
    {
        treeIt = m_trees.begin();
        int numSameIndices = 0;
        while ( treeIt != treeEnd )
        {
            int index1 = treeIt->eval( pairIt->x1() );
            int index2 = treeIt->eval( pairIt->x2() );
            numSameIndices += (index1 == index2) ? 1 : 0;
            treeIt++;
        }
        //cout << "label: " << pairIt->label() << " predicted conf: " << (double)numSameIndices/(double)m_trees.size() << endl;
        numWrongPos += ( pairIt->label() == 1 && (double)numSameIndices/(double)m_trees.size() < 0.5) ? 1 : 0;
        numWrongNeg += ( pairIt->label() != 1 && (double)numSameIndices/(double)m_trees.size() >= 0.5) ? 1 : 0;
        if (pairIt->label() == 1) { totalPosPairs++; } else { totalNegPairs++; }
        ++pairIt;
    }
    cout << "pos error: " << (double)numWrongPos/(double)totalPosPairs << ", neg error: "
            << (double)numWrongNeg/(double)totalNegPairs << endl;

//    for(int i = 0; i < m_trees.size();i++) {
//        cout << m_trees[i].getDepth() << ", ";
//    }
//    cout << endl;
}

std::vector<int> PairForest::getPath(const std::vector<Pair>& pairs, const int treeIndex)
{
    return m_trees[treeIndex].getPath(pairs[0].x1());
}
