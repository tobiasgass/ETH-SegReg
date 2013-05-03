/*
 * data.h
 *
 *  Created on: Jan 26, 2009
 *      Author: leisti
 */

#include <string>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
using namespace std;
using namespace boost::numeric::ublas;

#ifndef DATA_H_
#define DATA_H_

class Pair
{
public:
        Pair(const std::vector<float>& x1,const std::vector<float>& x2,const int label);
        ~Pair();
        inline std::vector<float> x1() const { return m_x1; }
        inline std::vector<float> x2() const { return m_x2; }
        inline int label() const { return m_label; }
private:
    std::vector<float> m_x1;
    std::vector<float> m_x2;
    int m_label;
};

class FileData
{
public:
    FileData();
    FileData(const std::string& dataFileName, const std::string& labelsFileName);
    ~FileData();
    //! File operations
    void readData();
    void readLabels();
    void writeData();
    void writeLabels();

    void setData(const matrix<float>& data) {m_data = data;};
    void setLabels(const std::vector<int>& labels) { m_labels = labels; };
    inline matrix<float> getData() const { return m_data; };
    inline std::vector<int> getLabels() const { return m_labels; };
    void setDataFileName(const std::string& fileName);
    void setLabelsFileName(const std::string& fileName);
    int getNumDataSamples() const {return m_data.size1();};
    int getNumDataFeatures() const {return m_data.size2();};


    void createPairs();
    inline std::vector<Pair> getPairs() const { return m_pairs; }
private:
    std::string m_dataFileName;
    std::string m_labelsFileName;
    matrix<float> m_data;
    std::vector<int> m_labels;
    std::vector<Pair> m_pairs;
};
#endif /* DATA_H_ */
