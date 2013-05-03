#include "data.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/foreach.hpp>

using namespace std;

Pair::Pair(const std::vector<float>& x1,const std::vector<float>& x2,const int label) :
m_x1(x1), m_x2(x2), m_label(label)
{
}

Pair::~Pair()
{
}

// FileData
/**
 * DataFile Format should be:
 * datatype: eg: int/float/double/etc.)
 * size: eg: numRows numCols
 * data (last column should be the labels)
 */
FileData::FileData()
{
}

FileData::FileData(const std::string& dataFileName, const std::string& labelsFileName) : m_dataFileName( dataFileName ),
        m_labelsFileName( labelsFileName )
{

}

FileData::~FileData()
{
    // Implement me
}

void FileData::setDataFileName(const std::string& fileName)
{
    m_dataFileName = fileName;
}

void FileData::setLabelsFileName(const std::string& fileName)
{
    m_labelsFileName = fileName;
}

void FileData::readData()
{
    std::fstream myfile(m_dataFileName.c_str(),ios::in);
    if (myfile.is_open())
    {
        std::string line;
        // get datatype
        getline(myfile,line);
        // get size
        getline(myfile,line);
        stringstream ss(line);
        string tmp;
        int numRows, numCols;
        ss >> numRows;
        ss >> numCols;
        // pop-up the dense/sparse thing
        getline(myfile,line);

        m_data.resize(numRows,numCols);

        for (int row = 0; row < numRows; row++)
        {
            getline(myfile,line);
            stringstream ss2(line);
            for (int col = 0; col < numCols; col++)
            {
                ss2 >> m_data(row,col);
            }
        }

        myfile.close();
    }

    else
    {
        cout << "Unable to open file" << m_dataFileName << endl;
        exit(-1);
    }
}

void FileData::readLabels()
{
    std::fstream myfile(m_labelsFileName.c_str(),ios::in);
    if (myfile.is_open())
    {
        std::string line;
        // get datatype
        getline(myfile,line);
        // get size
        getline(myfile,line);
        stringstream ss(line);
        string tmp;
        int numLabels;
        ss >> numLabels;

        // pop-up the dense/sparse thing
        getline(myfile,line);

        m_labels.resize(numLabels);
        for (int row = 0; row < numLabels; row++)
        {
            getline(myfile, line);
            stringstream ss2(line);
            ss2 >> m_labels[row];
        }

        myfile.close();
    }

    else
    {
        cout << "Unable to open file" << m_dataFileName << endl;
        exit(-1);
    }
}

void FileData::writeData()
{
    std::fstream myfile;
    std::string dataFileName = m_dataFileName;
    dataFileName += ".data";
    myfile.open(dataFileName.c_str(), std::ios::out);

    myfile << "float" << endl;
    myfile << m_data.size1() << " " << m_data.size2() << endl;
    for (size_t row = 0; row < m_data.size1(); row++  )
    {
        for (size_t col = 0; col < m_data.size2(); col++)
        {
            myfile << m_data(row,col) << " ";
        }
        myfile << endl;
    }

    myfile.flush();
    myfile.close();
}

void FileData::writeLabels()
{
    std::fstream myfile;
    std::string dataFileName = m_labelsFileName;
    dataFileName += ".labels";
    myfile.open(dataFileName.c_str(), std::ios::out);

    myfile << "int" << endl;
    myfile << m_labels.size() << " " << 1 << endl;

    for (size_t row = 0; row < m_labels.size(); row++)
    {
        myfile << m_labels[row];
        myfile << endl;
    }

    myfile.flush();
    myfile.close();
}


void FileData::createPairs()
{
    int numSamples = m_data.size1();
    int numFeatures = m_data.size2();
    for(int p = 0; p < numSamples;p+=2) {
        std::vector<float> x1;
        std::vector<float> x2;
        for( int feat = 0; feat < numFeatures; feat++) {
            x1.push_back(m_data(p,feat));
            x2.push_back(m_data(p+1,feat));
        }
        int label = (m_labels[p] == 1) ? 1 : 0;
        Pair newPair(x1,x2,label);
        m_pairs.push_back(newPair);
    }
}















