#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <libxml/tree.h>
//#include <libxml/parser.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <vector>

using namespace boost::numeric;
using namespace std;

unsigned int getDevRandom();
double _rand(int i=-1);
int randomNumber(int min, int max);
int randomNumber( int limit);
double randomDouble( double limit);

double getTime();
double timeIt(int reset);

void dispVector(const std::vector<int>& inVector);
void dispVector(const std::vector<float>& inVector);
void dispVector(const std::vector<double>& inVector);
void dispMatrix(const ublas::matrix<float>& inMat);
void dispMatrix(const ublas::matrix<float>& inMat, const int rowIndex);

std::vector<int> subSampleWithReplacement(const int numSamples);
std::vector<int> unique(const std::vector<int> inIndex);
std::vector<int> setDiff(const std::vector<int> inSetA, const int numSamples);
std::vector<int> randPerm(const int inNum);
std::vector<int> randPerm(const int inNum, const int inPart);
void subSampleWithoutReplacement(const int numSamples, const int numInBagSamples,
                                 std::vector<int> &inBagSamples, std::vector<int> &outOfBagSamples);

// ######## XML Reading/Writing #########
std::string readStringProp( const xmlNodePtr node, const std::string& propName,
			    const std::string& defaultValue = "" );

int readIntProp( const xmlNodePtr node, const std::string& propName,
		 int defaultValue = 0 );

double readDoubleProp( const xmlNodePtr node, const std::string& propName,
		       double defaultValue = 0.0 );

std::string readString( const xmlDocPtr doc, const xmlNodePtr node );

//ImageType readImageTypeProp( const xmlNodePtr node );

//Rect readRect( const xmlNodePtr rectNode );

void addIntProp( xmlNodePtr node, const char* propName, int value );

void addDoubleProp( xmlNodePtr node, const char* propName, double value );

//void addImageTypeProp( xmlNodePtr node, ImageType imageType );

//xmlNodePtr saveRect( const std::string& name, const Rect& rect );
//xmlNodePtr saveBin( const std::string& name, const short pos );
//xmlNodePtr saveHistBin( const std::string& name, const short pos, const double value );

std::string intToString(int num);
std::string doubleToString(double num);

void fillWithRandomNumbers(std::vector<float>& tmpWeights);

#endif /* UTILITIES_H_ */
