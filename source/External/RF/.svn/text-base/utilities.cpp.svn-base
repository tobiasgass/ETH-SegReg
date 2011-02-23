#include "utilities.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <set>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <boost/foreach.hpp>

// time measurement
#ifdef WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#if WIN32
#define snprintf sprintf_s
#endif

using namespace std;

#ifndef WIN32
unsigned int getDevRandom() {
    ifstream devFile("/dev/urandom", ios::binary);
    unsigned int outInt = 0;
    char tempChar[sizeof(outInt)];

    devFile.read(tempChar, sizeof(outInt));
    outInt = atoi(tempChar);

    devFile.close();

    return outInt;
}
#endif

double getTime()
{
#ifdef USE_GPU
  cudaThreadSynchronize();
#endif
#ifdef WIN32
  LARGE_INTEGER current_time,frequency;
  QueryPerformanceCounter (&current_time);
  QueryPerformanceFrequency(&frequency);
  return current_time.QuadPart*1000.0/frequency.QuadPart;
#else
  timeval time;
  gettimeofday(&time, NULL);
  return time.tv_sec * 1000.0 + time.tv_usec / 1000.0;
#endif
}

double timeIt(int reset) {
  static double startTime, endTime;
  static int timerWorking = 0;

  if (reset) {
    startTime = getTime();
    timerWorking = 1;
    return -1;
  } else {
    if (timerWorking) {
      endTime = getTime();
      timerWorking = 0;
      return (double) (endTime - startTime)/1000.0;
    } else {
      startTime = getTime();
      timerWorking = 1;
      return -1;
    }
  }
}

double _rand()
{
    static bool didSeeding = false;

#ifdef WIN32
    if (!didSeeding) {
        unsigned int seedNum = (unsigned int) time(NULL);
        srand(seedNum);
        didSeeding = true;
    }
    return rand()/( RAND_MAX + 1.0 );
#else
    if (!didSeeding) {
        unsigned int seedNum;
        struct timeval TV;
        unsigned int curTime;

        gettimeofday(&TV, NULL);
        curTime = (unsigned int) TV.tv_usec;
        seedNum = (unsigned int) time(NULL) + curTime + getpid() + getDevRandom();

        srand(seedNum);
        didSeeding = true;
    }
    return random()/( RAND_MAX + 1.0 );
#endif
}

int randomNumber(int min, int max)
{
    srand((unsigned)time(0));
    int range=(max-min)+1;
#ifdef WIN32
    int randNum = min + int((double)range*rand()/(RAND_MAX + 1.0));
#else
    int randNum = min + int((double)range*random()/(RAND_MAX + 1.0));
#endif
    return randNum;
}

int randomNumber( int limit )
{
    srand((unsigned)time(0));
#ifdef WIN32
    return static_cast<int>( static_cast<double>( limit ) * rand() / ( RAND_MAX + 1.0 ) );
#else
    return static_cast<int>( static_cast<double>( limit ) * random() / ( RAND_MAX + 1.0 ) );
#endif
}

double randomDouble( double limit )
{
#ifdef WIN32
    return static_cast<double>( limit ) * rand() / ( RAND_MAX + 1.0 );
#else
    return static_cast<double>( limit ) * random() / ( RAND_MAX + 1.0 );
#endif
}

vector<int> subSampleWithReplacement(const int numSamples) {
    vector<int> outRand(numSamples);
    vector<int>::iterator itr = outRand.begin(), end = outRand.end();

    for (; itr != end; itr++) {
        *itr = (int) ::floor(numSamples*_rand());
    }

    return unique(outRand);
}

vector<int> unique(const vector<int> inIndex) {
    vector<int> outIndex;
    set<int> tmpIndex;
    for (register int nSamp = 0; nSamp < (int) inIndex.size(); nSamp++) {
        tmpIndex.insert(inIndex[nSamp]);
    }

    set<int>::iterator itr = tmpIndex.begin(), end = tmpIndex.end();
    for (; itr != end; itr++) {
        outIndex.push_back(*itr);
    }

    return outIndex;
}

vector<int> randPerm(const int inNum) {
    vector<int> outRand(inNum);
    vector<std::pair<double, int> > tmpPairs;

    for (register int n = 0; n < inNum; n++) {
        tmpPairs.push_back(std::pair<double,int>(_rand(),n));
    }


    sort(tmpPairs.begin(), tmpPairs.end());

    vector<int>::iterator outIt = outRand.begin(), outEnd = outRand.end();
    vector<std::pair<double, int> >::const_iterator it(tmpPairs.begin());
    vector<std::pair<double, int> >::const_iterator end(tmpPairs.end());
    while( it != end ){
        *outIt = it->second;
        ++it;
        outIt++;
    }

    return outRand;
}

vector<int> randPerm(const int inNum, const int inPart) {
    vector<int> outRand(inNum);
    int randIndex, tempIndex;
    for (int nFeat = 0; nFeat < inNum; nFeat++) {
        outRand[nFeat] = nFeat;
    }
    for ( register int nFeat = 0; nFeat < inPart; nFeat++) {
        randIndex = (int) floor(((double) inNum - nFeat)*_rand()) + nFeat;
        if (randIndex == inNum) {
            randIndex--;
        }
        tempIndex = outRand[nFeat];
        outRand[nFeat] = outRand[randIndex];
        outRand[randIndex] = tempIndex;
    }

    outRand.erase(outRand.begin() + inPart, outRand.end());
    sort(outRand.begin(), outRand.end());


//    vector<int> outRand(inPart);
//    vector<std::pair<double, int> > tmpPairs;
//
//    for (register int n = 0; n < inNum; n++) {
//        tmpPairs.push_back(std::pair<double,int>(_rand(),n));
//    }
//
//    sort(tmpPairs.begin(), tmpPairs.end());
//
//    vector<int>::iterator outIt = outRand.begin(), outEnd = outRand.end();
//    vector<std::pair<double, int> >::const_iterator it(tmpPairs.begin());
//    vector<std::pair<double, int> >::const_iterator end(tmpPairs.end());
//    int n = 0;
//    while( n < inPart ){
//        *outIt = it->second;
//        ++it;
//        outIt++;
//        n++;
//    }

    return outRand;
}

vector<int> setDiff(const vector<int> inSet, const int numSamples) {
    vector<int> outSet(numSamples), allSamplesSet(numSamples);
    vector<int>::iterator end, itr = allSamplesSet.begin();

    for (int n = 0; itr != allSamplesSet.end(); itr++, n++) {
        *itr = n;
    }

    end = set_difference(allSamplesSet.begin(), allSamplesSet.end(), inSet.begin(), inSet.end(), outSet.begin());

    outSet.resize(int(end - outSet.begin()));
    return outSet;
}

void subSampleWithoutReplacement(const int numSamples, const int numInBagSamples,
                                 vector<int> &inBagSamples, vector<int> &outOfBagSamples)
{
    vector<int> randPermNumbers = randPerm(numSamples);

    inBagSamples.insert(inBagSamples.begin(),randPermNumbers.begin(),randPermNumbers.begin() + numInBagSamples);
    outOfBagSamples.insert(outOfBagSamples.begin(),randPermNumbers.begin() + numInBagSamples ,randPermNumbers.end());
}

void dispVector(const vector<int>& inVector) {
    vector<int>::const_iterator itr(inVector.begin()), end(inVector.end());
    cout << "Vector [" << inVector.size() << "] (";
    for (; itr != end; itr++) {
        cout << *itr << " ";
    }
    cout << "\b)" << endl;
}

void dispVector(const vector<float>& inVector) {
    vector<float>::const_iterator itr(inVector.begin()), end(inVector.end());
    cout << "Vector [" << inVector.size() << "] (";
    for (; itr != end; itr++) {
        cout << *itr << " ";
    }
    cout << "\b)" << endl;
}

void dispVector(const vector<double>& inVector) {
    vector<double>::const_iterator itr(inVector.begin()), end(inVector.end());
    cout << "Vector [" << inVector.size() << "] (";
    for (; itr != end; itr++) {
        cout << *itr << " ";
    }
    cout << "\b)" << endl;
}


void dispMatrix(const ublas::matrix<float>& inMat) {
    cout << "Matrix [" << inMat.size1() << ", " << inMat.size2() << "]" << endl << "(";
    for (int n = 0; n < (int) inMat.size1(); n++) {
        for (int m = 0; m < (int) inMat.size2(); m++) {
            cout << inMat(n,m) << " ";
        }
        cout << endl;
    }
    cout << "\b)" << endl;
}

void dispMatrix(const ublas::matrix<float>& inMat, const int rowIndex) {
   cout << "Matrix [" << inMat.size1() << ", " << inMat.size2() << "]" << endl << "(";
   for (int m = 0; m < (int) inMat.size2(); m++) {
     cout << inMat(rowIndex,m) << " ";
   }
   cout << endl << "\b)" << endl;
}

std::string readStringProp( const xmlNodePtr node, const std::string& propName,
			    const std::string& defaultValue )
{
	if ( !node )
    return defaultValue;

  xmlChar* tmp = xmlGetProp( node, reinterpret_cast<const xmlChar*>( propName.c_str() ) );
  if ( !tmp )
    return defaultValue;

  std::string ret( reinterpret_cast<const char*>( tmp ) );

  xmlFree( tmp );
  return ret;
}

int readIntProp( const xmlNodePtr node, const std::string& propName,
		 int defaultValue )
{
  if ( !node )
    return defaultValue;

  xmlChar* tmp = xmlGetProp( node, reinterpret_cast<const xmlChar*>( propName.c_str() ) );
  if ( !tmp )
    return defaultValue;

  int value = atoi( reinterpret_cast<const char*>( tmp ) );

  xmlFree( tmp );
  return value;
}

double readDoubleProp( const xmlNodePtr node, const std::string& propName,
		       double defaultValue )
{
  if ( !node )
    return defaultValue;

  xmlChar* tmp = xmlGetProp( node, reinterpret_cast<const xmlChar*>( propName.c_str() ) );
  if ( !tmp )
    return defaultValue;

  double value = atof( reinterpret_cast<const char*>( tmp ) );

  xmlFree( tmp );
  return value;
}

std::string readString( const xmlDocPtr doc, const xmlNodePtr node )
{
  if ( !doc || !node )
    return std::string();

  xmlChar* tmp = xmlNodeListGetString( doc, node->xmlChildrenNode, 0 );
  if ( !tmp )
    return std::string();

  std::string ret( reinterpret_cast<const char*>( tmp ) );

  xmlFree( tmp );
  return ret;
}

//ImageType readImageTypeProp( const xmlNodePtr node )
//{
//  ImageType ret = ORIGINAL;
//  if ( !node )
//    return ret;
//
//  xmlChar* tmp = xmlGetProp( node, reinterpret_cast<const xmlChar*>( "imageType" ) );
//  if ( !tmp )
//    return ret;
//
//  if ( xmlStrcmp( tmp, reinterpret_cast<const xmlChar*>( "original" ) ) == 0 )
//    ret = ORIGINAL;
//  else if ( xmlStrcmp( tmp, reinterpret_cast<const xmlChar*>( "delta" ) ) == 0 )
//    ret = DELTA;
//  else if ( xmlStrcmp( tmp, reinterpret_cast<const xmlChar*>( "up" ) ) == 0 )
//    ret = UP;
//  else if ( xmlStrcmp( tmp, reinterpret_cast<const xmlChar*>( "down" ) ) == 0 )
//    ret = DOWN;
//  else if ( xmlStrcmp( tmp, reinterpret_cast<const xmlChar*>( "left" ) ) == 0 )
//    ret = LEFT;
//  else if ( xmlStrcmp( tmp, reinterpret_cast<const xmlChar*>( "right" ) ) == 0 )
//    ret = RIGHT;
//
//  xmlFree( tmp );
//  return ret;
//}

//Rect readRect( const xmlNodePtr rectNode )
//{
//  int x = readIntProp( rectNode, "x", 0 );
//  int y = readIntProp( rectNode, "y", 0 );
//  int width = readIntProp( rectNode, "width", 0 );
//  int height = readIntProp( rectNode, "height", 0 );
//  return Rect( x, y, width, height );
//}

void addIntProp( xmlNodePtr node, const char* propName, int value )
{
  char buffer[16];
  snprintf( buffer, 16, "%d", value );
  xmlNewProp( node, reinterpret_cast<const xmlChar*>( propName ),
	      reinterpret_cast<const xmlChar*>( buffer ) );
}

void addDoubleProp( xmlNodePtr node, const char* propName, double value )
{
  char buffer[32];
  snprintf( buffer, 32, "%.12e", value );
  xmlNewProp( node, reinterpret_cast<const xmlChar*>( propName ),
	      reinterpret_cast<const xmlChar*>( buffer ) );
}


//void addImageTypeProp( xmlNodePtr node, ImageType imageType )
//{
//  switch( imageType ) {
//  case ORIGINAL:
//    xmlNewProp( node, reinterpret_cast<const xmlChar*>( "imageType" ),
//		reinterpret_cast<const xmlChar*>( "original" ) );
//    break;
//  case DELTA:
//    xmlNewProp( node, reinterpret_cast<const xmlChar*>( "imageType" ),
//		reinterpret_cast<const xmlChar*>( "delta" ) );
//    break;
//  case UP:
//    xmlNewProp( node, reinterpret_cast<const xmlChar*>( "imageType" ),
//		reinterpret_cast<const xmlChar*>( "up" ) );
//    break;
//  case DOWN:
//    xmlNewProp( node, reinterpret_cast<const xmlChar*>( "imageType" ),
//		reinterpret_cast<const xmlChar*>( "down" ) );
//    break;
//  case RIGHT:
//    xmlNewProp( node, reinterpret_cast<const xmlChar*>( "imageType" ),
//		reinterpret_cast<const xmlChar*>( "right" ) );
//    break;
//  case LEFT:
//    xmlNewProp( node, reinterpret_cast<const xmlChar*>( "imageType" ),
//		reinterpret_cast<const xmlChar*>( "left" ) );
//    break;
//  default:
//    break;
//  }
//}
//
//xmlNodePtr saveRect( const std::string& name, const Rect& rect )
//{
//  const xmlChar* tmpName = reinterpret_cast<const xmlChar*>( name.c_str() );
//  xmlNodePtr node = xmlNewNode( NULL, tmpName );
//  addIntProp( node, "x", rect.x() );
//  addIntProp( node, "y", rect.y() );
//  addIntProp( node, "width", rect.width() );
//  addIntProp( node, "height", rect.height() );
//  return node;
//}
//
//xmlNodePtr saveBin( const std::string& name, const short pos )
//{
//  const xmlChar* tmpName = reinterpret_cast<const xmlChar*>( name.c_str() );
//  xmlNodePtr node = xmlNewNode( NULL, tmpName );
//  addIntProp( node, "pos", pos );
//  return node;
//}
//
//xmlNodePtr saveHistBin( const std::string& name, const short pos, const double value )
//{
//  const xmlChar* tmpName = reinterpret_cast<const xmlChar*>( name.c_str() );
//  xmlNodePtr node = xmlNewNode( NULL, tmpName );
//  addIntProp( node, "pos", pos );
//  addDoubleProp( node, "value", value);
//  return node;
//}

std::string intToString(int num)
{
  char buffer[21];
  snprintf(buffer,20,"%d",num);
  return std::string(buffer);
}

std::string doubleToString(double num)
{
  char buffer[21];
  snprintf(buffer,20,"%.10f",num);
  return std::string(buffer);
}

void fillWithRandomNumbers(std::vector<float>& tmpWeights)
{

  std::vector<float>::iterator it(tmpWeights.begin()), end(tmpWeights.end());
  for(; it != end;it++){
     *it = static_cast<float> (2.0*(_rand() - 0.5));
  }
}
