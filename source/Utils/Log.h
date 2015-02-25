#pragma once

#include "boost/format.hpp"
//#include "boost/timer.hpp"
//#include "boost/timer/timer.hpp"
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <cstring>
#include <map>
#include <utility>
#include <stack>

#define logSetStage(stage) mylog.setStage(stage)
#define logResetStage mylog.resetStage()
#define logUpdateStage(stage) mylog.updateStage(stage)
#define logSetVerbosity(level) mylog.setVerbosity(level)
#define VAR(x)  #x " = " << x 


///wrapper to a wall clock timer.
class MyCPUTimer{
private:
    double m_starTime;
    struct timeval m_tim;  
public:
    MyCPUTimer();
    ///get elapsed time
    double elapsed();
};

///class to handle logging. supports varying degrees of verbosity at run-time
///supports direct logging to file
///also supports reporting on 'stage' to be set in the code, this facilitates tracking of highly verbose output.
class MyLog {
public:
    std::ostream * mOut;
private:
    MyCPUTimer m_timer;
    std::string m_stage;
    std::stack<std::string> m_stages;
   
    int m_verb;
    bool m_cachedOutput;
    int m_timerOffset;
public:
    MyLog();
    boost::format  getStatus();
    void setStage(std::string stage);
    void updateStage(std::string stage);
    void resetStage();
    void setVerbosity(int v);
    int getVerbosity();
    void setCachedLogging();
    void flushLog(std::string filename);
    void addTime(int t);
};



///GLOBAL log variable!
MyLog mylog;


///macros for using the log class. the log object is not used directly in the client code, but through these macros
#define LOGADDTIME(t) mylog.addTime(t)

#define LOG \
    if (mylog.getVerbosity()>=10)                                        \
        (*mylog.mOut) <<  " [" << __FILE__<<":"<<__LINE__<<":"<<__FUNCTION__<<"] "; \
    if (mylog.getVerbosity()<30 )                                 \
        (*mylog.mOut) << mylog.getStatus()<<" "

#define LOGV(level) \
    if (mylog.getVerbosity()>=level && mylog.getVerbosity()>=10) \
        (*mylog.mOut) <<  " [" << __FILE__<<":"<<__LINE__<<":"<<__FUNCTION__<<"] "; \
    if (mylog.getVerbosity()>=level)                                    \
        (*mylog.mOut)<<mylog.getStatus()<<" ["<<level<<"] "

#define LOGI(level, instruction) \
    if (mylog.getVerbosity()>=level)  \
        instruction

///more variable timer class
///can keep track of separate timers for a number of strings, easy to call with a repeated function call
class MyTimer{
private:
    //std::map<std::string, boost::timer::cpu_timer> m_timers;
    std::map<std::string, MyCPUTimer> m_timers;
    std::map<std::string, double> m_timings;
    std::map<std::string,int> m_calls;
    
public:
    void time();
    void start(std::string tag);
    void end(std::string tag);
    void print();
};

MyTimer timeLOG;
#define TIME(instruction) \
    timeLOG.start(#instruction);                  \
    instruction;                                \
    timeLOG.end(#instruction);
#define TIMEI(level,instruction)                  \
    if (mylog.getVerbosity()>=level){             \
        timeLOG.start(#instruction);              \
    instruction;                                  \
    timeLOG.end(#instruction); }

#define OUTPUTTIMER   timeLOG.print()

double tOpt=0;
double tUnary=0;
double tPairwise=0;

struct MatchPathSeparator
{
    bool operator()( char ch ) const
    {
        return ch == '/';
    }
};

std::string
basename( std::string const& pathname )
{
    return std::string( 
        std::find_if( pathname.rbegin(), pathname.rend(),
                      MatchPathSeparator() ).base(),
        pathname.end() );
}

std::vector<std::string> &split(std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}


void gen_random(char *s, const int len) {
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

    for (int i = 0; i < len; ++i) {
        s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }

    s[len] = 0;
}

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
