#pragma once

#include "boost/format.hpp"
#include "boost/timer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stack>
#define logSetStage(stage) mylog.setStage(stage)
#define logResetStage mylog.resetStage()
#define logSetVerbosity(level) mylog.setVerbosity(level)

using namespace std;
class MyLog {
public:
    ostream * mOut;
private:
    boost::timer m_timer;
    std::string m_stage;
    std::stack<std::string> m_stages;
   
    int m_verb;
    bool m_cachedOutput;
public:
    MyLog(){
        m_stages.push("");
        m_verb=0;
        mOut=&std::cout;
        m_cachedOutput=false;
    }
    boost::format  getStatus(){
        unsigned int elapsed = m_timer.elapsed();
        boost::format logLine("%2d:%02d [%-50s] - ");
        logLine % (elapsed / 60);
        logLine % (elapsed % 60);
        if (m_stages.size()){
            logLine % m_stages.top();
        }
        return logLine;
    }
    void setStage(std::string stage) {
        
        m_stage = m_stage + stage+":";
        m_stages.push( m_stage);
    }
    void resetStage(){
        m_stages.pop();
        m_stage=m_stages.top();
    }
    void setVerbosity(int v){
        m_verb=v;
    }
    int getVerbosity(){
        return m_verb;
    }
    void setCachedLogging(){
        std::ostringstream * oss=new std::ostringstream;
        mOut = (ostream *) oss;
        m_cachedOutput=true;
    }
    void flushLog(string filename){
        if (!m_cachedOutput){
            std::cerr<<"output not cached but attempting to read stringstream.. aborting"<<std::endl;
        }else{
            ofstream  ofs(filename.c_str());
            ostringstream * oss =  (ostringstream *)mOut;
            ofs <<oss->str();
        }
    }
};
MyLog mylog;
#define VAR(x)  #x " = " << x 

#if 1

#define LOG \
    if (mylog.getVerbosity()>=30)                                        \
        (*mylog.mOut) <<  " [" << __FILE__<<":"<<__LINE__<<":"<<__FUNCTION__<<"] "; \
    if (mylog.getVerbosity()>=0 )                                       \
    (*mylog.mOut) << mylog.getStatus()<<" "

#define LOGV(level) \
    if (mylog.getVerbosity()>=level && mylog.getVerbosity()>=30) \
        (*mylog.mOut) <<  " [" << __FILE__<<":"<<__LINE__<<":"<<__FUNCTION__<<"] "; \
    if (mylog.getVerbosity()>=level)                                    \
        (*mylog.mOut)<<mylog.getStatus()<<" ["<<level<<"] "

#define LOGI(level, instruction) \
    if (mylog.getVerbosity()>=level)  \
        instruction

#else
#define LOG \
    if (mylog.getVerbosity()>=30)                                        \
        std::cout <<  " [" << __FILE__<<":"<<__LINE__<<":"<<__FUNCTION__<<"] "; \
    if (mylog.getVerbosity()>=0 )                                        \
        std::cout << mylog.getStatus()<<" "

#define LOGV(level) \
    if (mylog.getVerbosity()>=level && mylog.getVerbosity()>=30) \
        std::cout <<  " [" << __FILE__<<":"<<__LINE__<<":"<<__FUNCTION__<<"] "; \
    if (mylog.getVerbosity()>=level)                                    \
        std::cout<<mylog.getStatus()<<" ["<<level<<"] "

#define LOGI(level, instruction) \
    if (mylog.getVerbosity()>=level)  \
        instruction


#endif

double tOpt=0;
double tUnary=0;
double tPairwise=0;
