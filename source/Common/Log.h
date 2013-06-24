#pragma once

#include "boost/format.hpp"
#include "boost/timer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <cstring>
#include <map>
#include <utility>
#include <stack>
using namespace std;

#define logSetStage(stage) mylog.setStage(stage)
#define logResetStage mylog.resetStage()
#define logSetVerbosity(level) mylog.setVerbosity(level)
#define VAR(x)  #x " = " << x 


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
        m_stage="";
        m_verb=0;
        mOut=&std::cout;
        m_cachedOutput=false;
    }
    boost::format  getStatus(){
        unsigned int elapsed = m_timer.elapsed();
        boost::format logLine("%2d:%02d [%-50s] - ");
        logLine % (elapsed / 60);
        logLine % (elapsed % 60);
        logLine % m_stage;
        
        return logLine;
    }
    void setStage(std::string stage) {
        m_stages.push(m_stage);
        m_stage = m_stage + stage+":";
    }
    void resetStage(){
        //std::cout<<"old stage : "<<VAR(m_stage)<<endl;
        if (!m_stages.empty()){
            m_stage=m_stages.top();
            m_stages.pop();
        }
        //std::cout<<"new stage : "<<VAR(m_stage)<<endl;
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

class MyTimer{
private:
    std::map<std::string, boost::timer> m_timers;
    std::map<std::string, double> m_timings;
    std::map<std::string,int> m_calls;
    
public:
    void time(){};
    void start(std::string tag){
        m_timers.insert(std::pair<std::string,boost::timer>(tag,boost::timer()));
        std::map<std::string,double>::iterator it = m_timings.find(tag);
        if (it == m_timings.end()){
            m_timings.insert(std::pair<std::string,double>(tag,0.0));
            m_calls.insert(std::pair<std::string,int>(tag,0));
            
        }

    }
    void end(std::string tag){
        m_timings[tag]+=m_timers[tag].elapsed();
        m_calls[tag]+=1;
        m_timers.erase(tag);
    }
    void print(){
        for (std::map<std::string, double>::iterator it = m_timings.begin(); it != m_timings.end(); ++it){
            if (m_calls.find(it->first) != m_calls.end())
                LOG<<m_calls[it->first]<<" calls to "<<it->first<<", total run time "<<it->second<<" seconds."<<std::endl;
        }
    }
};

MyTimer timeLOG;
#define TIME(instruction) \
    timeLOG.start(#instruction);                  \
    instruction;                                \
    timeLOG.end(#instruction);

#define OUTPUTTIMER   timeLOG.print()

double tOpt=0;
double tUnary=0;
double tPairwise=0;
