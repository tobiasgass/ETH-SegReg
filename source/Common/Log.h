#pragma once

#include "boost/format.hpp"
#include "boost/timer.hpp"

#include <stack>
#define logSetStage(stage) mylog.setStage(stage)
#define logResetStage mylog.resetStage()
#define logSetVerbosity(level) mylog.setVerbosity(level)
class MyLog {

    boost::timer m_timer;
    std::string m_stage,m_oldStage;
    std::stack<std::string> m_stages;
    int m_verb;
public:
    MyLog(){
        m_stages.push("void");
        m_verb=0;
    }
    boost::format  getStatus(){
        unsigned int elapsed = m_timer.elapsed();
        boost::format logLine("%2d:%02d [%25s] - ");
        logLine % (elapsed / 60);
        logLine % (elapsed % 60);
        if (m_stages.size()){
            logLine % m_stages.top();
        }
        return logLine;
    }
    void setStage(std::string stage) {
        m_oldStage=m_stage;
        m_stage = stage;
        m_stages.push(stage);
    }
    void resetStage(){
        m_stage=m_oldStage;
        m_stages.pop();
    }
    void setVerbosity(int v){
        m_verb=v;
    }
    int getVerbosity(){
        return m_verb;
    }

};
MyLog mylog;



#define LOG \
    if (mylog.getVerbosity()>=10)                                        \
        std::cout <<  " [" << __FILE__<<":"<<__LINE__<<":"<<__FUNCTION__<<"] "; \
    std::cout << mylog.getStatus()<<" "

#define LOGV(level) \
    if (mylog.getVerbosity()>=10) \
        std::cout <<  " [" << __FILE__<<":"<<__LINE__<<":"<<__FUNCTION__<<"] "; \
    if (mylog.getVerbosity()>=level)                                    \
        std::cout<<mylog.getStatus()<<" ["<<level<<"] "

