#pragma once

#include "boost/format.hpp"
#include "boost/timer.hpp"
#define logSetStage(stage) mylog.setStage(stage)
#define logResetStage mylog.resetStage()
class MyLog {

   boost::timer m_timer;
    std::string m_stage,m_oldStage;
    
public:
    boost::format  getStatus(){
        unsigned int elapsed = m_timer.elapsed();
        boost::format logLine("%2d:%02d [%15s] - ");
        logLine % (elapsed / 60);
        logLine % (elapsed % 60);
        logLine % m_stage;
        return logLine;
    }
    void setStage(std::string stage) {
        m_oldStage=m_stage;
        m_stage = stage;
    }
    void resetStage(){
        m_stage=m_oldStage;
    }
};
MyLog mylog;

#define LOG std::cout << mylog.getStatus()<<" "

