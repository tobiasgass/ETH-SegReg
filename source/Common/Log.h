#pragma once




#include "boost/format.hpp"
#include "boost/timer.hpp"

#define log mylog << boost::format
#define logSetStage(stage) mylog.setStage(stage)

class MyLog {

   boost::timer m_timer;
   std::string m_stage;

public:

    template <class T> void operator<<(T t) {
        
        unsigned int elapsed = m_timer.elapsed();
        
        boost::format logLine("%2d:%02d [%15s] - %s\n");
        logLine % (elapsed / 60);
        logLine % (elapsed % 60);
        logLine % m_stage;
        logLine % t;
        
        std::clog << logLine;
    }
    
    void setStage(std::string stage) {
        m_stage = stage;
    }

};
MyLog LOG;

