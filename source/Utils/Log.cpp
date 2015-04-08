#include "Log.h"
#include <iostream>
#include <sstream>



MyCPUTimer::MyCPUTimer(){
    //gettimeofday(&m_tim, NULL);  
    //m_starTime=m_tim.tv_sec;//+(m_tim.tv_usec/1000000.0);  
}
double MyCPUTimer::elapsed(){
    //gettimeofday(&m_tim, NULL);  
    //return (m_tim.tv_sec)-m_starTime;//+(m_tim.tv_usec/1000000.0));  
  return 1;
}



MyLog::MyLog(){
    m_stage="";
    m_verb=0;
    mOut=&std::cout;
    m_cachedOutput=false;
    m_timerOffset=0;
        
}
boost::format  MyLog::getStatus(){
    //boost::timer::cpu_times elapsedT=m_timer.elapsed();
    unsigned int elapsed = m_timer.elapsed()+m_timerOffset;
    //unsigned int elapsed = elapsedT.wall+m_timerOffset;
    boost::format logLine("%4d:%02d [%-50s] - ");
    logLine % (elapsed / 60);
    logLine % (elapsed % 60);
    logLine % m_stage;
        
    return logLine;
}
void MyLog::setStage(std::string stage) {
    m_stages.push(m_stage);
    m_stage = m_stage + stage+":";
}
void MyLog::updateStage(std::string stage){
    resetStage();
    setStage(stage);
}
void MyLog::resetStage(){
    if (!m_stages.empty()){
        m_stage=m_stages.top();
        m_stages.pop();
    }
}
void MyLog::setVerbosity(int v){
    m_verb=v;
}
int MyLog::getVerbosity(){
    return m_verb;
}
void MyLog::setCachedLogging(){
    std::ostringstream * oss=new std::ostringstream;
    mOut = (std::ostream *) oss;
    m_cachedOutput=true;
}

void MyLog::flushLog(std::string filename){
    if (!m_cachedOutput){
        std::cerr<<"output not cached but attempting to read stringstream.. aborting"<<std::endl;
    }else{
        std::ofstream  ofs(filename.c_str());
        std::ostringstream * oss =  (std::ostringstream *)mOut;
        ofs <<oss->str();
    }
}
void MyLog::addTime(int t){}//m_timerOffset+=t;}





void MyTimer::time(){};
void MyTimer::start(std::string tag){
    //m_timers.insert(std::pair<std::string,boost::timer::cpu_timer > (tag,boost::timer::cpu_timer()));
    m_timers.insert(std::pair<std::string,MyCPUTimer> (tag,MyCPUTimer()));
    std::map<std::string,double>::iterator it = m_timings.find(tag);
    if (it == m_timings.end()){
        m_timings.insert(std::pair<std::string,double>(tag,0.0));
        m_calls.insert(std::pair<std::string,int>(tag,0));
            
    }

}
void MyTimer::end(std::string tag){
    m_timings[tag]+=m_timers[tag].elapsed();
    m_calls[tag]+=1;
    m_timers.erase(tag);
}
void MyTimer::print(){
    for (std::map<std::string, double>::iterator it = m_timings.begin(); it != m_timings.end(); ++it){
        if (m_calls.find(it->first) != m_calls.end())
            LOG<<m_calls[it->first]<<" calls to "<<it->first<<", total run time "<<it->second<<" seconds."<<std::endl;
    }
}




MyLog mylog;
double tOpt=0;     
double tUnary=0;   
double tPairwise=0;
MyTimer timeLOG;

bool MatchPathSeparator::operator()( char ch ) const
    {
        return ch == '/';
    }


std::string basename( std::string const& pathname )
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
