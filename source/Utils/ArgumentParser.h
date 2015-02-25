#pragma once

#include <boost/program_options.hpp>
#include <string>
#include "boost/filesystem.hpp" 
#include <sstream>
#include <iostream>


namespace po = boost::program_options; 

class ArgumentParser{

public:
    ArgumentParser(int argc, char** argv);
    ArgumentParser();
    
    template<typename T>
    void parameter(const char * opt, T& variable, const char * descr, bool required){
        if (required){
            std::stringstream ss;
            ss<<descr<<" (REQUIRED)";
            m_desc.add_options() 
                (opt,po::value<T>(&variable)->required() ,ss.str().c_str()) ;
        }
        else{
            std::stringstream ss;
            ss<<descr<<" (default :"<<variable<<")";
            m_desc.add_options() 
                (opt,po::value<T>(&variable) ,ss.str().c_str()) ;
        }

    }

    void option(const char * opt, bool & variable, const char * descr);

    bool parse();
    void help();

private:
    po::options_description m_desc; 
    std::string m_brief;
    int m_argc;
    char ** m_argv;

};


//explicit instantiations
template void ArgumentParser::parameter<float>(const char * , float& , const char * , bool);
template void ArgumentParser::parameter<double>(const char * , double& , const char * , bool);
template void ArgumentParser::parameter<int>(const char * , int& , const char * , bool);
template void ArgumentParser::parameter<std::string>(const char * , std::string& , const char * , bool);
template void ArgumentParser::parameter<unsigned char>(const char * , unsigned char& , const char * , bool);
template void ArgumentParser::parameter<short int>(const char * , short int& , const char * , bool);
template void ArgumentParser::parameter<unsigned int>(const char * , unsigned int& , const char * , bool);
template void ArgumentParser::parameter<unsigned short int>(const char * , unsigned short int& , const char * , bool);
