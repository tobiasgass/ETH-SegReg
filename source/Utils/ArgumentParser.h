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
    ~ArgumentParser(){delete m_desc; delete m_optional;}
    
    template<typename T>
      void parameter(const char * opt, T& variable, const char * descr, bool required,bool optional=false){
      std::stringstream ss;
      po::options_description * optionsObjectPointer;
      if (!optional){
	optionsObjectPointer=this->m_desc;
      }else{
	optionsObjectPointer=this->m_optional;
      }
      if (required){
            ss<<descr<<" (REQUIRED)";
            optionsObjectPointer->add_options() 
                (opt,po::value<T>(&variable)->required() ,ss.str().c_str()) ;
        }
        else{
	  ss<<descr<<" (default :"<<variable<<")";
	  optionsObjectPointer->add_options() 
	    (opt,po::value<T>(&variable) ,ss.str().c_str()) ;
        }

    }

    void option(const char * opt, bool & variable, const char * descr, bool optional=false);

    bool parse();
    void help();

private:
    po::options_description * m_desc, * m_optional; 
    std::string m_brief;
    int m_argc;
    char ** m_argv;

};


//explicit instantiations
template void ArgumentParser::parameter<float>(const char * , float& , const char * , bool,bool);
template void ArgumentParser::parameter<double>(const char * , double& , const char * , bool,bool);
template void ArgumentParser::parameter<int>(const char * , int& , const char * , bool,bool);
template void ArgumentParser::parameter<std::string>(const char * , std::string& , const char * , bool,bool);
template void ArgumentParser::parameter<unsigned char>(const char * , unsigned char& , const char * , bool,bool);
template void ArgumentParser::parameter<short int>(const char * , short int& , const char * , bool,bool);
template void ArgumentParser::parameter<unsigned int>(const char * , unsigned int& , const char * , bool,bool);
template void ArgumentParser::parameter<unsigned short int>(const char * , unsigned short int& , const char * , bool,bool);
