#include "ArgumentParser.h"

ArgumentParser::ArgumentParser(int argc, char** argv){
    m_argc=argc; m_argv=argv; m_brief="";   m_desc.add_options()("help,h","display help");
}

ArgumentParser::ArgumentParser(){m_argc=0; m_argv=NULL; m_brief="";   m_desc.add_options()("help,h","display help");}
    //ArgumentParser(std::string b){m_brief=b;  m_desc.add_options()("help,h","display help");}
    
void ArgumentParser::option(const char * opt, bool & variable, const char * descr){
    m_desc.add_options() 
        (opt,po::value<bool>(&variable) ,descr) ;
}

bool ArgumentParser::parse(){
    std::string appName = boost::filesystem::basename(m_argv[0]); 
    po::variables_map vm; 
    try 
        { 
            po::store(po::parse_command_line(m_argc, m_argv, m_desc),  
                      vm); // can throw 
             
            /** --help option 
             */ 
            if ( vm.count("help")  ) 
                { 
                    std::cout << m_brief << std::endl <<m_desc<<std::endl;;
                    //rad::OptionPrinter::printStandardAppDesc(appName, 
                    //                                       std::cout, 
                    //                                       m_desc); 
                    exit(0);
                } 
            
            po::notify(vm); // throws on error, so do after help in case 
            // there are any problems 
        } 
    catch(boost::program_options::required_option& e) 
        { 
            //rad::OptionPrinter::formatRequiredOptionError(e); 
            std::cerr << "ERROR: " << e.what() << std::endl ;
            std::cout<<m_desc<<std::endl; 
            //rad::OptionPrinter::printStandardAppDesc(appName, 
            //                                         std::cout, 
            //                                        m_desc                                               ); 
            exit(0);
        } 
    catch(po::error& e) 
        { 
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
            std::cout << m_desc << std::endl; 
            exit(0);
        } 
}
void ArgumentParser::help(){}
