#include "ArgumentParser.h"

static const unsigned LINEWIDTH=160;

ArgumentParser::ArgumentParser(int argc, char** argv){
  m_argc=argc;
  m_argv=argv;
  m_brief="";
  m_desc=new po::options_description("General options",LINEWIDTH,LINEWIDTH/2);
  m_optional=new po::options_description("Experimental and legacy options",LINEWIDTH,LINEWIDTH/2);
  m_desc->add_options()("help,h","display brief help");
  m_desc->add_options()("experimental","display full help, including experimental and deprecated options");
  
}


ArgumentParser::ArgumentParser(){
  m_argc=0; m_argv=NULL;
  m_brief="";
  m_desc=new po::options_description("General options",LINEWIDTH,LINEWIDTH/2);
  m_optional=new po::options_description("Experimental and legacy options",LINEWIDTH,LINEWIDTH/2);
  m_desc->add_options()("help,h","display brief help");
  m_desc->add_options()("experimental","display full help, including experimental and deprecated options");
}    
void ArgumentParser::option(const char * opt, bool & variable, const char * descr, bool optional){
  if (!optional){
    m_desc->add_options() 
      (opt,po::bool_switch(&variable) ,descr) ;
  }else{
    m_optional->add_options() 
      (opt,po::bool_switch(&variable) ,descr) ;
  }
}

bool ArgumentParser::parse(){
  std::string appName = boost::filesystem::basename(m_argv[0]); 
  po::variables_map vm;
  po::options_description all("Allowed options",LINEWIDTH);
  all.add(*m_desc);
  all.add(*m_optional);
    
  try 
    { 
      po::store(po::parse_command_line(m_argc, m_argv, all),  
		vm); // can throw 
             
      /** --help option 
       */ 
      if ( vm.count("help")  ) 
	{ 
	  std::cout << m_brief << std::endl <<*m_desc<<std::endl;;
	  exit(0);
	}
      if ( vm.count("experimental")  ) 
	{ 
	  std::cout << m_brief << std::endl <<all<<std::endl;;
                  
	  exit(0);
	} 
            
      po::notify(vm); // throws on error, so do after help in case 
      // there are any problems 
    } 
  catch(boost::program_options::required_option& e) 
    { 
      //rad::OptionPrinter::formatRequiredOptionError(e); 
      std::cerr << "ERROR: " << e.what() << std::endl ;
      std::cout<<*m_desc<<std::endl; 
      //rad::OptionPrinter::printStandardAppDesc(appName, 
      //                                         std::cout, 
      //                                        m_desc                                               ); 
      exit(0);
    } 
  catch(po::error& e) 
    { 
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
      std::cout <<* m_desc << std::endl; 
      exit(0);
    }
  return 1;
}
void ArgumentParser::help(){}
