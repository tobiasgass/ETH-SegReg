//---------------------------------------------------------------------------

#include <stdio.h>
#include "Fast_PD.h"

typedef CV_Fast_PD::Real Real;

/*
 *
 * Usage: FastPD.exe input_file output_file
 *
 */
int main( int argc, char **argv )
{
	FILE *fp = fopen( argv[1], "rb" );
	if (!fp)
	{
		printf( "Error: I cannot open input file\n" );
		exit(1);
	}
	assert( fp );

	// Read input
	//
	printf( "Reading input data..." );

	int _numpoints; 
	int _numlabels; 
	int _numpairs ;
	int _max_iters;
	fread( &_numpoints, sizeof(int), 1, fp );
	fread( &_numpairs , sizeof(int), 1, fp );
	fread( &_numlabels, sizeof(int), 1, fp );
	fread( &_max_iters, sizeof(int), 1, fp );

	Real  *_lcosts = new Real[_numpoints*_numlabels];
	int   *_pairs  = new int [_numpairs*2];
	Real  *_dist   = new Real[_numlabels*_numlabels];
	Real  *_wcosts = new Real[_numpairs];
	fread( _lcosts, sizeof(Real), _numpoints*_numlabels, fp );
	fread( _pairs , sizeof(int ), _numpairs*2          , fp );
	fread( _dist  , sizeof(Real), _numlabels*_numlabels, fp );
	fread( _wcosts, sizeof(Real), _numpairs            , fp );
	
	printf( "Done\n" );

	printf( "#MRF-nodes = %d, #labels = %d, #MRF-edges = %d, #max-iterations = %d\n", 
	         _numpoints, _numlabels, _numpairs, _max_iters );	

	fclose(fp);

	// Run the algorith
	//
	CV_Fast_PD pd( _numpoints, _numlabels, _lcosts,
	               _numpairs, _pairs, _dist, _max_iters,
				   _wcosts );
	pd.run();


	// Save computed labels to output file
	//
	printf( "Writing labels to output file..." );
	FILE *outf = fopen( argv[2], "wb" );
	assert( outf );

	for( int i = 0; i < _numpoints; i++ )
	 	fwrite( &pd._pinfo[i].label, sizeof(Graph::Label), 1, outf );  

	fclose(outf);
	printf( "Done\n" );
}

