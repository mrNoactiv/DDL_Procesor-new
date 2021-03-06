/**
*	\file  cUniformRandomGenerator.h
*	\author Jiri Dvorsky (1998-2003), Jiří Walder (2010-2011)
*	\version 2.1
*	\date dec 2011
*	\brief Generator nahodnych cisel s rovnomernym rozlozenim. Algoritmus L'Ecuyer, delka periody > 2.10^18
*          Zalozeno na kodu z knihy Numerical Recipes. 
*
*  Poznamky:
*    Poznamky:
*    cUniformRandomGenerator(const bool Randomize = true)
*    Randomize == true
*      Pri kazdem vytvoreni tridy se bude generovat jina sekvence cisel.
*      Pocatecni hodnota (seminko) je odvozeno od casu.
*    Randomize == false
*      Pokazde se generuje stejna posloupnost.
*
*    cUniformRandomGenerator(const int ARandSeed)
*      Pocatecni hodnotu (seminko) je mozno explicitne uvest.
*/

#ifndef __cUniformRandomGenerator_new_h__
#define __cUniformRandomGenerator_new_h__

#include <time.h>
#include "cAbstractRandomGenerator.h"

namespace common {
  namespace random_new {

class cUniformRandomGenerator: public cAbstractRandomGenerator
{
	static const int NTAB = 32;

public:
	cUniformRandomGenerator(const bool Randomize = true);	// false = pokazde stejna posl. cisel
	cUniformRandomGenerator(const int ARandSeed);
	virtual ~cUniformRandomGenerator();

	virtual double GetNext();	// vraci cisla z otevreneho intervalu (0;1)

protected:
	static const int IM1;
	static const int IM2;
	static const double AM;
	static const int IMM1;
	static const int IA1;
	static const int IA2;
	static const int IQ1;
	static const int IQ2;
	static const int IR1;
	static const int IR2;
	static const int NDIV;
	static const double EPS;
	static const double RNMX;

	int iv[NTAB];
	int idum;
	int idum2;
	int iy;
};  // cUniformRandomGenerator
}}
#endif  // __cUniformRandomGenerator_h__