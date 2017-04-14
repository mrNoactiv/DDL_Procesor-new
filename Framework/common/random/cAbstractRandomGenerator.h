/**************************************************************************}
{                                                                          }
{    cAbstractRandomGenerator.h                           		      		}
{                                                                          }
{                                                                          }
{                 Copyright (c) 1998, 2003					Jiri Dvorsky		}
{                                                                          }
{    VERSION: 2.0														DATE 27/2/2003    }
{                                                                          }
{             following functionality:													}
{    Abstraktni generator nahodnych cisel.                                 }
{    Metoda GetNext vraci nahodna cisla z otevreneho intervalu (0;1)       }
{                                                                          }
{    UPDATE HISTORY:                                                       }
{                                                                          }
{                                                                          }
{**************************************************************************/
#ifndef __cAbstractRandomGenerator_h__
#define __cAbstractRandomGenerator_h__

namespace common {
  namespace random {

class cAbstractRandomGenerator
{
public:
	cAbstractRandomGenerator();
	virtual ~cAbstractRandomGenerator();

	virtual double GetNext() = 0;
};  // cAbstractRandomGenerator


inline cAbstractRandomGenerator::cAbstractRandomGenerator()
{
}


inline cAbstractRandomGenerator::~cAbstractRandomGenerator()
{
}
}}
#endif  // __cAbstractRandomGenerator_h__
