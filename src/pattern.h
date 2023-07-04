//
// C++ Interface: pattern
//
// Description: 
//
//
// Author: BUI Quang Minh, Steffen Klaere, Arndt von Haeseler <minh.bui@univie.ac.at>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef PATTERN_H
#define PATTERN_H

#include <iostream>
#include <string>

using namespace std;

/**
	Site-patterns in a multiple sequence alignment
	@author BUI Quang Minh, Steffen Klaere, Arndt von Haeseler <minh.bui@univie.ac.at>
*/
class Pattern : public string
{
public:
	/** 
		constructor
	*/
    Pattern();

	/** 
		determine if the pattern is constant. update the is_const variable.
	*/
	void computeConst(int STATE_UNKNOWN);

	/**
		@param num_states number of states of the model
		@return the number of ambiguous character incl. gaps 
	*/
	int computeAmbiguousChar(int num_states);

	/**
		@param num_states number of states of the model
		@return the number of gaps 
	*/
	int computeGapChar(int num_states, int STATE_UNKNOWN);

	/** 
		destructor
	*/
    virtual ~Pattern();

	/**
		frequency appearance of the pattern
	*/
	int frequency;

	/**
		true if this is a constant pattern
	*/
	bool is_const;

	int ras_pars_score; // Diep added: for sorting pattern by score
};

#endif
