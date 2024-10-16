/***************************************************************************
 *   Copyright (C) 2009 by BUI Quang Minh   *
 *   minh.bui@univie.ac.at   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#ifndef MODELPROTEIN_H
#define MODELPROTEIN_H

#include "modelgtr.h"

/**
Substitution models for protein sequences

	@author BUI Quang Minh <minh.bui@univie.ac.at>
*/
class ModelProtein : public ModelGTR
{
public:
	/**
		constructor
		@param model_name model name, e.g., JTT, WAG.
		@param freq state frequency type
		@param tree associated phylogenetic tree
	*/
    ModelProtein(const char *model_name, string model_params, StateFreqType freq, string freq_params, PhyloTree *tree, bool count_rates = true);

	/**
		initialization, called automatically by the constructor, no need to call it
		@param model_name model name, e.g., JTT, WAG.
		@param freq state frequency type
	*/
	virtual void init(const char *model_name, string model_params, StateFreqType freq, string freq_params);

	/**
		read the rates from an input stream. it will throw error messages if failed
		@param in input stream
	*/
	virtual void readRates(istream &in);

	/**
	 * @return model name with parameters in form of e.g. GTR{a,b,c,d,e,f}
	 */
	virtual string getNameParams() { return name; }


};

#endif
