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
#ifndef MODELDNA_H
#define MODELDNA_H

#include "modelgtr.h"
#include <string>

/**
 * return name and other information of an input model_name
 * @param model_name DNA model name
 * @param full_name (OUT) full model name with citation
 * @param rate_type (OUT) a 6-digit string showing 6 rate constraints
 * @param def_freq (OUT) base frequency type, either FREQ_UNNOWN, FREQ_EQUAL, or FREQ_ESIMATE
 * @return unique model name or "" (empty string) if model_name is unrecognized
 */
string getDNAModelInfo(string model_name, string &full_name, string &rate_type, StateFreqType &def_freq);

/**
All DNA models are managed here

	@author BUI Quang Minh <minh.bui@univie.ac.at>
*/
class ModelDNA : public ModelGTR
{
public:
	/**
		constructor
		@param tree associated tree for the model
	*/
    ModelDNA(PhyloTree *tree, bool count_rates = true);

	/**
		constructor
		@param model_name model name, e.g., JC, HKY.
		@param freq state frequency type
		@param tree associated phylogenetic tree
	*/
    ModelDNA(const char *model_name, string model_params, StateFreqType freq, string freq_params, PhyloTree *tree, bool count_rates = true);

	/**
		initialization, called automatically by the constructor, no need to call it
		@param model_name model name, e.g., JC, HKY.
		@param freq state frequency type
	*/
	virtual void init(const char *model_name, string model_params, StateFreqType freq, string freq_params);

	/**
	 * @return model name with parameters in form of e.g. GTR{a,b,c,d,e,f}
	 */
	virtual string getNameParams();

	/**
		Read the rate parameters from a comma-separated string
		It will throw error messages if failed
		@param in input stream
	*/
	virtual void readRates(string str);

	/**
		set the substitution rate parameters by a specification
		@param rate_spec a string of six letters describing how rates are related
		@return TRUE if successful, FALSE otherwise
	*/
	bool setRateType(const char *rate_spec);

	/**
		return the number of dimensions
	*/
	virtual int getNDim();

	/**
		write parameters, used with modeltest
		@param out output stream
	*/
	virtual void writeParameters(ostream &out);

protected:

	/**
		this function is served for the multi-dimension optimization. It should pack the model parameters 
		into a vector that is index from 1 (NOTE: not from 0)
		@param variables (OUT) vector of variables, indexed from 1
	*/
	virtual void setVariables(double *variables);

	/**
		this function is served for the multi-dimension optimization. It should assign the model parameters 
		from a vector of variables that is index from 1 (NOTE: not from 0)
		@param variables vector of variables, indexed from 1
	*/
	virtual void getVariables(double *variables);

	/**
		rate parameter specification, a string of 6 characters
	*/
	string param_spec;
	
	/** vector of boolean, TRUE if corresponding parameter is fixed and FALSE otherwise */
	vector<bool> param_fixed;

};

#endif
