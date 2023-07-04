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
#include "rateinvar.h"

RateInvar::RateInvar(double p_invar_sites, PhyloTree *tree)
 : RateHeterogeneity()
{
	if (tree)
		p_invar = max(tree->aln->frac_const_sites, MIN_PINVAR);
	else
		p_invar = MIN_PINVAR;
	fix_p_invar = false;
	phylo_tree = tree;
	name = "+I";
	full_name = "Invar";
	if (p_invar_sites > 0) {
		p_invar = p_invar_sites;
		fix_p_invar = true;
	}
}

string RateInvar::getNameParams() {
	ostringstream str;
	str << "+I{" << p_invar << '}';
	return str.str();
}

double RateInvar::computeFunction(double p_invar_value) {
	p_invar = p_invar_value;
	// fix bug: computeTip... will update ptn_invar vector
	phylo_tree->computePtnInvar();
	return -phylo_tree->computeLikelihood();
}

double RateInvar::targetFunk(double x[]) {
	getVariables(x);
	// fix bug: computeTip... will update ptn_invar vector
	phylo_tree->computePtnInvar();
	return -phylo_tree->computeLikelihood();
}

void RateInvar::setBounds(double *lower_bound, double *upper_bound, bool *bound_check) {
	if (getNDim() == 0) return;
	lower_bound[1] = MIN_PINVAR;
	upper_bound[1] = phylo_tree->aln->frac_const_sites;
	bound_check[1] = false;
}

double RateInvar::optimizeParameters(double epsilon) {
	if (fix_p_invar)
		return -computeFunction(p_invar);
	if (verbose_mode >= VB_MAX)
		cout << "Optimizing proportion of invariable sites..." << endl;
	double negative_lh;
	double ferror;
	p_invar = minimizeOneDimen(MIN_PINVAR, p_invar, phylo_tree->aln->frac_const_sites, max(epsilon, TOL_PINVAR), &negative_lh, &ferror);
	//p_invar = minimizeOneDimen(MIN_PINVAR, p_invar, 1.0 - MIN_PINVAR, TOL_PINVAR, &negative_lh, &ferror);
	phylo_tree->computePtnInvar();
	return -negative_lh;
}

void RateInvar::writeInfo(ostream &out) {
	out << "Proportion of invariable sites: " << p_invar << endl;
}

void RateInvar::writeParameters(ostream &out) {
	out << "\t" << p_invar;
}

void RateInvar::setVariables(double *variables) {
	if (RateInvar::getNDim() == 0) return;
	variables[1] = p_invar;
}

void RateInvar::getVariables(double *variables) {
	if (RateInvar::getNDim() == 0) return;
	p_invar = variables[1];
}
