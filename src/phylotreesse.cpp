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
#include <type_traits>

#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>

#include "phylotree.h"
#include "model/modelgtr.h"

namespace hn = hwy::HWY_NAMESPACE;

/* BQM: to ignore all-gapp subtree at an alignment site */
//#define IGNORE_GAP_LH

//#define TINY_POSITIVE 1e-290

template <typename D>
inline hn::TFromD<D> horizontal_add(hn::Vec<D> x) {
    return hn::GetLane(hn::SumOfLanes(D{}, x));
}

template <typename D, typename std::enable_if_t<hn::Lanes(D{}) == 2, int> = 0>
inline hn::Vec<D> horizontal_add(hn::Vec<D> x[2]) {
#if HWY_TARGET <= (1LL << HWY_HIGHEST_TARGET_BIT_X86)
    // begin x86 SSE
#if HWY_TARGET <= HWY_SSSE3
    return hn::Vec<D>{_mm_hadd_pd(x[0].raw,x[1].raw)};
#else
    return hn::Reverse2(D{}, hn::OddEven(x[0], x[1])) + hn::OddEven(x[1], x[0]);
#endif
    // end x86 SSE
#elif HWY_TARGET <= (1LL << HWY_HIGHEST_TARGET_BIT_ARM)
    // begin ARM NEON
    return hn::Vec<D>{vpaddq_f64(x[0].raw, x[1].raw)};
    // end ARM NEON
#else
    static_assert(!hn::Lanes(D{}), "Not supported");
#endif
}

template <typename D>
inline double horizontal_max(hn::Vec<D> const &a) {
    return hn::GetLane(hn::MaxOfLanes(D{}, a));
}

/*
// lower 64 bits of result contain the sum of a[0], a[1], a[2], a[3]
// upper 64 bits of result contain the sum of b[0], b[1], b[2], b[3]
static inline Vec2d horizontal_add(Vec4d const & a, Vec4d const & b) {
	// calculate 4 two-element horizontal sums:
	// lower 64 bits contain a[0] + a[1]
	// next 64 bits contain b[0] + b[1]
	// next 64 bits contain a[2] + a[3]
	// next 64 bits contain b[2] + b[3]
	__m256d sum1 = _mm256_hadd_pd(a, b);
	// extract upper 128 bits of result
	__m128d sum_high = _mm256_extractf128_pd(sum1, 1);
	// add upper 128 bits of sum to its lower 128 bits
	return _mm_add_pd(sum_high, _mm256_castpd256_pd128(sum1));
}
*/

template <typename D, typename std::enable_if_t<hn::Lanes(D{}) == 4, int> = 0>
inline hn::Vec<D> horizontal_add(hn::Vec<D> x[4]) {
#if HWY_TARGET <= (1LL << HWY_HIGHEST_TARGET_BIT_X86)
    // begin x86 SSE
#if HWY_TARGET <= HWY_AVX2
	// {a[0]+a[1], b[0]+b[1], a[2]+a[3], b[2]+b[3]}
	__m256d sumab = _mm256_hadd_pd(x[0].raw, x[1].raw);
	// {c[0]+c[1], d[0]+d[1], c[2]+c[3], d[2]+d[3]}
	__m256d sumcd = _mm256_hadd_pd(x[2].raw, x[3].raw);

	// {a[0]+a[1], b[0]+b[1], c[2]+c[3], d[2]+d[3]}
	__m256d blend = _mm256_blend_pd(sumab, sumcd, 12/* 0b1100*/);
	// {a[2]+a[3], b[2]+b[3], c[0]+c[1], d[0]+d[1]}
	__m256d perm = _mm256_permute2f128_pd(sumab, sumcd, 0x21);

	return hn::Vec<D>{_mm256_add_pd(perm, blend)};
#else
    // why does it reach this code path?
    static_assert(!hn::Lanes(D{}), "Not supported");
#endif
    // end x86 sse
#elif HWY_TARGET <= (1LL << HWY_HIGHEST_TARGET_BIT_ARM)
    // begin ARM NEON
    // no support for ARM SVE whose vector size greater than 128 bits
    static_assert(!hn::Lanes(D{}), "Not supported");
    // end ARM NEON
#else
    static_assert(!hn::Lanes(D{}), "Not supported");
#endif
}

//#define USING_SSE

void PhyloTree::changeLikelihoodKernel(LikelihoodKernel lk) {
	if (sse == lk) return;
	if ((sse == LK_EIGEN || sse == LK_EIGEN_SSE) && (lk == LK_NORMAL || lk == LK_SSE)) {
		// need to increase the memory usage when changing from new kernel to old kernel
		sse = lk;
		deleteAllPartialLh();
		initializeAllPartialLh();
		clearAllPartialLH();
	} else {
		// otherwise simply assign variable sse
		sse = lk;
	}
}


void PhyloTree::computeTipPartialLikelihood() {
	if (tip_partial_lh_computed)
		return;
	tip_partial_lh_computed = true;
	int i, x, state, nstates = aln->num_states;

	double *evec = model->getEigenvectors();
	double *inv_evec = model->getInverseEigenvectors();
	assert(inv_evec && evec);

	assert(tip_partial_lh);
	for (state = 0; state < nstates; state++)
		for (i = 0; i < nstates; i++)
			tip_partial_lh[state*nstates + i] = inv_evec[i*nstates+state];
	// special treatment for unknown char
	for (i = 0; i < nstates; i++) {
		double lh_unknown = 0.0;
		for (x = 0; x < nstates; x++)
			lh_unknown += inv_evec[i*nstates+x];
		tip_partial_lh[aln->STATE_UNKNOWN * nstates + i] = lh_unknown;
	}

	double lh_ambiguous;
	// ambiguous characters
	int ambi_aa[2] = {4+8, 32+64};
	switch (aln->seq_type) {
	case SEQ_DNA:
		for (state = 4; state < 18; state++) {
			int cstate = state-nstates+1;
			for (i = 0; i < nstates; i++) {
				lh_ambiguous = 0.0;
				for (x = 0; x < nstates; x++)
					if ((cstate) & (1 << x))
						lh_ambiguous += inv_evec[i*nstates+x];
				tip_partial_lh[state*nstates+i] = lh_ambiguous;
			}
		}
		break;
	case SEQ_PROTEIN:
		//map[(unsigned char)'B'] = 4+8+19; // N or D
		//map[(unsigned char)'Z'] = 32+64+19; // Q or E
		for (state = 0; state < 2; state++) {
			for (i = 0; i < nstates; i++) {
				lh_ambiguous = 0.0;
				for (x = 0; x < 7; x++)
					if (ambi_aa[state] & (1 << x))
						lh_ambiguous += inv_evec[i*nstates+x];
				tip_partial_lh[(state+20)*nstates+i] = lh_ambiguous;
			}
		}
		break;
	default:
		break;
	}

	//-------------------------------------------------------
	// initialize ptn_freq and ptn_invar
	//-------------------------------------------------------

	size_t nptn = aln->getNPattern();
	size_t maxptn = get_safe_upper_limit(nptn+model_factory->unobserved_ptns.size());
	int ptn;
	for (ptn = 0; ptn < nptn; ptn++)
		ptn_freq[ptn] = (*aln)[ptn].frequency;
	for (ptn = nptn; ptn < maxptn; ptn++)
		ptn_freq[ptn] = 0.0;

	// for +I model
	computePtnInvar();
}

void PhyloTree::computePtnInvar() {
	size_t nptn = aln->getNPattern(), ptn;
	size_t maxptn = get_safe_upper_limit(nptn+model_factory->unobserved_ptns.size());
	int nstates = aln->num_states;

    double *state_freq = aligned_alloc_double(nstates);
    model->getStateFrequency(state_freq);
	memset(ptn_invar, 0, maxptn*sizeof(double));
	double p_invar = site_rate->getPInvar();
	if (p_invar != 0.0) {
		for (ptn = 0; ptn < nptn; ptn++)
			if ((*aln)[ptn].is_const && (*aln)[ptn][0] < nstates) {
					ptn_invar[ptn] = p_invar * state_freq[(int) (*aln)[ptn][0]];
			}
		// ascertmain bias correction
		for (ptn = 0; ptn < model_factory->unobserved_ptns.size(); ptn++)
			ptn_invar[nptn+ptn] = p_invar * state_freq[(int)model_factory->unobserved_ptns[ptn]];
	}
	aligned_free(state_freq);
}

/**
 * this version uses Alexis' technique that stores the dot product of partial likelihoods and eigenvectors at node
 * for faster branch length optimization
 */
template <const int nstates>
void PhyloTree::computePartialLikelihoodEigen(PhyloNeighbor *dad_branch, PhyloNode *dad) {
    // don't recompute the likelihood
	assert(dad);
    if (dad_branch->partial_lh_computed & 1)
        return;
    dad_branch->partial_lh_computed |= 1;

    size_t nptn = aln->size()+model_factory->unobserved_ptns.size();
    PhyloNode *node = (PhyloNode*)(dad_branch->node);

	if (node->isLeaf()) {
	    dad_branch->lh_scale_factor = 0.0;
	    //memset(dad_branch->scale_num, 0, nptn * sizeof(UBYTE));

		if (!tip_partial_lh_computed)
			computeTipPartialLikelihood();
		return;
	}

    size_t ptn, c;
    size_t orig_ntn = aln->size();
    size_t ncat = site_rate->getNRate();
    //size_t nstates = aln->num_states;
    const size_t nstatesqr=nstates*nstates;
    size_t i, x;
    size_t block = nstates * ncat;

//    double *partial_lh = dad_branch->partial_lh;

	double *evec = model->getEigenvectors();
	double *inv_evec = model->getInverseEigenvectors();
	assert(inv_evec && evec);
	double *eval = model->getEigenvalues();

    dad_branch->lh_scale_factor = 0.0;

	// internal node
	assert(node->degree() == 3); // it works only for strictly bifurcating tree
	PhyloNeighbor *left = NULL, *right = NULL; // left & right are two neighbors leading to 2 subtrees
	FOR_NEIGHBOR_IT(node, dad, it) {
		if (!left) left = (PhyloNeighbor*)(*it); else right = (PhyloNeighbor*)(*it);
	}

	if (!left->node->isLeaf() && right->node->isLeaf()) {
		PhyloNeighbor *tmp = left;
		left = right;
		right = tmp;
	}
	if ((left->partial_lh_computed & 1) == 0)
		computePartialLikelihoodEigen<nstates>(left, node);
	if ((right->partial_lh_computed & 1) == 0)
		computePartialLikelihoodEigen<nstates>(right, node);
	dad_branch->lh_scale_factor = left->lh_scale_factor + right->lh_scale_factor;
	double partial_lh_tmp[nstates];
	double *eleft = new double[block*nstates], *eright = new double[block*nstates];

	// precompute information buffer
	for (c = 0; c < ncat; c++) {
		double *expleft = new double[nstates];
		double *expright = new double[nstates];
		double len_left = site_rate->getRate(c) * left->length;
		double len_right = site_rate->getRate(c) * right->length;
		for (i = 0; i < nstates; i++) {
			expleft[i] = exp(eval[i]*len_left);
			expright[i] = exp(eval[i]*len_right);
		}
		for (x = 0; x < nstates; x++)
			for (i = 0; i < nstates; i++) {
				eleft[c*nstatesqr+x*nstates+i] = evec[x*nstates+i] * expleft[i];
				eright[c*nstatesqr+x*nstates+i] = evec[x*nstates+i] * expright[i];
			}
		delete [] expright;
		delete [] expleft;
	}

	MappedMat(nstates) ei_inv_evec(inv_evec);
	MappedRowVec(nstates) ei_partial_lh_tmp(partial_lh_tmp);

	if (left->node->isLeaf() && right->node->isLeaf()) {
		// special treatment for TIP-TIP (cherry) case

		// pre compute information for both tips
		double *partial_lh_left = new double[(aln->STATE_UNKNOWN+1)*block];
		double *partial_lh_right = new double[(aln->STATE_UNKNOWN+1)*block];

		vector<int>::iterator it;
		for (it = aln->seq_states[left->node->id].begin(); it != aln->seq_states[left->node->id].end(); it++) {
			int state = (*it);
			for (x = 0; x < block; x++) {
				double vleft = 0.0;
				for (i = 0; i < nstates; i++) {
					vleft += eleft[x*nstates+i] * tip_partial_lh[state*nstates+i];
				}
				partial_lh_left[state*block+x] = vleft;
			}
		}

		for (it = aln->seq_states[right->node->id].begin(); it != aln->seq_states[right->node->id].end(); it++) {
			int state = (*it);
			for (x = 0; x < block; x++) {
				double vright = 0.0;
				for (i = 0; i < nstates; i++) {
					vright += eright[x*nstates+i] * tip_partial_lh[state*nstates+i];
				}
				partial_lh_right[state*block+x] = vright;
			}
		}

		for (x = 0; x < block; x++) {
			size_t addr = aln->STATE_UNKNOWN * block;
			partial_lh_left[addr+x] = 1.0;
			partial_lh_right[addr+x] = 1.0;
		}


		// scale number must be ZERO
	    memset(dad_branch->scale_num, 0, nptn * sizeof(UBYTE));
#ifdef _OPENMP
#pragma omp parallel for private(ptn, c, x, i, partial_lh_tmp)
#endif
		for (ptn = 0; ptn < nptn; ptn++) {
			double *partial_lh = dad_branch->partial_lh + ptn*block;
			int state_left = (ptn < orig_ntn) ? (aln->at(ptn))[left->node->id] : model_factory->unobserved_ptns[ptn-orig_ntn];
			int state_right = (ptn < orig_ntn) ? (aln->at(ptn))[right->node->id] : model_factory->unobserved_ptns[ptn-orig_ntn];
			for (c = 0; c < ncat; c++) {
				// compute real partial likelihood vector
				double *left = partial_lh_left + (state_left*block+c*nstates);
				double *right = partial_lh_right + (state_right*block+c*nstates);
				for (x = 0; x < nstates; x++) {
					partial_lh_tmp[x] = left[x] * right[x];
				}

				// compute dot-product with inv_eigenvector
				for (i = 0; i < nstates; i++) {
					double res = 0.0;
					for (x = 0; x < nstates; x++) {
						res += partial_lh_tmp[x]*inv_evec[i*nstates+x];
					}
					partial_lh[c*nstates+i] = res;
				}
			}
//			partial_lh += block;
		}
		delete [] partial_lh_right;
		delete [] partial_lh_left;
	} else if (left->node->isLeaf() && !right->node->isLeaf()) {
		// special treatment to TIP-INTERNAL NODE case
		// only take scale_num from the right subtree
		memcpy(dad_branch->scale_num, right->scale_num, nptn * sizeof(UBYTE));

		// pre compute information for left tip
		double *partial_lh_left = new double[(aln->STATE_UNKNOWN+1)*block];
//		double *partial_lh_right = right->partial_lh;

		vector<int>::iterator it;
		for (it = aln->seq_states[left->node->id].begin(); it != aln->seq_states[left->node->id].end(); it++) {
			int state = (*it);
			for (x = 0; x < block; x++) {
				double vleft = 0.0;
				for (i = 0; i < nstates; i++) {
					vleft += eleft[x*nstates+i] * tip_partial_lh[state*nstates+i];
				}
				partial_lh_left[state*block+x] = vleft;
			}
		}
		for (x = 0; x < block; x++) {
			size_t addr = aln->STATE_UNKNOWN * block;
			partial_lh_left[addr+x] = 1.0;
		}



		double sum_scale = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+: sum_scale) private(ptn, c, x, i, partial_lh_tmp)
#endif
		for (ptn = 0; ptn < nptn; ptn++) {
			double *partial_lh = dad_branch->partial_lh + ptn*block;
			double *partial_lh_right = right->partial_lh + ptn*block;
			int state_left = (ptn < orig_ntn) ? (aln->at(ptn))[left->node->id] : model_factory->unobserved_ptns[ptn-orig_ntn];
            double lh_max = 0.0;
            
			for (c = 0; c < ncat; c++) {
				// compute real partial likelihood vector
				for (x = 0; x < nstates; x++) {
					double vleft = 0.0, vright = 0.0;
					size_t addr = c*nstatesqr+x*nstates;
					vleft = partial_lh_left[state_left*block+c*nstates+x];
					for (i = 0; i < nstates; i++) {
						vright += eright[addr+i] * partial_lh_right[c*nstates+i];
					}
					partial_lh_tmp[x] = vleft * (vright);
				}
				// compute dot-product with inv_eigenvector
				for (i = 0; i < nstates; i++) {
					double res = 0.0;
					for (x = 0; x < nstates; x++) {
						res += partial_lh_tmp[x]*inv_evec[i*nstates+x];
					}
					partial_lh[c*nstates+i] = res;
                    lh_max = max(fabs(res), lh_max);
				}
			}
            // check if one should scale partial likelihoods
//			bool do_scale = true;
//            for (i = 0; i < block; i++)
//				if (fabs(partial_lh[i]) > SCALING_THRESHOLD) {
//					do_scale = false;
//					break;
//				}
//            assert(lh_max > 0);
            if (lh_max < SCALING_THRESHOLD) {
				// now do the likelihood scaling
				for (i = 0; i < block; i++) {
					partial_lh[i] *= SCALING_THRESHOLD_INVER;
//                    partial_lh[i] /= lh_max;
				}
				// unobserved const pattern will never have underflow
				sum_scale += LOG_SCALING_THRESHOLD * ptn_freq[ptn];
//				sum_scale += log(lh_max) * ptn_freq[ptn];
				dad_branch->scale_num[ptn] += 1;
            }


		}
		dad_branch->lh_scale_factor += sum_scale;
		delete [] partial_lh_left;

	} else {
		// both left and right are internal node
//		double *partial_lh_left = left->partial_lh;
//		double *partial_lh_right = right->partial_lh;

		double sum_scale = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+: sum_scale) private(ptn, c, x, i, partial_lh_tmp)
#endif
		for (ptn = 0; ptn < nptn; ptn++) {
			double *partial_lh = dad_branch->partial_lh + ptn*block;
			double *partial_lh_left = left->partial_lh + ptn*block;
			double *partial_lh_right = right->partial_lh + ptn*block;
            double lh_max = 0.0;
			dad_branch->scale_num[ptn] = left->scale_num[ptn] + right->scale_num[ptn];

			for (c = 0; c < ncat; c++) {
				// compute real partial likelihood vector
				for (x = 0; x < nstates; x++) {
					double vleft = 0.0, vright = 0.0;
					size_t addr = c*nstatesqr+x*nstates;
					for (i = 0; i < nstates; i++) {
						vleft += eleft[addr+i] * partial_lh_left[c*nstates+i];
						vright += eright[addr+i] * partial_lh_right[c*nstates+i];
					}
					partial_lh_tmp[x] = vleft*vright;
				}
				// compute dot-product with inv_eigenvector
				for (i = 0; i < nstates; i++) {
					double res = 0.0;
					for (x = 0; x < nstates; x++) {
						res += partial_lh_tmp[x]*inv_evec[i*nstates+x];
					}
					partial_lh[c*nstates+i] = res;
                    lh_max = max(lh_max, fabs(res));
				}
			}

            // check if one should scale partial likelihoods
//			bool do_scale = true;
//            for (i = 0; i < block; i++)
//				if (fabs(partial_lh[i]) > SCALING_THRESHOLD) {
//					do_scale = false;
//					break;
//				}
//            assert(lh_max > 0.0);
            if (lh_max < SCALING_THRESHOLD) {
				// now do the likelihood scaling
				for (i = 0; i < block; i++) {
                    partial_lh[i] *= SCALING_THRESHOLD_INVER;
//                    partial_lh[i] /= lh_max;
				}
				// unobserved const pattern will never have underflow
                sum_scale += LOG_SCALING_THRESHOLD * ptn_freq[ptn];
//				sum_scale += log(lh_max) * ptn_freq[ptn];
				dad_branch->scale_num[ptn] += 1;
            }

		}
		dad_branch->lh_scale_factor += sum_scale;

	}

	delete [] eright;
	delete [] eleft;
//	delete [] partial_lh_tmp;
}

template <const int nstates>
double PhyloTree::computeLikelihoodDervEigen(PhyloNeighbor *dad_branch, PhyloNode *dad, double &df, double &ddf) {
    PhyloNode *node = (PhyloNode*) dad_branch->node;
    PhyloNeighbor *node_branch = (PhyloNeighbor*) node->findNeighbor(dad);
    if (!central_partial_lh)
        initializeAllPartialLh();
    if (node->isLeaf()) {
    	PhyloNode *tmp_node = dad;
    	dad = node;
    	node = tmp_node;
    	PhyloNeighbor *tmp_nei = dad_branch;
    	dad_branch = node_branch;
    	node_branch = tmp_nei;
    }
    if ((dad_branch->partial_lh_computed & 1) == 0)
        computePartialLikelihoodEigen<nstates>(dad_branch, dad);
    if ((node_branch->partial_lh_computed & 1) == 0)
        computePartialLikelihoodEigen<nstates>(node_branch, node);
    double tree_lh = node_branch->lh_scale_factor + dad_branch->lh_scale_factor;
    size_t ncat = site_rate->getNRate();

//    double p_invar = site_rate->getPInvar();
    //double p_var_cat = (1.0 - p_invar) / (double) ncat;
    //size_t nstates = aln->num_states;
    size_t block = ncat * nstates;
    size_t ptn; // for big data size > 4GB memory required
    size_t c, i;
    size_t orig_nptn = aln->size();
    size_t nptn = aln->size()+model_factory->unobserved_ptns.size();
    double *eval = model->getEigenvalues();
    assert(eval);

	assert(theta_all);
	if (!theta_computed) {
		// precompute theta for fast branch length optimization
//	    double *partial_lh_dad = dad_branch->partial_lh;
//    	double *theta = theta_all;

	    if (dad->isLeaf()) {
	    	// special treatment for TIP-INTERNAL NODE case
#ifdef _OPENMP
#pragma omp parallel for private(ptn, i)
#endif
	    	for (ptn = 0; ptn < nptn; ptn++) {
				double *partial_lh_dad = dad_branch->partial_lh + ptn*block;
				double *theta = theta_all + ptn*block;
				double *lh_tip = tip_partial_lh + ((int)((ptn < orig_nptn) ? (aln->at(ptn))[dad->id] :  model_factory->unobserved_ptns[ptn-orig_nptn]))*nstates;
				for (i = 0; i < block; i++) {
					theta[i] = lh_tip[i%nstates] * partial_lh_dad[i];
				}

			}
			// ascertainment bias correction
	    } else {
	    	// both dad and node are internal nodes
		    double *partial_lh_node = node_branch->partial_lh;
		    double *partial_lh_dad = dad_branch->partial_lh;

	    	size_t all_entries = nptn*block;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	    	for (i = 0; i < all_entries; i++) {
				theta_all[i] = partial_lh_node[i] * partial_lh_dad[i];
			}
	    }
		theta_computed = true;
	}

    double *val0 = new double[block];
    double *val1 = new double[block];
    double *val2 = new double[block];
	for (c = 0; c < ncat; c++) {
		double prop = site_rate->getProp(c);
//		double cof = site_rate->getRate(c) * dad_branch->length;
		for (i = 0; i < nstates; i++) {
			double cof = eval[i]*site_rate->getRate(c);
//			double val = exp(cof*dad_branch->length);
			double val = exp(cof*dad_branch->length) * prop;
			double val1_ = cof*val;
			val0[c*nstates+i] = val;
			val1[c*nstates+i] = val1_;
			val2[c*nstates+i] = cof*val1_;
		}
	}


//    double *theta = theta_all;
    double my_df = 0.0, my_ddf = 0.0, prob_const = 0.0, df_const = 0.0, ddf_const = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+: tree_lh, my_df, my_ddf, prob_const, df_const, ddf_const) private(ptn, i)
#endif
    for (ptn = 0; ptn < nptn; ptn++) {
		double lh_ptn = ptn_invar[ptn], df_ptn = 0.0, ddf_ptn = 0.0;
		double *theta = theta_all + ptn*block;
		for (i = 0; i < block; i++) {
			lh_ptn += val0[i] * theta[i];
			df_ptn += val1[i] * theta[i];
			ddf_ptn += val2[i] * theta[i];
		}

        assert(lh_ptn > 0.0);
//        if (lh_ptn <= 0) lh_ptn = TINY_POSITIVE;
        
        if (ptn < orig_nptn) {
			double df_frac = df_ptn / lh_ptn;
			double ddf_frac = ddf_ptn / lh_ptn;
			double freq = ptn_freq[ptn];
			double tmp1 = df_frac * freq;
			double tmp2 = ddf_frac * freq;
			my_df += tmp1;
			my_ddf += tmp2 - tmp1 * df_frac;
			lh_ptn = log(lh_ptn);
			tree_lh += lh_ptn * freq;
			_pattern_lh[ptn] = lh_ptn;
		} else {
			// ascertainment bias correction
			prob_const += lh_ptn;
			df_const += df_ptn;
			ddf_const += ddf_ptn;
		}
    }
	df = my_df;
	ddf = my_ddf;

	if (orig_nptn < nptn) {
    	// ascertainment bias correction
    	prob_const = 1.0 - prob_const;
    	double df_frac = df_const / prob_const;
    	double ddf_frac = ddf_const / prob_const;
    	int nsites = aln->getNSite();
    	df += nsites * df_frac;
    	ddf += nsites *(ddf_frac + df_frac*df_frac);
    	prob_const = log(prob_const);
    	tree_lh -= nsites * prob_const;
    	for (ptn = 0; ptn < orig_nptn; ptn++)
    		_pattern_lh[ptn] -= prob_const;
    }


    delete [] val2;
    delete [] val1;
    delete [] val0;

    return tree_lh;
}

template <const int nstates>
double PhyloTree::computeLikelihoodBranchEigen(PhyloNeighbor *dad_branch, PhyloNode *dad, double *pattern_lh) {
    PhyloNode *node = (PhyloNode*) dad_branch->node;
    PhyloNeighbor *node_branch = (PhyloNeighbor*) node->findNeighbor(dad);
    if (!central_partial_lh)
        initializeAllPartialLh();
    if (node->isLeaf()) {
    	PhyloNode *tmp_node = dad;
    	dad = node;
    	node = tmp_node;
    	PhyloNeighbor *tmp_nei = dad_branch;
    	dad_branch = node_branch;
    	node_branch = tmp_nei;
    }
    if ((dad_branch->partial_lh_computed & 1) == 0)
        computePartialLikelihoodEigen<nstates>(dad_branch, dad);
    if ((node_branch->partial_lh_computed & 1) == 0)
        computePartialLikelihoodEigen<nstates>(node_branch, node);
    double tree_lh = node_branch->lh_scale_factor + dad_branch->lh_scale_factor;
    size_t ncat = site_rate->getNRate();

//    double p_invar = site_rate->getPInvar();
//    double p_var_cat = (1.0 - p_invar) / (double) ncat;
    size_t block = ncat * nstates;
    size_t ptn; // for big data size > 4GB memory required
    size_t c, i;
    size_t orig_nptn = aln->size();
    size_t nptn = aln->size()+model_factory->unobserved_ptns.size();
    double *eval = model->getEigenvalues();
    assert(eval);

//    double *partial_lh_dad = dad_branch->partial_lh;
//    double *partial_lh_node = node_branch->partial_lh;
    double *val = new double[block];
	for (c = 0; c < ncat; c++) {
		double len = site_rate->getRate(c)*dad_branch->length;
		double prop = site_rate->getProp(c);
		for (i = 0; i < nstates; i++)
			val[c*nstates+i] = exp(eval[i]*len) * prop;
	}

	double prob_const = 0.0;
//	double *lh_cat = _pattern_lh_cat;
	memset(_pattern_lh_cat, 0, nptn*ncat*sizeof(double));

    if (dad->isLeaf()) {
    	// special treatment for TIP-INTERNAL NODE case
    	double *partial_lh_node = new double[(aln->STATE_UNKNOWN+1)*block];
    	IntVector states_dad = aln->seq_states[dad->id];
    	states_dad.push_back(aln->STATE_UNKNOWN);
//    	cout << "here" << endl;
    	// precompute information from one tip
    	for (IntVector::iterator it = states_dad.begin(); it != states_dad.end(); it++) {
    		double *lh_node = partial_lh_node +(*it)*block;
    		double *lh_tip = tip_partial_lh + (*it)*nstates;
    		double *val_tmp = val;
			for (c = 0; c < ncat; c++) {
				for (i = 0; i < nstates; i++) {
					  lh_node[i] = val_tmp[i] * lh_tip[i];
				}
				lh_node += nstates;
				val_tmp += nstates;
			}
    	}
//    	cout.unsetf(ios::fixed);
//		for (c = 0; c < ncat; c++) {
//			for (i = 0; i < nstates; i++)
//				cout << partial_lh_node[aln->STATE_UNKNOWN * block + c*nstates+i] << "\t";
//			cout << endl;
//		}
//		exit(0);

    	// now do the real computation
#ifdef _OPENMP
#pragma omp parallel for reduction(+: tree_lh, prob_const) private(ptn, i, c)
#endif
    	for (ptn = 0; ptn < nptn; ptn++) {
			double lh_ptn = ptn_invar[ptn];
			double *lh_cat = _pattern_lh_cat + ptn*ncat;
			double *partial_lh_dad = dad_branch->partial_lh + ptn*block;
			int state_dad = (ptn < orig_nptn) ? (aln->at(ptn))[dad->id] : model_factory->unobserved_ptns[ptn-orig_nptn];
			double *lh_node = partial_lh_node + state_dad*block;
			for (c = 0; c < ncat; c++) {
				for (i = 0; i < nstates; i++) {
//					*lh_cat +=  val[c*nstates+i] * tip_partial_lh[state_dad*nstates+i] * partial_lh_dad[c*nstates+i];
					*lh_cat += lh_node[i] * partial_lh_dad[i];
				}
				lh_node += nstates;
				partial_lh_dad += nstates;
				lh_ptn += *lh_cat;
				lh_cat++;
			}
			assert(lh_ptn > 0.0);
//            if (lh_ptn <= 0) lh_ptn = TINY_POSITIVE;
			if (ptn < orig_nptn) {
				lh_ptn = log(lh_ptn);
				_pattern_lh[ptn] = lh_ptn;
				tree_lh += lh_ptn * ptn_freq[ptn];
			} else {
				prob_const += lh_ptn;
			}
		}
		delete [] partial_lh_node;
    } else {
    	// both dad and node are internal nodes
#ifdef _OPENMP
#pragma omp parallel for reduction(+: tree_lh, prob_const) private(ptn, i, c)
#endif
    	for (ptn = 0; ptn < nptn; ptn++) {
			double lh_ptn = ptn_invar[ptn];
			double *lh_cat = _pattern_lh_cat + ptn*ncat;
			double *partial_lh_dad = dad_branch->partial_lh + ptn*block;
			double *partial_lh_node = node_branch->partial_lh + ptn*block;
			double *val_tmp = val;
			for (c = 0; c < ncat; c++) {
				for (i = 0; i < nstates; i++) {
					*lh_cat +=  val_tmp[i] * partial_lh_node[i] * partial_lh_dad[i];
				}
				lh_ptn += *lh_cat;
				partial_lh_node += nstates;
				partial_lh_dad += nstates;
				val_tmp += nstates;
				lh_cat++;
			}

			assert(lh_ptn > 0.0);
//            if (lh_ptn <= 0) lh_ptn = TINY_POSITIVE;
            if (ptn < orig_nptn) {
				lh_ptn = log(lh_ptn);
				_pattern_lh[ptn] = lh_ptn;
				tree_lh += lh_ptn * ptn_freq[ptn];
			} else {
				prob_const += lh_ptn;
			}
		}
    }


    if (orig_nptn < nptn) {
    	// ascertainment bias correction
    	prob_const = log(1.0 - prob_const);
    	for (ptn = 0; ptn < orig_nptn; ptn++)
    		_pattern_lh[ptn] -= prob_const;
    	tree_lh -= aln->getNSite()*prob_const;
		assert(!isnan(tree_lh) && !isinf(tree_lh));
    }

	assert(!isnan(tree_lh) && !isinf(tree_lh));

    if (pattern_lh)
        memmove(pattern_lh, _pattern_lh, aln->size() * sizeof(double));
    delete [] val;
    return tree_lh;
}

/************************************************************************************************
 *
 *   SSE vectorized versions of above functions
 *
 *************************************************************************************************/


template <typename D, const int nstates>
void PhyloTree::computePartialLikelihoodEigenTipSSE(PhyloNeighbor *dad_branch, PhyloNode *dad) {
    constexpr D d{};
    // don't recompute the likelihood
	assert(dad);
    if (dad_branch->partial_lh_computed & 1)
        return;
    dad_branch->partial_lh_computed |= 1;

    size_t nptn = aln->size() + model_factory->unobserved_ptns.size();
    PhyloNode *node = (PhyloNode*)(dad_branch->node);

	if (node->isLeaf()) {
	    dad_branch->lh_scale_factor = 0.0;
	    //memset(dad_branch->scale_num, 0, nptn * sizeof(UBYTE));

		if (!tip_partial_lh_computed)
			computeTipPartialLikelihood();
		return;
	}

    size_t ptn, c;
    size_t orig_ntn = aln->size();

    size_t ncat = site_rate->getNRate();
    assert(nstates == aln->num_states && nstates >= hn::Lanes(d));
    assert(model->isReversible()); // only works with reversible model!
    const size_t nstatesqr=nstates*nstates;
    size_t i, x, j;
    size_t block = nstates * ncat;

	// internal node
	assert(node->degree() == 3); // it works only for strictly bifurcating tree
	PhyloNeighbor *left = NULL, *right = NULL; // left & right are two neighbors leading to 2 subtrees
	FOR_NEIGHBOR_IT(node, dad, it) {
		if (!left) left = (PhyloNeighbor*)(*it); else right = (PhyloNeighbor*)(*it);
	}

	if (!left->node->isLeaf() && right->node->isLeaf()) {
		// swap left and right
		PhyloNeighbor *tmp = left;
		left = right;
		right = tmp;
	}
	if ((left->partial_lh_computed & 1) == 0)
		computePartialLikelihoodEigenTipSSE<D, nstates>(left, node);
	if ((right->partial_lh_computed & 1) == 0)
		computePartialLikelihoodEigenTipSSE<D, nstates>(right, node);

//    double *partial_lh = dad_branch->partial_lh;

	double *evec = model->getEigenvectors();
	double *inv_evec = model->getInverseEigenvectors();

	hn::Vec<decltype(d)> vc_inv_evec[nstates*nstates/hn::Lanes(d)];
	assert(inv_evec && evec);
	for (i = 0; i < nstates; i++) {
		for (x = 0; x < nstates/hn::Lanes(d); x++)
			// inv_evec is not aligned!
			vc_inv_evec[i*nstates/hn::Lanes(d)+x] = hn::LoadU(d, &inv_evec[i*nstates+x*hn::Lanes(d)]);
	}
	double *eval = model->getEigenvalues();

	dad_branch->lh_scale_factor = left->lh_scale_factor + right->lh_scale_factor;

	auto eleft = aligned_alloc_double(block*nstates);
	auto eright = aligned_alloc_double(block*nstates);

	// precompute information buffer
	for (c = 0; c < ncat; c++) {
		auto vc_evec = hn::Undefined(d);
		hn::Vec<decltype(d)> expleft[nstates/hn::Lanes(d)];
		hn::Vec<decltype(d)> expright[nstates/hn::Lanes(d)];
		double len_left = site_rate->getRate(c) * left->length;
		double len_right = site_rate->getRate(c) * right->length;
		for (i = 0; i < nstates/hn::Lanes(d); i++) {
			// eval is not aligned!
			expleft[i] = hn::Exp(d, hn::LoadU(d, &eval[i*hn::Lanes(d)]) * hn::Set(d, len_left));
			expright[i] = hn::Exp(d, hn::LoadU(d, &eval[i*hn::Lanes(d)]) * hn::Set(d, len_right));
		}
		for (x = 0; x < nstates; x++)
			for (i = 0; i < nstates/hn::Lanes(d); i++) {
				// evec is not be aligned!
				vc_evec = hn::LoadU(d, &evec[x * nstates + i * hn::Lanes(d)]);
				hn::Store(vc_evec * expleft[i], d, &eleft[c * nstatesqr + x * nstates + i * hn::Lanes(d)]);
				hn::Store(vc_evec * expright[i], d, &eright[c * nstatesqr + x * nstates + i * hn::Lanes(d)]);
			}
	}

	if (left->node->isLeaf() && right->node->isLeaf()) {
		// special treatment for TIP-TIP (cherry) case

		// pre compute information for both tips
		double *partial_lh_left = aligned_alloc_double((aln->STATE_UNKNOWN+1)*block);
		double *partial_lh_right = aligned_alloc_double((aln->STATE_UNKNOWN+1)*block);

		vector<int>::iterator it;
		for (it = aln->seq_states[left->node->id].begin(); it != aln->seq_states[left->node->id].end(); it++) {
			int state = (*it);
			hn::Vec<decltype(d)> vc_partial_lh_tmp[nstates/hn::Lanes(d)];
			hn::Vec<decltype(d)> vleft[hn::Lanes(d)];
			size_t addr = state*nstates;
			for (i = 0; i < nstates/hn::Lanes(d); i++)
				vc_partial_lh_tmp[i] = hn::Load(d, &tip_partial_lh[addr+i*hn::Lanes(d)]);
			for (x = 0; x < block; x+=hn::Lanes(d)) {
				addr = x*nstates/hn::Lanes(d);
				for (j = 0; j < hn::Lanes(d); j++)
					vleft[j] = hn::Load(d, &eleft[addr * hn::Lanes(d) + j * nstates]) * vc_partial_lh_tmp[0];
				for (i = 1; i < nstates/hn::Lanes(d); i++) {
					for (j = 0; j < hn::Lanes(d); j++)
						vleft[j] = hn::MulAdd(hn::Load(d, &eleft[addr * hn::Lanes(d) + j * nstates + i * hn::Lanes(d)]),
                                              vc_partial_lh_tmp[i], vleft[j]);
				}
				hn::Store(horizontal_add<D>(vleft), d, &partial_lh_left[state*block+x]);
			}
		}

		for (it = aln->seq_states[right->node->id].begin(); it != aln->seq_states[right->node->id].end(); it++) {
			int state = (*it);
			hn::Vec<decltype(d)> vright[hn::Lanes(d)];
			hn::Vec<decltype(d)> vc_partial_lh_tmp[nstates/hn::Lanes(d)];

			for (i = 0; i < nstates/hn::Lanes(d); i++)
				vc_partial_lh_tmp[i] = hn::Load(d, &tip_partial_lh[state*nstates+i*hn::Lanes(d)]);
			for (x = 0; x < block; x+=hn::Lanes(d)) {
				for (j = 0; j < hn::Lanes(d); j++)
					vright[j] = hn::Load(d, &eright[(x+j)*nstates]) * vc_partial_lh_tmp[0];
				for (i = 1; i < nstates/hn::Lanes(d); i++) {
					for (j = 0; j < hn::Lanes(d); j++)
						vright[j] = hn::MulAdd(hn::Load(d, &eright[(x+j) * nstates + i * hn::Lanes(d)]),
                                               vc_partial_lh_tmp[i], vright[j]);
				}
				hn::Store(horizontal_add<D>(vright), d, &partial_lh_right[state*block+x]);
			}
		}

		size_t addr_unknown = aln->STATE_UNKNOWN * block;
		for (x = 0; x < block; x++) {
			partial_lh_left[addr_unknown+x] = 1.0;
			partial_lh_right[addr_unknown+x] = 1.0;
		}

		// assign pointers for left and right partial_lh
		double **lh_left_ptr = aligned_alloc<double*>(nptn);
		double **lh_right_ptr = aligned_alloc<double*>(nptn);
		for (ptn = 0; ptn < orig_ntn; ptn++) {
			lh_left_ptr[ptn] = &partial_lh_left[block *  (aln->at(ptn))[left->node->id]];
			lh_right_ptr[ptn] = &partial_lh_right[block * (aln->at(ptn))[right->node->id]];
		}
		for (ptn = orig_ntn; ptn < nptn; ptn++) {
			lh_left_ptr[ptn] = &partial_lh_left[block * model_factory->unobserved_ptns[ptn-orig_ntn]];
			lh_right_ptr[ptn] = &partial_lh_right[block * model_factory->unobserved_ptns[ptn-orig_ntn]];
		}

		// scale number must be ZERO
	    memset(dad_branch->scale_num, 0, nptn * sizeof(UBYTE));
		hn::Vec<decltype(d)> vc_partial_lh_tmp[nstates/hn::Lanes(d)];
		hn::Vec<decltype(d)> res[hn::Lanes(d)];

#ifdef _OPENMP
#pragma omp parallel for private(ptn, c, x, i, j, vc_partial_lh_tmp, res)
#endif
		for (ptn = 0; ptn < nptn; ptn++) {
	        double *partial_lh = dad_branch->partial_lh + ptn*block;

//	    	int state_left  = (ptn < orig_ntn) ? (aln->at(ptn))[left->node->id] : model_factory->unobserved_ptns[ptn-orig_ntn];
//	    	int state_right = (ptn < orig_ntn) ? (aln->at(ptn))[right->node->id] : model_factory->unobserved_ptns[ptn-orig_ntn];
//			double *lh_left  = &partial_lh_left [block * state_left];
//			double *lh_right = &partial_lh_right[block * state_right];
	        double *lh_left = lh_left_ptr[ptn];
	        double *lh_right = lh_right_ptr[ptn];
			for (c = 0; c < ncat; c++) {
				// compute real partial likelihood vector

				for (x = 0; x < nstates/hn::Lanes(d); x++) {
					vc_partial_lh_tmp[x] = hn::Load(d, &lh_left[x*hn::Lanes(d)]) * hn::Load(d, &lh_right[x*hn::Lanes(d)]);
				}
				// compute dot-product with inv_eigenvector
				for (i = 0; i < nstates; i+=hn::Lanes(d)) {
					for (j = 0; j < hn::Lanes(d); j++) {
						res[j] = vc_partial_lh_tmp[0] * vc_inv_evec[(i+j)*nstates/hn::Lanes(d)];
					}
					for (x = 1; x < nstates/hn::Lanes(d); x++)
						for (j = 0; j < hn::Lanes(d); j++) {
							res[j] = hn::MulAdd(vc_partial_lh_tmp[x], vc_inv_evec[(i+j)*nstates/hn::Lanes(d)+x], res[j]);
						}
					hn::Store(horizontal_add<D>(res), d, &partial_lh[i]);
				}

				lh_left += nstates;
				lh_right += nstates;
				partial_lh += nstates;
			}
		}

//#ifdef _OPENMP
//	    aligned_free(master_res);
//	    aligned_free(master_partial_lh_tmp);
//#endif

	    aligned_free(lh_left_ptr);
	    aligned_free(lh_right_ptr);
		aligned_free(partial_lh_right);
		aligned_free(partial_lh_left);
	} else if (left->node->isLeaf() && !right->node->isLeaf()) {
		// special treatment to TIP-INTERNAL NODE case
		// only take scale_num from the right subtree
		memcpy(dad_branch->scale_num, right->scale_num, nptn * sizeof(UBYTE));

		// pre compute information for left tip
		double *partial_lh_left = aligned_alloc_double((aln->STATE_UNKNOWN+1)*block);
//		double *partial_lh_right = right->partial_lh;


		vector<int>::iterator it;
		for (it = aln->seq_states[left->node->id].begin(); it != aln->seq_states[left->node->id].end(); it++) {
			int state = (*it);
			hn::Vec<decltype(d)> vc_tip_lh[nstates/hn::Lanes(d)];
			hn::Vec<decltype(d)> vleft[hn::Lanes(d)];
			for (i = 0; i < nstates/hn::Lanes(d); i++)
                vc_tip_lh[i] = hn::Load(d, &tip_partial_lh[state*nstates+i*hn::Lanes(d)]);
			for (x = 0; x < block; x+=hn::Lanes(d)) {
				for (j = 0; j < hn::Lanes(d); j++)
					vleft[j] = hn::Load(d, &eleft[(x+j)*nstates]) * vc_tip_lh[0];
				for (i = 1; i < nstates/hn::Lanes(d); i++) {
					for (j = 0; j < hn::Lanes(d); j++)
						vleft[j] = hn::MulAdd(hn::Load(d, &eleft[(x+j) * nstates + i * hn::Lanes(d)]),
                                              vc_tip_lh[i], vleft[j]);
				}
				hn::Store(horizontal_add<D>(vleft), d, &partial_lh_left[state*block+x]);
			}
		}

		size_t addr_unknown = aln->STATE_UNKNOWN * block;
		for (x = 0; x < block; x++) {
			partial_lh_left[addr_unknown+x] = 1.0;
		}

		// assign pointers for partial_lh_left
		double **lh_left_ptr = aligned_alloc<double*>(nptn);
		for (ptn = 0; ptn < orig_ntn; ptn++) {
			lh_left_ptr[ptn] = &partial_lh_left[block *  (aln->at(ptn))[left->node->id]];
		}
		for (ptn = orig_ntn; ptn < nptn; ptn++) {
			lh_left_ptr[ptn] = &partial_lh_left[block * model_factory->unobserved_ptns[ptn-orig_ntn]];
		}

		double sum_scale = 0.0;
		hn::Vec<decltype(d)> vc_lh_right[nstates/hn::Lanes(d)];
		hn::Vec<decltype(d)> vc_partial_lh_tmp[nstates/hn::Lanes(d)];
		hn::Vec<decltype(d)> res[hn::Lanes(d)];
		hn::Vec<decltype(d)> vc_max; // maximum of partial likelihood, for scaling check
		hn::Vec<decltype(d)> vright[hn::Lanes(d)];

#ifdef _OPENMP
#pragma omp parallel for reduction(+: sum_scale) private (ptn, c, x, i, j, vc_lh_right, vc_partial_lh_tmp, res, vc_max, vright)
#endif
		for (ptn = 0; ptn < nptn; ptn++) {
	        double *partial_lh = dad_branch->partial_lh + ptn*block;
	        double *partial_lh_right = right->partial_lh + ptn*block;

//			int state_left = (ptn < orig_ntn) ? (aln->at(ptn))[left->node->id] : model_factory->unobserved_ptns[ptn-orig_ntn];
//			double *lh_left = &partial_lh_left[block * state_left];
	        double *lh_left = lh_left_ptr[ptn];
			vc_max = hn::Set(d, 0.0);
			for (c = 0; c < ncat; c++) {
				// compute real partial likelihood vector
				for (i = 0; i < nstates/hn::Lanes(d); i++)
					vc_lh_right[i] = hn::Load(d, &partial_lh_right[i*hn::Lanes(d)]);

				for (x = 0; x < nstates/hn::Lanes(d); x++) {
					size_t addr = c * nstatesqr / hn::Lanes(d) + x * nstates;
					for (j = 0; j < hn::Lanes(d); j++) {
						vright[j] = hn::Load(d, &eright[addr * hn::Lanes(d) + nstates * j]) * vc_lh_right[0];
					}
					for (i = 1; i < nstates/hn::Lanes(d); i++)
						for (j = 0; j < hn::Lanes(d); j++) {
							vright[j] = hn::MulAdd(
                                    hn::Load(d, &eright[addr * hn::Lanes(d) + i * hn::Lanes(d) + nstates * j]),
                                    vc_lh_right[i], vright[j]);
						}
					vc_partial_lh_tmp[x] = hn::Load(d, &lh_left[x * hn::Lanes(d)]) * horizontal_add<D>(vright);
				}
				// compute dot-product with inv_eigenvector
				for (i = 0; i < nstates; i+=hn::Lanes(d)) {
					for (j = 0; j < hn::Lanes(d); j++) {
						res[j] = vc_partial_lh_tmp[0] * vc_inv_evec[(i+j)*nstates/hn::Lanes(d)];
					}
					for (x = 1; x < nstates/hn::Lanes(d); x++) {
						for (j = 0; j < hn::Lanes(d); j++) {
							res[j] = hn::MulAdd(vc_partial_lh_tmp[x], vc_inv_evec[(i+j)*nstates/hn::Lanes(d)+x], res[j]);
						}
					}
					auto sum_res = horizontal_add<D>(res);
					hn::Store(sum_res, d, &partial_lh[i]);
					vc_max = hn::Max(vc_max, hn::Abs(sum_res)); // take the maximum for scaling check
				}
				lh_left += nstates;
				partial_lh_right += nstates;
				partial_lh += nstates;
			}
            // check if one should scale partial likelihoods
			double lh_max = horizontal_max<D>(vc_max);
            if (lh_max < SCALING_THRESHOLD) {
            	// now do the likelihood scaling
            	partial_lh -= block; // revert its pointer
            	auto scale_thres = hn::Set(d, SCALING_THRESHOLD_INVER);
				for (i = 0; i < block; i+=hn::Lanes(d)) {
                    hn::Store(hn::Load(d, &partial_lh[i]) * scale_thres, d, &partial_lh[i]);
				}
				// unobserved const pattern will never have underflow
				sum_scale += LOG_SCALING_THRESHOLD * ptn_freq[ptn];
				dad_branch->scale_num[ptn] += 1;
//				if (pattern_scale)
//					pattern_scale[ptn] += LOG_SCALING_THRESHOLD;
				partial_lh += block; // increase the pointer again
            }

		}
		dad_branch->lh_scale_factor += sum_scale;

	    aligned_free(lh_left_ptr);
		aligned_free(partial_lh_left);

	} else {
		// both left and right are internal node

		double sum_scale = 0.0;
		hn::Vec<decltype(d)> vc_max; // maximum of partial likelihood, for scaling check
		hn::Vec<decltype(d)> vc_partial_lh_tmp[nstates/hn::Lanes(d)];
		hn::Vec<decltype(d)> vc_lh_left[nstates/hn::Lanes(d)], vc_lh_right[nstates/hn::Lanes(d)];
		hn::Vec<decltype(d)> res[hn::Lanes(d)];
		hn::Vec<decltype(d)> vleft[hn::Lanes(d)], vright[hn::Lanes(d)];

#ifdef _OPENMP
#pragma omp parallel for reduction (+: sum_scale) private(ptn, c, x, i, j, vc_max, vc_partial_lh_tmp, vc_lh_left, vc_lh_right, res, vleft, vright)
#endif
		for (ptn = 0; ptn < nptn; ptn++) {
	        double *partial_lh = dad_branch->partial_lh + ptn*block;
			double *partial_lh_left = left->partial_lh + ptn*block;
			double *partial_lh_right = right->partial_lh + ptn*block;

			dad_branch->scale_num[ptn] = left->scale_num[ptn] + right->scale_num[ptn];
			vc_max = hn::Set(d, 0.0);
			for (c = 0; c < ncat; c++) {
				// compute real partial likelihood vector
				for (i = 0; i < nstates/hn::Lanes(d); i++) {
					vc_lh_left[i] = hn::Load(d, &partial_lh_left[i * hn::Lanes(d)]);
					vc_lh_right[i] = hn::Load(d, &partial_lh_right[i * hn::Lanes(d)]);
				}

				for (x = 0; x < nstates/hn::Lanes(d); x++) {
					size_t addr = c*nstatesqr/hn::Lanes(d)+x*nstates;
					for (j = 0; j < hn::Lanes(d); j++) {
						size_t addr_com = addr + j * nstates / hn::Lanes(d);
						vleft[j] = hn::Load(d, &eleft[addr_com * hn::Lanes(d)]) * vc_lh_left[0];
						vright[j] = hn::Load(d, &eright[addr_com * hn::Lanes(d)]) * vc_lh_right[0];
					}
					for (i = 1; i < nstates/hn::Lanes(d); i++) {
						for (j = 0; j < hn::Lanes(d); j++) {
							size_t addr_com = addr + i + j * nstates / hn::Lanes(d);
							vleft[j] = hn::MulAdd(hn::Load(d, &eleft[addr_com * hn::Lanes(d)]), vc_lh_left[i], vleft[j]);
							vright[j] = hn::MulAdd(hn::Load(d, &eright[addr_com * hn::Lanes(d)]), vc_lh_right[i], vright[j]);
						}
					}
					vc_partial_lh_tmp[x] = horizontal_add<D>(vleft) * horizontal_add<D>(vright);
				}
				// compute dot-product with inv_eigenvector
				for (i = 0; i < nstates; i+=hn::Lanes(d)) {
					for (j = 0; j < hn::Lanes(d); j++) {
						res[j] = vc_partial_lh_tmp[0] * vc_inv_evec[(i+j)*nstates/hn::Lanes(d)];
					}
					for (x = 1; x < nstates/hn::Lanes(d); x++)
						for (j = 0; j < hn::Lanes(d); j++)
							res[j] = hn::MulAdd(vc_partial_lh_tmp[x], vc_inv_evec[(i+j)*nstates/hn::Lanes(d)+x], res[j]);

					auto sum_res = horizontal_add<D>(res);
					hn::Store(sum_res, d, &partial_lh[i]);
					vc_max = hn::Max(vc_max, hn::Abs(sum_res)); // take the maximum for scaling check
				}
				partial_lh += nstates;
				partial_lh_left += nstates;
				partial_lh_right += nstates;
			}

            // check if one should scale partial likelihoods
			double lh_max = horizontal_max<D>(vc_max);
            if (lh_max < SCALING_THRESHOLD) {
				// now do the likelihood scaling
            	partial_lh -= block; // revert its pointer
            	auto scale_thres = hn::Set(d, SCALING_THRESHOLD_INVER);
				for (i = 0; i < block; i+=hn::Lanes(d)) {
					hn::Store(hn::Load(d, &partial_lh[i]) * scale_thres, d, &partial_lh[i]);
				}
				// unobserved const pattern will never have underflow
				sum_scale += LOG_SCALING_THRESHOLD * ptn_freq[ptn];
				dad_branch->scale_num[ptn] += 1;
//				if (pattern_scale)
//					pattern_scale[ptn] += LOG_SCALING_THRESHOLD;
				partial_lh += block; // increase the pointer again
            }

//			partial_lh_left += block;
//			partial_lh_right += block;
		}
		dad_branch->lh_scale_factor += sum_scale;

	}

	aligned_free(eright);
	aligned_free(eleft);
}

template <typename D, const int nstates>
double PhyloTree::computeLikelihoodDervEigenTipSSE(PhyloNeighbor *dad_branch, PhyloNode *dad, double &df, double &ddf) {
    constexpr D d{};
    PhyloNode *node = (PhyloNode*) dad_branch->node;
    PhyloNeighbor *node_branch = (PhyloNeighbor*) node->findNeighbor(dad);
    if (!central_partial_lh)
        initializeAllPartialLh();
    if (node->isLeaf()) {
    	PhyloNode *tmp_node = dad;
    	dad = node;
    	node = tmp_node;
    	PhyloNeighbor *tmp_nei = dad_branch;
    	dad_branch = node_branch;
    	node_branch = tmp_nei;
    }
    if ((dad_branch->partial_lh_computed & 1) == 0)
        computePartialLikelihoodEigenTipSSE<D, nstates>(dad_branch, dad);
    if ((node_branch->partial_lh_computed & 1) == 0)
        computePartialLikelihoodEigenTipSSE<D, nstates>(node_branch, node);
    double tree_lh = node_branch->lh_scale_factor + dad_branch->lh_scale_factor;
    df = ddf = 0.0;
    size_t ncat = site_rate->getNRate();

//    double p_invar = site_rate->getPInvar();
//    double p_var_cat = (1.0 - p_invar) / (double) ncat;
    size_t block = ncat * nstates;
    size_t ptn; // for big data size > 4GB memory required
    size_t c, i, j;
    size_t orig_nptn = aln->size();
    size_t nptn = aln->size()+model_factory->unobserved_ptns.size();
    size_t maxptn = get_safe_upper_limit(nptn);
    double *eval = model->getEigenvalues();
    assert(eval);

	auto vc_val0 = aligned_alloc_double(block);
	auto vc_val1 = aligned_alloc_double(block);
	auto vc_val2 = aligned_alloc_double(block);

	auto vc_len = hn::Set(d, dad_branch->length);
	for (c = 0; c < ncat; c++) {
		auto vc_rate = hn::Set(d, site_rate->getRate(c));
		auto vc_prop = hn::Set(d, site_rate->getProp(c));
		for (i = 0; i < nstates / hn::Lanes(d); i++) {
			auto cof = hn::Load(d, &eval[i * hn::Lanes(d)]) * vc_rate;
			auto val = hn::Exp(d, cof * vc_len) * vc_prop;
			auto val1_ = cof * val;
			hn::Store(val,          d, &vc_val0[c * nstates + i * hn::Lanes(d)]);
			hn::Store(val1_,        d, &vc_val1[c * nstates + i * hn::Lanes(d)]);
			hn::Store(cof*val1_,    d, &vc_val2[c * nstates + i * hn::Lanes(d)]);
		}
	}

	assert(theta_all);
	if (!theta_computed) {
		theta_computed = true;
		// precompute theta for fast branch length optimization


		if (dad->isLeaf()) {
	    	// special treatment for TIP-INTERNAL NODE case
#ifdef _OPENMP
#pragma omp parallel for private(ptn, i)
#endif
			for (ptn = 0; ptn < orig_nptn; ptn++) {
			    double *partial_lh_dad = dad_branch->partial_lh + ptn*block;
				double *theta = theta_all + ptn*block;
				double *lh_dad = &tip_partial_lh[(aln->at(ptn))[dad->id] * nstates];
				for (i = 0; i < block; i+=hn::Lanes(d)) {
					hn::Store(hn::Load(d, &lh_dad[i%nstates]) * hn::Load(d, &partial_lh_dad[i]), d, &theta[i]);
				}
//				partial_lh_dad += block;
//				theta += block;
			}
			// ascertainment bias correction
			for (ptn = orig_nptn; ptn < nptn; ptn++) {
			    double *partial_lh_dad = dad_branch->partial_lh + ptn*block;
				double *theta = theta_all + ptn*block;
				double *lh_dad = &tip_partial_lh[model_factory->unobserved_ptns[ptn-orig_nptn] * nstates];
				for (i = 0; i < block; i+=hn::Lanes(d)) {
					hn::Store(hn::Load(d, &lh_dad[i%nstates]) * hn::Load(d, &partial_lh_dad[i]), d, &theta[i]);
				}
//				partial_lh_dad += block;
//				theta += block;
			}
	    } else {
	    	// both dad and node are internal nodes
		    double *partial_lh_node = node_branch->partial_lh;
		    double *partial_lh_dad = dad_branch->partial_lh;
	    	size_t all_entries = nptn*block;
#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
	    	for (i = 0; i < all_entries; i+=hn::Lanes(d)) {
				hn::Store(hn::Load(d, &partial_lh_node[i]) * hn::Load(d, &partial_lh_dad[i]), d, &theta_all[i]);
			}
	    }
		if (nptn < maxptn) {
			// copy dummy values
			for (ptn = nptn; ptn < maxptn; ptn++)
				memcpy(&theta_all[ptn*block], theta_all, block*sizeof(double));
		}
	}



//    double *theta = theta_all;

	hn::Vec<decltype(d)> vc_ptn[hn::Lanes(d)], vc_df[hn::Lanes(d)], vc_ddf[hn::Lanes(d)], vc_theta[hn::Lanes(d)];
//	hn::Vec<decltype(d)> vc_var_cat = p_var_cat;
	hn::Vec<decltype(d)> vc_unit = hn::Set(d, 1.0);
	hn::Vec<decltype(d)> vc_freq;
	hn::Vec<decltype(d)> lh_final = hn::Zero(d), df_final = hn::Zero(d), ddf_final = hn::Zero(d);
	// these stores values of 2 consecutive patterns
	hn::Vec<decltype(d)> lh_ptn, df_ptn, ddf_ptn, inv_lh_ptn;
//    VectorClass tiny_num(TINY_POSITIVE);

	// perform 2 sites at the same time for SSE/AVX efficiency

#ifdef _OPENMP
#pragma omp parallel private (ptn, i, j, vc_freq, vc_ptn, vc_df, vc_ddf, vc_theta, inv_lh_ptn, lh_ptn, df_ptn, ddf_ptn)
	{
	auto lh_final_th = hn::Zero(d);
	auto df_final_th = hn::Zero(d);
	auto ddf_final_th = hn::Zero(d);
#pragma omp for nowait
#endif
	for (ptn = 0; ptn < orig_nptn; ptn+=hn::Lanes(d)) {
		double *theta = theta_all + ptn*block;
		// initialization
		for (i = 0; i < hn::Lanes(d); i++) {
			vc_theta[i] = hn::Load(d, theta + i * block);
			vc_ptn[i]   = hn::Load(d, &vc_val0[0]) * vc_theta[i];
			vc_df[i]    = hn::Load(d, &vc_val1[0]) * vc_theta[i];
			vc_ddf[i]   = hn::Load(d, &vc_val2[0]) * vc_theta[i];
		}

		for (i = 1; i < block/hn::Lanes(d); i++) {
			for (j = 0; j < hn::Lanes(d); j++) {
				vc_theta[j] = hn::Load(d, &theta[i * hn::Lanes(d) + j * block]);
				vc_ptn[j]   = hn::MulAdd(vc_theta[j], hn::Load(d, &vc_val0[i * hn::Lanes(d)]), vc_ptn[j]);
				vc_df[j]    = hn::MulAdd(vc_theta[j], hn::Load(d, &vc_val1[i * hn::Lanes(d)]), vc_df[j]);
				vc_ddf[j]   = hn::MulAdd(vc_theta[j], hn::Load(d, &vc_val2[i * hn::Lanes(d)]), vc_ddf[j]);
			}
		}
//		theta += block*VCSIZE;

//		lh_ptn = mul_add(horizontal_add(vc_ptn), vc_var_cat, VectorClass().load_a(&ptn_invar[ptn]));
//		inv_lh_ptn = vc_var_cat/lh_ptn;
		lh_ptn = horizontal_add<D>(vc_ptn) + hn::Load(d, &ptn_invar[ptn]);
        // BQM: to avoid rare case that lh_ptn == 0
//        lh_ptn = max(lh_ptn, tiny_num);

		inv_lh_ptn = vc_unit / lh_ptn;

		lh_ptn = hn::Log(d, lh_ptn);
		hn::Store(lh_ptn, d, &_pattern_lh[ptn]);
		vc_freq = hn::Load(d, &ptn_freq[ptn]);

		df_ptn = horizontal_add<D>(vc_df) * inv_lh_ptn;
		ddf_ptn = horizontal_add<D>(vc_ddf) * inv_lh_ptn;

		// multiply with pattern frequency
//		lh_ptn *= vc_freq;
//		ddf_ptn = (ddf_ptn - df_ptn * df_ptn); // this must become before changing df_ptn
//		df_ptn *= vc_freq;
		ddf_ptn = hn::NegMulAdd(df_ptn, df_ptn, ddf_ptn);

#ifdef _OPENMP
		lh_final_th     = hn::MulAdd(lh_ptn, vc_freq, lh_final_th);
		df_final_th     = hn::MulAdd(df_ptn, vc_freq, df_final_th);
		ddf_final_th    = hn::MulAdd(ddf_ptn, vc_freq, ddf_final_th);
#else
		lh_final        = hn::MulAdd(lh_ptn, vc_freq, lh_final);
		df_final        = hn::MulAdd(df_ptn, vc_freq, df_final);
		ddf_final       = hn::MulAdd(ddf_ptn, vc_freq, ddf_final);
#endif

	}

#ifdef _OPENMP
#pragma omp critical
	{
		lh_final += lh_final_th;
		df_final += df_final_th;
		ddf_final += ddf_final_th;
	}
}
#endif
	tree_lh += horizontal_add<D>(lh_final);
	df = horizontal_add<D>(df_final);
	ddf = horizontal_add<D>(ddf_final);

	assert(isnormal(tree_lh));
	if (orig_nptn < nptn) {
		// ascertaiment bias correction
		lh_final    = hn::Zero(d);
		df_final    = hn::Zero(d);
		ddf_final   = hn::Zero(d);
		lh_ptn      = hn::Zero(d);
		df_ptn      = hn::Zero(d);
		ddf_ptn     = hn::Zero(d);
		double prob_const, df_const, ddf_const;
		double *theta = &theta_all[orig_nptn*block];
		for (ptn = orig_nptn; ptn < nptn; ptn+=hn::Lanes(d)) {
			lh_final += lh_ptn;
			df_final += df_ptn;
			ddf_final += ddf_ptn;

			// initialization
			for (i = 0; i < hn::Lanes(d); i++) {
				vc_theta[i] = hn::Load(d, theta + i * block);
				vc_ptn[i]   = hn::Load(d, &vc_val0[0]) * vc_theta[i];
				vc_df[i]    = hn::Load(d, &vc_val1[0]) * vc_theta[i];
				vc_ddf[i]   = hn::Load(d, &vc_val2[0]) * vc_theta[i];
			}

			for (i = 1; i < block/hn::Lanes(d); i++) {
				for (j = 0; j < hn::Lanes(d); j++) {
					vc_theta[j] = hn::Load(d, &theta[i * hn::Lanes(d) + j * block]);
					vc_ptn[j]   = hn::MulAdd(vc_theta[j], hn::Load(d, &vc_val0[i * hn::Lanes(d)]), vc_ptn[j]);
					vc_df[j]    = hn::MulAdd(vc_theta[j], hn::Load(d, &vc_val1[i * hn::Lanes(d)]), vc_df[j]);
					vc_ddf[j]   = hn::MulAdd(vc_theta[j], hn::Load(d, &vc_val2[i * hn::Lanes(d)]), vc_ddf[j]);
				}
			}
			theta += block * hn::Lanes(d);

			// ptn_invar[ptn] is not aligned
//			lh_ptn = mul_add(horizontal_add(vc_ptn), vc_var_cat, VectorClass().load(&ptn_invar[ptn]));
//			df_ptn = horizontal_add(vc_df) * vc_var_cat;
//			ddf_ptn = horizontal_add(vc_ddf) * vc_var_cat;
			lh_ptn = horizontal_add<D>(vc_ptn) + hn::LoadU(d, &ptn_invar[ptn]);

		}
		switch ((nptn-orig_nptn) % hn::Lanes(d)) {
		case 0:
			prob_const  = horizontal_add<D>(lh_final   + lh_ptn);
			df_const    = horizontal_add<D>(df_final   + df_ptn);
			ddf_const   = horizontal_add<D>(ddf_final  + ddf_ptn);
			break;
		case 1:
			prob_const  = horizontal_add<D>(lh_final)  + hn::GetLane(lh_ptn);
			df_const    = horizontal_add<D>(df_final)  + hn::GetLane(df_ptn);
			ddf_const   = horizontal_add<D>(ddf_final) + hn::GetLane(ddf_ptn);
			break;
		case 2:
			prob_const  = horizontal_add<D>(lh_final)  + hn::GetLane(lh_ptn)  + hn::ExtractLane(lh_ptn, 1);
			df_const    = horizontal_add<D>(df_final)  + hn::GetLane(df_ptn)  + hn::ExtractLane(df_ptn, 1);
			ddf_const   = horizontal_add<D>(ddf_final) + hn::GetLane(ddf_ptn) + hn::ExtractLane(ddf_ptn, 1);
			break;
		case 3:
			prob_const  = horizontal_add<D>(lh_final)  + hn::GetLane(lh_ptn)
                        + hn::ExtractLane(lh_ptn, 1)  + hn::ExtractLane(lh_ptn, 2);
			df_const    = horizontal_add<D>(df_final)  + hn::GetLane(df_ptn)
                        + hn::ExtractLane(df_ptn, 1)  + hn::ExtractLane(df_ptn, 2);
			ddf_const   = horizontal_add<D>(ddf_final) + hn::GetLane(ddf_ptn)
                        + hn::ExtractLane(ddf_ptn, 1) + hn::ExtractLane(ddf_ptn, 2);
			break;
		default:
			assert(0);
			break;
		}
    	prob_const = 1.0 - prob_const;
    	double df_frac = df_const / prob_const;
    	double ddf_frac = ddf_const / prob_const;
    	int nsites = aln->getNSite();
    	df += nsites * df_frac;
    	ddf += nsites *(ddf_frac + df_frac*df_frac);
    	prob_const = log(prob_const);
    	tree_lh -= nsites * prob_const;
    	for (ptn = 0; ptn < orig_nptn; ptn++)
    		_pattern_lh[ptn] -= prob_const;
	}

    aligned_free(vc_val2);
    aligned_free(vc_val1);
    aligned_free(vc_val0);
    return tree_lh;
}


template <typename D, const int nstates>
double PhyloTree::computeLikelihoodBranchEigenTipSSE(PhyloNeighbor *dad_branch, PhyloNode *dad, double *pattern_lh) {
    constexpr D d{};
    PhyloNode *node = (PhyloNode*) dad_branch->node;
    PhyloNeighbor *node_branch = (PhyloNeighbor*) node->findNeighbor(dad);
    if (!central_partial_lh)
        initializeAllPartialLh();
    if (node->isLeaf()) {
    	PhyloNode *tmp_node = dad;
    	dad = node;
    	node = tmp_node;
    	PhyloNeighbor *tmp_nei = dad_branch;
    	dad_branch = node_branch;
    	node_branch = tmp_nei;
    }
    if ((dad_branch->partial_lh_computed & 1) == 0)
        computePartialLikelihoodEigenTipSSE<D, nstates>(dad_branch, dad);
    if ((node_branch->partial_lh_computed & 1) == 0)
        computePartialLikelihoodEigenTipSSE<D, nstates>(node_branch, node);
    double tree_lh = node_branch->lh_scale_factor + dad_branch->lh_scale_factor;
    size_t ncat = site_rate->getNRate();

//    double p_invar = site_rate->getPInvar();
//    double p_var_cat = (1.0 - p_invar) / (double) ncat;
    size_t block = ncat * nstates;
    size_t ptn; // for big data size > 4GB memory required
    size_t c, i, j;
    size_t orig_nptn = aln->size();
    size_t nptn = aln->size()+model_factory->unobserved_ptns.size();
    size_t maxptn = get_safe_upper_limit(nptn);
    double *eval = model->getEigenvalues();
    assert(eval);

//    double *partial_lh_dad = dad_branch->partial_lh;
//    double *partial_lh_node = node_branch->partial_lh;
    auto vc_val = aligned_alloc_double(block);


	for (c = 0; c < ncat; c++) {
		double len = site_rate->getRate(c)*dad_branch->length;
		auto vc_len = hn::Set(d, len);
		auto vc_prop = hn::Set(d, site_rate->getProp(c));
		for (i = 0; i < nstates/hn::Lanes(d); i++) {
			// eval is not aligned!
			hn::Store(hn::Exp(d, hn::LoadU(d, &eval[i * hn::Lanes(d)]) * vc_len) * vc_prop, d,
                      &vc_val[c * nstates + i * hn::Lanes(d)]);
		}
	}

	double prob_const = 0.0;

	if (dad->isLeaf()) {
    	// special treatment for TIP-INTERNAL NODE case
    	hn::Vec<decltype(d)> vc_tip_partial_lh[nstates];
    	hn::Vec<decltype(d)> vc_partial_lh_dad[hn::Lanes(d)], vc_ptn[hn::Lanes(d)];
//    	hn::Vec<decltype(d)> vc_var_cat(p_var_cat);
    	hn::Vec<decltype(d)> lh_final = hn::Zero(d), vc_freq;
		hn::Vec<decltype(d)> lh_ptn; // store likelihoods of VCSIZE consecutive patterns
//        hn::Vec<decltype(d)> tiny_num(TINY_POSITIVE);

    	double **lh_states_dad = aligned_alloc<double*>(maxptn);
    	for (ptn = 0; ptn < orig_nptn; ptn++)
    		lh_states_dad[ptn] = &tip_partial_lh[(aln->at(ptn))[dad->id] * nstates];
    	for (ptn = orig_nptn; ptn < nptn; ptn++)
    		lh_states_dad[ptn] = &tip_partial_lh[model_factory->unobserved_ptns[ptn-orig_nptn] * nstates];
    	// initialize beyond #patterns for efficiency
    	for (ptn = nptn; ptn < maxptn; ptn++)
    		lh_states_dad[ptn] = &tip_partial_lh[aln->STATE_UNKNOWN * nstates];

		// copy dummy values because VectorClass will access beyond nptn
		for (ptn = nptn; ptn < maxptn; ptn++)
			memcpy(&dad_branch->partial_lh[ptn*block], dad_branch->partial_lh, block*sizeof(double));

#ifdef _OPENMP
#pragma omp parallel private(ptn, i, j, vc_tip_partial_lh, vc_partial_lh_dad, vc_ptn, vc_freq, lh_ptn)
    {
    	auto lh_final_th = hn::Zero(d);
#pragma omp for nowait
#endif
   		// main loop over all patterns with a step size of VCSIZE
		for (ptn = 0; ptn < orig_nptn; ptn+=hn::Lanes(d)) {
			double *partial_lh_dad = dad_branch->partial_lh + ptn*block;

			// initialize vc_tip_partial_lh
			for (j = 0; j < hn::Lanes(d); j++) {
				double *lh_dad = lh_states_dad[ptn+j];
				for (i = 0; i < nstates/hn::Lanes(d); i++) {
//					vc_tip_partial_lh[j*(nstates/hn::Lanes(d))+i].load_a(&tip_partial_lh[state_dad*nstates+i*hn::Lanes(d)]);
					vc_tip_partial_lh[j*(nstates/hn::Lanes(d))+i] = hn::Load(d, &lh_dad[i * hn::Lanes(d)]);
				}
				vc_partial_lh_dad[j] = hn::Load(d, &partial_lh_dad[j * block]);
				vc_ptn[j] = hn::Load(d, &vc_val[0])
                                * vc_tip_partial_lh[j * (nstates / hn::Lanes(d))] * vc_partial_lh_dad[j];
			}

			// compute vc_ptn
			for (i = 1; i < block/hn::Lanes(d); i++)
				for (j = 0; j < hn::Lanes(d); j++) {
					vc_partial_lh_dad[j] = hn::Load(d, &partial_lh_dad[j * block + i * hn::Lanes(d)]);
					vc_ptn[j] = hn::MulAdd(hn::Load(d, &vc_val[i * hn::Lanes(d)])
                                    * vc_tip_partial_lh[j * (nstates / hn::Lanes(d)) + i % (nstates / hn::Lanes(d))],
                                    vc_partial_lh_dad[j], vc_ptn[j]);
				}

			vc_freq = hn::Load(d, &ptn_freq[ptn]);
//			lh_ptn = mul_add(horizontal_add(vc_ptn), vc_var_cat, VectorClass().load_a(&ptn_invar[ptn]));
			lh_ptn = horizontal_add<D>(vc_ptn) + hn::Load(d, &ptn_invar[ptn]);
            // BQM: to avoid rare case that lh_ptn == 0
//            lh_ptn = max(lh_ptn, tiny_num);
			lh_ptn = hn::Log(d, lh_ptn);
			hn::Store(lh_ptn, d, &_pattern_lh[ptn]);

			// multiply with pattern frequency
//			lh_ptn *= vc_freq;
#ifdef _OPENMP
			lh_final_th = hn::MulAdd(lh_ptn, vc_freq, lh_final_th);
#else
			lh_final = hn::MulAdd(lh_ptn, vc_freq, lh_final);
#endif
//			partial_lh_dad += block*VCSIZE;
		}

#ifdef _OPENMP
#pragma omp critical
		{
			lh_final += lh_final_th;
    	}
    }
#endif
		tree_lh += horizontal_add<D>(lh_final);
		assert(!isnan(tree_lh) && !isinf(tree_lh));

//		switch (orig_nptn%VCSIZE) {
//		case 0: tree_lh += horizontal_add(lh_final+lh_ptn); break;
//		case 1: tree_lh += horizontal_add(lh_final)+lh_ptn[0]; break;
//		case 2: tree_lh += horizontal_add(lh_final)+lh_ptn[0]+lh_ptn[1]; break;
//		case 3: tree_lh += horizontal_add(lh_final)+lh_ptn[0]+lh_ptn[1]+lh_ptn[2]; break;
//		default: assert(0); break;
//		}


		// ascertainment bias correction
		if (orig_nptn < nptn) {
			lh_final = hn::Zero(d);
			lh_ptn = hn::Zero(d);
			for (ptn = orig_nptn; ptn < nptn; ptn+=hn::Lanes(d)) {
				double *partial_lh_dad = &dad_branch->partial_lh[ptn*block];
				lh_final += lh_ptn;

				// initialize vc_tip_partial_lh
				for (j = 0; j < hn::Lanes(d); j++) {
					double *lh_dad = lh_states_dad[ptn+j];
					for (i = 0; i < nstates / hn::Lanes(d); i++) {
	//					vc_tip_partial_lh[j*(nstates/hn::Lanes(d))+i].load_a(&tip_partial_lh[state_dad*nstates+i*hn::Lanes(d)]);
						vc_tip_partial_lh[j*(nstates/hn::Lanes(d))+i] = hn::Load(d, &lh_dad[i * hn::Lanes(d)]);
					}
					vc_partial_lh_dad[j] = hn::Load(d, &partial_lh_dad[j * block]);
					vc_ptn[j] = hn::Load(d, &vc_val[0])
                                    * vc_tip_partial_lh[j * (nstates / hn::Lanes(d))] * vc_partial_lh_dad[j];
				}

				// compute vc_ptn
				for (i = 1; i < block/hn::Lanes(d); i++)
					for (j = 0; j < hn::Lanes(d); j++) {
						vc_partial_lh_dad[j] = hn::Load(d, &partial_lh_dad[j * block + i * hn::Lanes(d)]);
						vc_ptn[j] = hn::MulAdd(hn::Load(d, &vc_val[i * hn::Lanes(d)])
                                * vc_tip_partial_lh[j * (nstates / hn::Lanes(d)) + i % (nstates / hn::Lanes(d))],
								vc_partial_lh_dad[j], vc_ptn[j]);
					}
				// ptn_invar[ptn] is not aligned
//				lh_ptn = mul_add(horizontal_add(vc_ptn), vc_var_cat, VectorClass().load(&ptn_invar[ptn]));
				lh_ptn = horizontal_add<D>(vc_ptn) + hn::Load(d, &ptn_invar[ptn]);
//				partial_lh_dad += block*VCSIZE;
			}
			switch ((nptn - orig_nptn) % hn::Lanes(d)) {
			case 0: prob_const = horizontal_add<D>(lh_final  + lh_ptn); break;
			case 1: prob_const = horizontal_add<D>(lh_final) + hn::GetLane(lh_ptn); break;
            case 2: prob_const = horizontal_add<D>(lh_final) + hn::GetLane(lh_ptn) + hn::ExtractLane(lh_ptn, 1); break;
            case 3: prob_const = horizontal_add<D>(lh_final) + hn::GetLane(lh_ptn) + hn::ExtractLane(lh_ptn, 1)
                                                                                   + hn::ExtractLane(lh_ptn, 2); break;
			default: assert(0); break;
			}
		}
		aligned_free(lh_states_dad);
    } else {
    	// both dad and node are internal nodes
    	hn::Vec<decltype(d)> vc_partial_lh_node[hn::Lanes(d)];
    	hn::Vec<decltype(d)> vc_partial_lh_dad[hn::Lanes(d)], vc_ptn[hn::Lanes(d)];
//    	hn::Vec<decltype(d)> vc_var_cat(p_var_cat);
    	hn::Vec<decltype(d)> lh_final = hn::Zero(d), vc_freq;
		hn::Vec<decltype(d)> lh_ptn;
//        VectorClass tiny_num(TINY_POSITIVE);

		// copy dummy values because VectorClass will access beyond nptn
		for (ptn = nptn; ptn < maxptn; ptn++) {
			memcpy(&dad_branch->partial_lh[ptn*block], dad_branch->partial_lh, block*sizeof(double));
			memcpy(&node_branch->partial_lh[ptn*block], node_branch->partial_lh, block*sizeof(double));
		}

#ifdef _OPENMP
#pragma omp parallel private(ptn, i, j, vc_partial_lh_node, vc_partial_lh_dad, vc_ptn, vc_freq, lh_ptn)
		{
		VectorClass lh_final_th = 0.0;
#pragma omp for nowait
#endif
		for (ptn = 0; ptn < orig_nptn; ptn+=hn::Lanes(d)) {
			double *partial_lh_dad = dad_branch->partial_lh + ptn*block;
			double *partial_lh_node = node_branch->partial_lh + ptn*block;

			for (j = 0; j < hn::Lanes(d); j++)
				vc_ptn[j] = hn::Zero(d);

			for (i = 0; i < block; i+=hn::Lanes(d)) {
				for (j = 0; j < hn::Lanes(d); j++) {
					vc_partial_lh_node[j] = hn::Load(d, &partial_lh_node[i + j * block]);
					vc_partial_lh_dad[j]  = hn::Load(d, &partial_lh_dad [i + j * block]);
					vc_ptn[j] = hn::MulAdd(hn::Load(d, &vc_val[i]) * vc_partial_lh_node[j],
                                           vc_partial_lh_dad[j], vc_ptn[j]);
				}
			}

			vc_freq = hn::Load(d, &ptn_freq[ptn]);

//			lh_ptn = mul_add(horizontal_add(vc_ptn), p_var_cat, VectorClass().load_a(&ptn_invar[ptn]));
			lh_ptn = horizontal_add<D>(vc_ptn) + hn::Load(d, &ptn_invar[ptn]);
            // BQM: to avoid rare case that lh_ptn == 0
//            lh_ptn = max(lh_ptn, tiny_num);

			lh_ptn = hn::Log(d, lh_ptn);
			hn::Store(lh_ptn, d, &_pattern_lh[ptn]);
//			lh_ptn *= vc_freq;
#ifdef _OPENMP
			lh_final_th = hn::MulAdd(lh_ptn, vc_freq, lh_final_th);
#else
			lh_final    = hn::MulAdd(lh_ptn, vc_freq, lh_final);
#endif
//			partial_lh_node += block*VCSIZE;
//			partial_lh_dad += block*VCSIZE;
		}
#ifdef _OPENMP
#pragma omp critical
		{
			lh_final += lh_final_th;
		}
	}
#endif

		tree_lh += horizontal_add<D>(lh_final);
		assert(!isnan(tree_lh) && !isinf(tree_lh));

//		switch (orig_nptn%VCSIZE) {
//		case 0: tree_lh += horizontal_add(lh_final+lh_ptn); break;
//		case 1: tree_lh += horizontal_add(lh_final)+lh_ptn[0]; break;
//		case 2: tree_lh += horizontal_add(lh_final)+lh_ptn[0]+lh_ptn[1]; break;
//		case 3:tree_lh += horizontal_add(lh_final)+lh_ptn[0]+lh_ptn[1]+lh_ptn[2]; break;
//		default: assert(0); break;
//		}

		if (orig_nptn < nptn) {
			// ascertainment bias correction
			lh_final = hn::Zero(d);
			lh_ptn = hn::Zero(d);
			double *partial_lh_node = &node_branch->partial_lh[orig_nptn*block];
			double *partial_lh_dad = &dad_branch->partial_lh[orig_nptn*block];

			for (ptn = orig_nptn; ptn < nptn; ptn+=hn::Lanes(d)) {
				lh_final += lh_ptn;

				for (j = 0; j < hn::Lanes(d); j++)
					vc_ptn[j] = hn::Zero(d);

				for (i = 0; i < block; i+=hn::Lanes(d)) {
					for (j = 0; j < hn::Lanes(d); j++) {
						vc_partial_lh_node[j] = hn::Load(d, &partial_lh_node[i + j * block]);
						vc_partial_lh_dad[j]  = hn::Load(d, &partial_lh_dad [i + j * block]);
						vc_ptn[j] = hn::MulAdd(hn::Load(d, &vc_val[i]) * vc_partial_lh_node[j],
                                               vc_partial_lh_dad[j], vc_ptn[j]);
					}
				}

				// ptn_invar[ptn] is not aligned
//				lh_ptn = mul_add(horizontal_add(vc_ptn), p_var_cat, VectorClass().load(&ptn_invar[ptn]));
				lh_ptn = horizontal_add<D>(vc_ptn) + hn::LoadU(d, &ptn_invar[ptn]);
				partial_lh_node += block * hn::Lanes(d);
				partial_lh_dad  += block * hn::Lanes(d);
			}
			switch ((nptn - orig_nptn) % hn::Lanes(d)) {
			case 0: prob_const = horizontal_add<D>(lh_final  + lh_ptn); break;
			case 1: prob_const = horizontal_add<D>(lh_final) + hn::GetLane(lh_ptn); break;
            case 2: prob_const = horizontal_add<D>(lh_final) + hn::GetLane(lh_ptn) + hn::ExtractLane(lh_ptn, 1); break;
            case 3: prob_const = horizontal_add<D>(lh_final) + hn::GetLane(lh_ptn) + hn::ExtractLane(lh_ptn, 1)
                                                                                   + hn::ExtractLane(lh_ptn, 2); break;
			default: assert(0); break;
			}
		}
    }

	if (orig_nptn < nptn) {
    	// ascertainment bias correction
    	prob_const = log(1.0 - prob_const);
    	for (ptn = 0; ptn < orig_nptn; ptn++)
    		_pattern_lh[ptn] -= prob_const;
    	tree_lh -= aln->getNSite()*prob_const;
    }

    if (pattern_lh)
        memmove(pattern_lh, _pattern_lh, aln->size() * sizeof(double));
    aligned_free(vc_val);
    return tree_lh;
}


/************************************************************************************************
 *
 *   SSE vectorized functions of the Naive implementation
 *
 *************************************************************************************************/

template<const int NSTATES>
inline double PhyloTree::computeLikelihoodBranchSSE(PhyloNeighbor *dad_branch, PhyloNode *dad, double *pattern_lh) {
    PhyloNode *node = (PhyloNode*) dad_branch->node; // Node A
    PhyloNeighbor *node_branch = (PhyloNeighbor*) node->findNeighbor(dad); // Node B
    assert(node_branch);
    if (!central_partial_lh)
        initializeAllPartialLh();
    // swap node and dad if dad is a leaf
    if (node->isLeaf()) {
        PhyloNode *tmp_node = dad;
        dad = node;
        node = tmp_node;
        PhyloNeighbor *tmp_nei = dad_branch;
        dad_branch = node_branch;
        node_branch = tmp_nei;
        //cout << "swapped\n";
    }
    if ((dad_branch->partial_lh_computed & 1) == 0)
        computePartialLikelihoodSSE<NSTATES>(dad_branch, dad);
    if ((node_branch->partial_lh_computed & 1) == 0)
        computePartialLikelihoodSSE<NSTATES>(node_branch, node);

    // now combine likelihood at the branch
    double tree_lh = node_branch->lh_scale_factor + dad_branch->lh_scale_factor;
    int ptn, cat, state1, state2;
    double *partial_lh_site;
    double *partial_lh_child;
    double *trans_state;
    double p_invar = site_rate->getPInvar();
    int numCat = site_rate->getNRate();
    int numStates = model->num_states;
    int tranSize = numStates * numStates;
    int alnSize = aln->size() + model_factory->unobserved_ptns.size();
    int orig_alnSize = aln->size();
    int block = numStates * numCat;

    double p_var_cat = (1.0 - p_invar) / (double) numCat;

    EIGEN_ALIGN16 double *trans_mat_orig = new double[numCat * tranSize + 1];
    double *trans_mat = trans_mat_orig;
    if (((intptr_t) trans_mat) % 16 != 0)
        trans_mat = trans_mat + 1;
    EIGEN_ALIGN16 double state_freq[NSTATES];
    model->getStateFrequency(state_freq);
    for (cat = 0; cat < numCat; cat++) {
        double *trans_cat = trans_mat + (cat * tranSize);
        model_factory->computeTransMatrix(dad_branch->length * site_rate->getRate(cat), trans_cat);
        for (state1 = 0; state1 < NSTATES; state1++) {
            double *trans_mat_state = trans_cat + (state1 * NSTATES);
            for (state2 = 0; state2 < NSTATES; state2++)
                trans_mat_state[state2] *= state_freq[state1];
        }
    }

    double prob_const = 0.0; // probability of unobserved const patterns

#ifdef _OPENMP
#pragma omp parallel for reduction(+: tree_lh, prob_const) private(ptn, cat)
#endif
    for (ptn = 0; ptn < alnSize; ++ptn) {
        double lh_ptn = 0.0; // likelihood of the pattern
        for (cat = 0; cat < numCat; cat++) {
            partial_lh_site = node_branch->partial_lh + (ptn * block + cat * NSTATES);
            partial_lh_child = dad_branch->partial_lh + (ptn * block + cat * NSTATES);
            trans_state = trans_mat + cat * tranSize;
            Map<Matrix<double, 1, NSTATES>, Aligned> eigen_partial_lh_child(&partial_lh_child[0]);
            Map<Matrix<double, 1, NSTATES>, Aligned> eigen_partial_lh_site(&partial_lh_site[0]);
            Map<Matrix<double, NSTATES, NSTATES>, Aligned> eigen_trans_state(&trans_state[0]);
            lh_ptn += (eigen_partial_lh_child * eigen_trans_state).dot(eigen_partial_lh_site);
        }
        if (ptn < orig_alnSize) {
			lh_ptn *= p_var_cat;
			if ((*aln)[ptn].is_const && (*aln)[ptn][0] < NSTATES) {
				lh_ptn += p_invar * state_freq[(int) (*aln)[ptn][0]];
			}
			lh_ptn = log(lh_ptn);
			tree_lh += lh_ptn * (aln->at(ptn).frequency);
			_pattern_lh[ptn] = lh_ptn;
			// BQM: pattern_lh contains the LOG-likelihood, not likelihood
        } else {
			lh_ptn = lh_ptn*p_var_cat + p_invar*state_freq[(int)model_factory->unobserved_ptns[ptn-orig_alnSize]];
			prob_const += lh_ptn;

        }
    }
    if (orig_alnSize < alnSize) {
    	// ascertainment bias correction
    	prob_const = log(1.0 - prob_const);
    	for (ptn = 0; ptn < orig_alnSize; ptn++)
    		_pattern_lh[ptn] -= prob_const;
    	tree_lh -= aln->getNSite()*prob_const;
    }

    if (pattern_lh) {
        memmove(pattern_lh, _pattern_lh, orig_alnSize * sizeof(double));
    }
    delete[] trans_mat_orig;
    return tree_lh;
}

template<int NSTATES>
void PhyloTree::computePartialLikelihoodSSE(PhyloNeighbor *dad_branch, PhyloNode *dad) {
    // don't recompute the likelihood
    if (dad_branch->partial_lh_computed & 1)
        return;
    Node *node = dad_branch->node;
    int ptn, cat;
    //double *trans_state;
    double *partial_lh_site;
    double *partial_lh_child;
    //double *partial_lh_block;
    //bool do_scale = true;
    //double freq;
    dad_branch->lh_scale_factor = 0.0;

    int numCat = site_rate->getNRate();
    int numStates = model->num_states;
    int tranSize = numStates * numStates;
    int alnSize = aln->size() + model_factory->unobserved_ptns.size();
    int orig_alnSize = aln->size();
    int block = numStates * numCat;
    size_t lh_size = alnSize * block;
    memset(dad_branch->scale_num, 0, alnSize * sizeof(UBYTE));

    if (node->isLeaf() && dad) {
        // external node
        memset(dad_branch->partial_lh, 0, lh_size * sizeof(double));
        //double *partial_lh_site;
        for (ptn = 0; ptn < alnSize; ++ptn) {
            char state;
            partial_lh_site = dad_branch->partial_lh + (ptn * block);

            if (node->name == ROOT_NAME) {
                state = aln->STATE_UNKNOWN;
            } else if (ptn < orig_alnSize){
                state = (aln->at(ptn))[node->id];
            } else {
            	state = model_factory->unobserved_ptns[ptn-orig_alnSize];
            }

            if (state == aln->STATE_UNKNOWN) {
#ifndef KEEP_GAP_LH
                dad_branch->scale_num[ptn] = -1;
#endif
                for (int state2 = 0; state2 < block; state2++) {
                    partial_lh_site[state2] = 1.0;
                }
            } else if (state < NSTATES) {
                double *_par_lh_site = partial_lh_site + state;
                for (cat = 0; cat < numCat; cat++) {
                    *_par_lh_site = 1.0;
                    _par_lh_site += NSTATES;
                }
            } else if (aln->seq_type == SEQ_DNA) {
                // ambiguous character, for DNA, RNA
                state = state - (NSTATES - 1);
                for (int state2 = 0; state2 < NSTATES; state2++)
                    if (state & (1 << state2)) {
                        for (cat = 0; cat < numCat; cat++)
                            partial_lh_site[cat * NSTATES + state2] = 1.0;
                    }
            } else if (aln->seq_type == SEQ_PROTEIN) {
                // ambiguous character, for DNA, RNA
                state = state - (NSTATES);
                assert(state < 2);
                int state_map[2] = {4+8,32+64};
                for (int state2 = 0; state2 <= 6; state2++)
                    if (state_map[(int)state] & (1 << state2)) {
                        for (cat = 0; cat < numCat; cat++)
                            partial_lh_site[cat * NSTATES + state2] = 1.0;
                    }
            } else {
            	outError("Internal error ", __func__);
            }

//            } else {
//                // ambiguous character, for DNA, RNA
//                state = state - (NSTATES - 1);
//                for (int state2 = 0; state2 < NSTATES && state2 <= 6; state2++)
//                    if (state & (1 << state2)) {
//                        double *_par_lh_site = partial_lh_site + state2;
//                        for (cat = 0; cat < numCat; cat++) {
//                            *_par_lh_site = 1.0;
//                            _par_lh_site += NSTATES;
//                        }
//                    }
//            }
        }
    } else {
        // internal node
        EIGEN_ALIGN16 double *trans_mat_orig = new double[numCat * tranSize + 2];
        double *trans_mat = trans_mat_orig;
        if (((intptr_t) trans_mat) % 16 != 0)
            trans_mat = trans_mat + 1;
        for (ptn = 0; ptn < lh_size; ++ptn)
            dad_branch->partial_lh[ptn] = 1.0;
#ifndef KEEP_GAP_LH
        for (ptn = 0; ptn < alnSize; ptn++)
            dad_branch->scale_num[ptn] = -1;
#endif
        FOR_NEIGHBOR_IT(node, dad, it)if ((*it)->node->name != ROOT_NAME) {
            computePartialLikelihoodSSE<NSTATES > ((PhyloNeighbor*) (*it), (PhyloNode*) node);
            dad_branch->lh_scale_factor += ((PhyloNeighbor*) (*it))->lh_scale_factor;
            for (cat = 0; cat < numCat; cat++) {
                model_factory->computeTransMatrix((*it)->length * site_rate->getRate(cat), &trans_mat[cat * tranSize]);
            }
            partial_lh_site = dad_branch->partial_lh;
            partial_lh_child = ((PhyloNeighbor*) (*it))->partial_lh;
            double sum_scale = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+: sum_scale) private(ptn, cat, partial_lh_site, partial_lh_child)
#endif
            for (ptn = 0; ptn < alnSize; ++ptn)
#ifndef KEEP_GAP_LH
            if (((PhyloNeighbor*) (*it))->scale_num[ptn] < 0) {
#ifndef _OPENMP
                partial_lh_site += NSTATES * numCat;
                partial_lh_child += NSTATES * numCat;
#endif
            } else
#endif
            {
#ifndef KEEP_GAP_LH
                if (dad_branch->scale_num[ptn] < 0)
                dad_branch->scale_num[ptn] = 0;
#endif
#ifdef _OPENMP
                int lh_offset = ptn*block;
                partial_lh_site = dad_branch->partial_lh + lh_offset;
                partial_lh_child = ((PhyloNeighbor*) (*it))->partial_lh + lh_offset;
#endif
                dad_branch->scale_num[ptn] += ((PhyloNeighbor*) (*it))->scale_num[ptn];
                double *partial_lh_block = partial_lh_site;
                double *trans_state = trans_mat;
                bool do_scale = true;
                for (cat = 0; cat < numCat; cat++)
                {
                    MappedRowVec(NSTATES) ei_partial_lh_child(partial_lh_child);
                    MappedRowVec(NSTATES) ei_partial_lh_site(partial_lh_site);
                    MappedMat(NSTATES) ei_trans_state(trans_state);
                    //ei_partial_lh_site.noalias() = (ei_partial_lh_child * ei_trans_state).cwiseProduct(ei_partial_lh_site);
                    ei_partial_lh_site.array() *= (ei_partial_lh_child * ei_trans_state).array();
                    partial_lh_site += NSTATES;
                    partial_lh_child += NSTATES;
                    trans_state += tranSize;
                }
                for (cat = 0; cat < block; cat++)
                if (partial_lh_block[cat] > SCALING_THRESHOLD) {
                    do_scale = false;
                    break;
                }
                if (do_scale) {
                    // unobserved const pattern will never have underflow
                    Map<VectorXd, Aligned> ei_lh_block(partial_lh_block, block);
                    ei_lh_block *= SCALING_THRESHOLD_INVER;
                    sum_scale += LOG_SCALING_THRESHOLD *  (*aln)[ptn].frequency;
                    dad_branch->scale_num[ptn] += 1;
//                    if (pattern_scale)
//                    pattern_scale[ptn] += LOG_SCALING_THRESHOLD;
                }
            }
            dad_branch->lh_scale_factor += sum_scale;
        }
        delete[] trans_mat_orig;
    }

    dad_branch->partial_lh_computed |= 1;
}

/****************************************************************************
 computing derivatives of likelihood function
 ****************************************************************************/
template<int NSTATES>
inline double PhyloTree::computeLikelihoodDervSSE(PhyloNeighbor *dad_branch, PhyloNode *dad, double &df, double &ddf) {
    PhyloNode *node = (PhyloNode*) dad_branch->node;
    PhyloNeighbor *node_branch = (PhyloNeighbor*) node->findNeighbor(dad);
    //assert(node_branch);
    // swap node and dad if node is a leaf
    if (node->isLeaf()) {
        PhyloNode *tmp_node = dad;
        dad = node;
        node = tmp_node;
        PhyloNeighbor *tmp_nei = dad_branch;
        dad_branch = node_branch;
        node_branch = tmp_nei;
    }
    if ((dad_branch->partial_lh_computed & 1) == 0)
        computePartialLikelihoodSSE<NSTATES>(dad_branch, dad);
    if ((node_branch->partial_lh_computed & 1) == 0)
        computePartialLikelihoodSSE<NSTATES>(node_branch, node);
    // now combine likelihood at the branch
    double tree_lh = node_branch->lh_scale_factor + dad_branch->lh_scale_factor;
    df = ddf = 0.0;
    int cat = 0;
    double *partial_lh_site = node_branch->partial_lh;
    double *partial_lh_child = dad_branch->partial_lh;
    double lh_ptn; // likelihood of the pattern
    double lh_ptn_derv1;
    double lh_ptn_derv2;
    double derv1_frac;
    double derv2_frac;
    double *trans_state;
    double *derv1_state;
    double *derv2_state;
    double p_invar = site_rate->getPInvar();

    int numCat = site_rate->getNRate();
    int numStates = model->num_states;
    int tranSize = numStates * numStates;
    int alnSize = aln->size() + model_factory->unobserved_ptns.size();
    int orig_alnSize = aln->size();

    double p_var_cat = (1.0 - p_invar) / (double) numCat;
    double state_freq[NSTATES];
    model->getStateFrequency(state_freq);
    double *trans_mat_orig  = new double[numCat * tranSize + 1];
    double *trans_derv1_orig  = new double[numCat * tranSize + 1];
    double *trans_derv2_orig  = new double[numCat * tranSize + 1];
    // make alignment 16
    double *trans_mat = trans_mat_orig, *trans_derv1 = trans_derv1_orig, *trans_derv2 = trans_derv2_orig;
    if (((intptr_t) trans_mat) % 16 != 0)
        trans_mat = trans_mat + 1;
    if (((intptr_t) trans_derv1) % 16 != 0)
        trans_derv1 = trans_derv1 + 1;
    if (((intptr_t) trans_derv2) % 16 != 0)
        trans_derv2 = trans_derv2 + 1;

    int discrete_cat = site_rate->getNDiscreteRate();
    if (!site_rate->isSiteSpecificRate())
        for (cat = 0; cat < discrete_cat; cat++) {
            double *trans_cat = trans_mat + (cat * tranSize);
            double *derv1_cat = trans_derv1 + (cat * tranSize);
            double *derv2_cat = trans_derv2 + (cat * tranSize);
            double rate_val = site_rate->getRate(cat);
            model_factory->computeTransDervFreq(dad_branch->length, rate_val, state_freq, trans_cat, derv1_cat,
                    derv2_cat);
        }
    int dad_state = aln->STATE_UNKNOWN;
    double my_df = 0.0;
    double my_ddf = 0.0;
    double prob_const = 0.0, prob_const_derv1 = 0.0, prob_const_derv2 = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+: tree_lh, my_df, my_ddf,prob_const, prob_const_derv1, prob_const_derv2) \
	private(cat, partial_lh_child, partial_lh_site,\
	lh_ptn, lh_ptn_derv1, lh_ptn_derv2, derv1_frac, derv2_frac, dad_state, trans_state, derv1_state, derv2_state)
#endif
    for (int ptn = 0; ptn < alnSize; ++ptn) {
#ifdef _OPENMP
        int lh_offset = ptn*numCat*numStates;
        partial_lh_site = node_branch->partial_lh + lh_offset;
        partial_lh_child = dad_branch->partial_lh + lh_offset;
#endif
        lh_ptn = 0.0;
        lh_ptn_derv1 = 0.0;
        lh_ptn_derv2 = 0.0;
        int padding = 0;
        dad_state = aln->STATE_UNKNOWN; // FOR TUNG: This is missing in your codes!
        if (dad->isLeaf()) {
        	if (ptn < orig_alnSize)
        		dad_state = (*aln)[ptn][dad->id];
        	else
        		dad_state = model_factory->unobserved_ptns[ptn-orig_alnSize];
        }
        padding = dad_state * NSTATES;
        if (dad_state < NSTATES) {
            //external node
            trans_state = trans_mat + padding;
            derv1_state = trans_derv1 + padding;
            derv2_state = trans_derv2 + padding;
            for (cat = 0; cat < numCat; cat++) {
                MappedVec(NSTATES)ei_partial_lh_child(partial_lh_child);
                MappedVec(NSTATES) ei_trans_state(trans_state);
                MappedVec(NSTATES) ei_derv1_state(derv1_state);
                MappedVec(NSTATES) ei_derv2_state(derv2_state);
                lh_ptn += ei_partial_lh_child.dot(ei_trans_state);
                lh_ptn_derv1 += ei_partial_lh_child.dot(ei_derv1_state);
                lh_ptn_derv2 += ei_partial_lh_child.dot(ei_derv2_state);
                partial_lh_child += NSTATES;
                partial_lh_site += NSTATES;
                trans_state += tranSize;
                derv1_state += tranSize;
                derv2_state += tranSize;
            }
        } else {
            // internal node, or external node but ambiguous character
            trans_state = trans_mat;
            derv1_state = trans_derv1;
            derv2_state = trans_derv2;
            for (cat = 0; cat < numCat; cat++) {
                MappedRowVec(NSTATES) ei_partial_lh_site(partial_lh_site);
                MappedRowVec(NSTATES) ei_partial_lh_child(partial_lh_child);
                MappedMat(NSTATES) ei_trans_state(trans_state);
                MappedMat(NSTATES) ei_derv1_state(derv1_state);
                MappedMat(NSTATES) ei_derv2_state(derv2_state);
                lh_ptn += (ei_partial_lh_child * ei_trans_state).dot(ei_partial_lh_site);
                lh_ptn_derv1 += (ei_partial_lh_child * ei_derv1_state).dot(ei_partial_lh_site);
                lh_ptn_derv2 += (ei_partial_lh_child * ei_derv2_state).dot(ei_partial_lh_site);
                partial_lh_site += NSTATES;
                partial_lh_child += NSTATES;
                trans_state += tranSize;
                derv1_state += tranSize;
                derv2_state += tranSize;
            }
        }
        if (ptn < orig_alnSize) {
			lh_ptn = lh_ptn * p_var_cat;
			if ((*aln)[ptn].is_const && (*aln)[ptn][0] < NSTATES) {
				lh_ptn += p_invar * state_freq[(int) (*aln)[ptn][0]];
			}
			double pad = p_var_cat / lh_ptn;
			if (std::isinf(pad)) {
				lh_ptn_derv1 *= p_var_cat;
				lh_ptn_derv2 *= p_var_cat;
				derv1_frac = lh_ptn_derv1 / lh_ptn;
				derv2_frac = lh_ptn_derv2 / lh_ptn;
			} else {
				derv1_frac = lh_ptn_derv1 * pad;
				derv2_frac = lh_ptn_derv2 * pad;
			}
	        double freq = aln->at(ptn).frequency;
			double tmp1 = derv1_frac * freq;
			double tmp2 = derv2_frac * freq;
			my_df += tmp1;
			my_ddf += tmp2 - tmp1 * derv1_frac;
			lh_ptn = log(lh_ptn);
			tree_lh += lh_ptn * freq;
			_pattern_lh[ptn] = lh_ptn;
        } else {
        	lh_ptn = lh_ptn*p_var_cat + p_invar*state_freq[(int)model_factory->unobserved_ptns[ptn-orig_alnSize]];
        	prob_const += lh_ptn;
        	prob_const_derv1 += lh_ptn_derv1 * p_var_cat;
        	prob_const_derv2 += lh_ptn_derv2 * p_var_cat;
        }
    }
    if (orig_alnSize < alnSize) {
    	// ascertainment bias correction
    	prob_const = 1.0 - prob_const;
    	derv1_frac = prob_const_derv1 / prob_const;
    	derv2_frac = prob_const_derv2 / prob_const;
    	int nsites = aln->getNSite();
    	my_df += nsites * derv1_frac;
    	my_ddf += nsites *(derv2_frac + derv1_frac*derv1_frac);
    	prob_const = log(prob_const);
    	tree_lh -= nsites * prob_const;
    	for (int ptn = 0; ptn < orig_alnSize; ptn++)
    		_pattern_lh[ptn] -= prob_const;
    }

    delete[] trans_derv2_orig;
    delete[] trans_derv1_orig;
    delete[] trans_mat_orig;
    df = my_df;
    ddf = my_ddf;
    return tree_lh;
}

/*******************************************************
 *
 * master function: wrapper for other optimized functions
 *
 ******************************************************/

void PhyloTree::computePartialLikelihood(PhyloNeighbor *dad_branch, PhyloNode *dad) {
	switch(aln->num_states) {
	case 4:
		switch(sse) {
		case LK_SSE: computePartialLikelihoodSSE<4>(dad_branch, dad); break;
		case LK_EIGEN: computePartialLikelihoodEigen<4>(dad_branch, dad); break;
        case LK_EIGEN_SSE: computePartialLikelihoodEigenTipSSE<hn::ScalableTag<double>, 4>(dad_branch, dad); break;
		case LK_NORMAL: computePartialLikelihoodNaive(dad_branch, dad); break;
		}
		break;
	case 20:
		switch(sse) {
		case LK_SSE: computePartialLikelihoodSSE<20>(dad_branch, dad); break;
		case LK_EIGEN: computePartialLikelihoodEigen<20>(dad_branch, dad); break;
        case LK_EIGEN_SSE: computePartialLikelihoodEigenTipSSE<hn::ScalableTag<double>, 20>(dad_branch, dad); break;
		case LK_NORMAL: computePartialLikelihoodNaive(dad_branch, dad); break;
		}
		break;
	case 2:
		switch(sse) {
		case LK_SSE: computePartialLikelihoodSSE<2>(dad_branch, dad); break;
		case LK_EIGEN: computePartialLikelihoodEigen<2>(dad_branch, dad); break;
		case LK_EIGEN_SSE:
			// use SSE code as current AVX-code does not work with 2-state model
			computePartialLikelihoodEigenTipSSE<hn::CappedTag<double, 2>, 2>(dad_branch, dad); break;
		case LK_NORMAL: computePartialLikelihoodNaive(dad_branch, dad); break;
		}
		break;

	default:
		computePartialLikelihoodNaive(dad_branch, dad); break;
	}
}

double PhyloTree::computeLikelihoodBranch(PhyloNeighbor *dad_branch, PhyloNode *dad, double *pattern_lh) {
	switch(aln->num_states) {
	case 4:
		switch(sse) {
		case LK_SSE: return computeLikelihoodBranchSSE<4>(dad_branch, dad, pattern_lh);
		case LK_EIGEN: return computeLikelihoodBranchEigen<4>(dad_branch, dad, pattern_lh);
        case LK_EIGEN_SSE: return computeLikelihoodBranchEigenTipSSE<hn::ScalableTag<double>, 4>(dad_branch, dad, pattern_lh);
		case LK_NORMAL: return computeLikelihoodBranchNaive(dad_branch, dad, pattern_lh);
		}
		break;
	case 20:
		switch(sse) {
		case LK_SSE: return computeLikelihoodBranchSSE<20>(dad_branch, dad, pattern_lh);
		case LK_EIGEN: return computeLikelihoodBranchEigen<20>(dad_branch, dad, pattern_lh);
        case LK_EIGEN_SSE: return computeLikelihoodBranchEigenTipSSE<hn::ScalableTag<double>, 20>(dad_branch, dad, pattern_lh);
		case LK_NORMAL: return computeLikelihoodBranchNaive(dad_branch, dad, pattern_lh);
		}
		break;
	case 2:
		switch(sse) {
		case LK_SSE: return computeLikelihoodBranchSSE<2>(dad_branch, dad, pattern_lh);
		case LK_EIGEN: return computeLikelihoodBranchEigen<2>(dad_branch, dad, pattern_lh);
		case LK_EIGEN_SSE:
		// use SSE code as current AVX-code does not work with  2-state model
			return computeLikelihoodBranchEigenTipSSE<hn::CappedTag<double, 2>, 2>(dad_branch, dad, pattern_lh);
		case LK_NORMAL: return computeLikelihoodBranchNaive(dad_branch, dad, pattern_lh);
		}
		break;

	default:
		return computeLikelihoodBranchNaive(dad_branch, dad, pattern_lh);
	}
	return 0.0;
}

/*
 * This function is called millions times. So it is not a good idea to
 * have a if and switch here.
 */
double PhyloTree::computeLikelihoodDerv(PhyloNeighbor *dad_branch, PhyloNode *dad, double &df, double &ddf) {
    switch (aln->num_states) {
    case 4:
    	switch(sse) {
    	case LK_SSE: return computeLikelihoodDervSSE<4>(dad_branch, dad, df, ddf);
    	case LK_EIGEN: return computeLikelihoodDervEigen<4>(dad_branch, dad, df, ddf);
        case LK_EIGEN_SSE: return computeLikelihoodDervEigenTipSSE<hn::ScalableTag<double>, 4>(dad_branch, dad, df, ddf);
    	case LK_NORMAL: return computeLikelihoodDervNaive(dad_branch, dad, df, ddf);
    	}
    	break;
	case 20:
		switch(sse) {
		case LK_SSE: return computeLikelihoodDervSSE<20>(dad_branch, dad, df, ddf);
		case LK_EIGEN: return computeLikelihoodDervEigen<20>(dad_branch, dad, df, ddf);
		case LK_EIGEN_SSE: return computeLikelihoodDervEigenTipSSE<hn::ScalableTag<double>, 20>(dad_branch, dad, df, ddf);
		case LK_NORMAL: return computeLikelihoodDervNaive(dad_branch, dad, df, ddf);
		}
		break;
	case 2:
		switch(sse) {
		case LK_SSE: return computeLikelihoodDervSSE<2>(dad_branch, dad, df, ddf);
		case LK_EIGEN: return computeLikelihoodDervEigen<2>(dad_branch, dad, df, ddf);
		case LK_EIGEN_SSE:
		// use SSE code as current AVX-code does not work with  2-state model
			return computeLikelihoodDervEigenTipSSE<hn::CappedTag<double, 2>, 2>(dad_branch, dad, df, ddf);
		case LK_NORMAL: return computeLikelihoodDervNaive(dad_branch, dad, df, ddf);
		}
		break;
	default:
		return computeLikelihoodDervNaive(dad_branch, dad, df, ddf);

    }
    return 0.0;
}

