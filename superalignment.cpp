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

#include <stdarg.h>
#include "phylotree.h"
#include "superalignment.h"
#include "phylosupertree.h"

SuperAlignment::SuperAlignment()
 : Alignment() {}

SuperAlignment::SuperAlignment(PhyloSuperTree *super_tree)
 : Alignment()
{
	// first build taxa_index and partitions
	int site, seq, nsite = super_tree->size();
	PhyloSuperTree::iterator it;
	for (site = 0, it = super_tree->begin(); it != super_tree->end(); it++, site++) {
		partitions.push_back((*it)->aln);
		int nseq = (*it)->aln->getNSeq();
		//cout << "nseq  = " << nseq << endl;
		for (seq = 0; seq < nseq; seq++) {
			int id = getSeqID((*it)->aln->getSeqName(seq));
			if (id < 0) {
				seq_names.push_back((*it)->aln->getSeqName(seq));
				id = seq_names.size()-1;
				IntVector vec(nsite, -1);
				vec[site] = seq;
				taxa_index.push_back(vec);
			} else
				taxa_index[id][site] = seq;
		}
	}
	// now the patterns of sequence-genes presence/absence
	buildPattern();
}

void SuperAlignment::buildPattern() {
	int site, seq, nsite = partitions.size();

	seq_type = SEQ_BINARY;
	num_states = 2; // binary type because the super alignment presents the presence/absence of taxa in the partitions
	STATE_UNKNOWN = 2;
	site_pattern.resize(nsite, -1);
	clear();
	pattern_index.clear();
	VerboseMode save_mode = verbose_mode; 
	verbose_mode = min(verbose_mode, VB_MIN); // to avoid printing gappy sites in addPattern
	int nseq = getNSeq();
	for (site = 0; site < nsite; site++) {
 		Pattern pat;
 		pat.append(nseq, 0);
		for (seq = 0; seq < nseq; seq++)
			pat[seq] = (taxa_index[seq][site] >= 0)? 1 : 0;
		addPattern(pat, site);
	}
	verbose_mode = save_mode;
	countConstSite();
    countInformative();
    buildSeqStates();
}



void SuperAlignment::linkSubAlignment(int part) {
	assert(taxa_index.size() == getNSeq());
	int nseq = getNSeq(), seq;
	vector<bool> checked;
	checked.resize(partitions[part]->getNSeq(), false);
	for (seq = 0; seq < nseq; seq++) {
		int id = partitions[part]->getSeqID(getSeqName(seq));
		if (id < 0)
			taxa_index[seq][part] = -1;
		else {
			taxa_index[seq][part] = id;
			checked[id] = true;
		}
	}
	if (verbose_mode >= VB_MED) {

	}
	// sanity check that all seqnames in partition must be present in superalignment
	for (seq = 0; seq < checked.size(); seq++) {
		assert(checked[seq]);
	}
}

void SuperAlignment::extractSubAlignment(Alignment *aln, IntVector &seq_id, int min_true_char) {
	assert(aln->isSuperAlignment());
	SuperAlignment *saln = (SuperAlignment*)aln;

    IntVector::iterator it;
    for (it = seq_id.begin(); it != seq_id.end(); it++) {
        assert(*it >= 0 && *it < aln->getNSeq());
        seq_names.push_back(aln->getSeqName(*it));
    }

	// BUG HERE!
	//Alignment::extractSubAlignment(aln, seq_id, 0);

	taxa_index.resize(getNSeq());
	for (int i = 0; i < getNSeq(); i++)
		taxa_index[i].resize(saln->partitions.size(), -1);

	int part = 0;
	partitions.resize(saln->partitions.size());
	for (vector<Alignment*>::iterator ait = saln->partitions.begin(); ait != saln->partitions.end(); ait++, part++) {
		IntVector sub_seq_id;
		for (IntVector::iterator it = seq_id.begin(); it != seq_id.end(); it++)
			if (saln->taxa_index[*it][part] >= 0)
				sub_seq_id.push_back(saln->taxa_index[*it][part]);
		Alignment *subaln = new Alignment;
		subaln->extractSubAlignment(*ait, sub_seq_id, 0);
		partitions[part] = subaln;
		linkSubAlignment(part);
//		cout << subaln->getNSeq() << endl;
//		subaln->printPhylip(cout);
	}

	// now build the patterns based on taxa_index
	buildPattern();
}

Alignment *SuperAlignment::removeIdenticalSeq(string not_remove, bool keep_two, StrVector &removed_seqs, StrVector &target_seqs) {
    IntVector checked;
    vector<bool> removed;
    checked.resize(getNSeq(), 0);
    removed.resize(getNSeq(), false);
    int seq1;

	for (seq1 = 0; seq1 < getNSeq(); seq1++) {
        if (checked[seq1]) continue;
        bool first_ident_seq = true;
		for (int seq2 = seq1+1; seq2 < getNSeq(); seq2++) {
			if (getSeqName(seq2) == not_remove) continue;
			bool equal_seq = true;
			int part = 0;
			// check if seq1 and seq2 are identical over all partitions
			for (vector<Alignment*>::iterator ait = partitions.begin(); ait != partitions.end(); ait++, part++) {
				int subseq1 = taxa_index[seq1][part];
				int subseq2 = taxa_index[seq2][part];
				if (subseq1 < 0 && subseq2 < 0) // continue if both seqs are absent in this partition
					continue;
				if (subseq1 < 0 && subseq2 > 0) {
					// if one sequence is present and the other is absent for a gene, we conclude that they are not identical
					equal_seq = false;
					break;
				}
				if (subseq1 > 0 && subseq2 < 0) {
					// if one sequence is present and the other is absent for a gene, we conclude that they are not identical
					equal_seq = false;
					break;
				}
				// now if both seqs are present, check sequence content
				for (iterator it = (*ait)->begin(); it != (*ait)->end(); it++)
					if  ((*it)[subseq1] != (*it)[subseq2]) {
						equal_seq = false;
						break;
					}
			}
			if (equal_seq) {
				if (!keep_two || !first_ident_seq) {
					removed_seqs.push_back(getSeqName(seq2));
					target_seqs.push_back(getSeqName(seq1));
					removed[seq2] = true;
				}
				checked[seq2] = 1;
				first_ident_seq = false;
			}
		}
		checked[seq1] = 1;
	}

	if (removed_seqs.empty()) return this; // do nothing if the list is empty

	if (removed_seqs.size() > getNSeq()-3)
		outError("Your alignment contains too many identical sequences, quiting now...");

	// now remove identical sequences
	IntVector keep_seqs;
	for (seq1 = 0; seq1 < getNSeq(); seq1++)
		if (!removed[seq1]) keep_seqs.push_back(seq1);
	SuperAlignment *aln;
	aln = new SuperAlignment;
	aln->extractSubAlignment(this, keep_seqs, 0);
	return aln;
}

/*
void SuperAlignment::checkGappySeq() {
	int nseq = getNSeq(), part = 0, i;
	IntVector gap_only_seq;
	gap_only_seq.resize(nseq, 1);
	//cout << "Checking gaps..." << endl;
	for (vector<Alignment*>::iterator it = partitions.begin(); it != partitions.end(); it++, part++) {
		IntVector keep_seqs;
		for (i = 0; i < nseq; i++)
			if (taxa_index[i][part] >= 0)
			if (!(*it)->isGapOnlySeq(taxa_index[i][part])) {
				keep_seqs.push_back(taxa_index[i][part]);
				gap_only_seq[i] = 0;
			}
		if (keep_seqs.size() < (*it)->getNSeq()) {
			cout << "Discard " << (*it)->getNSeq() - keep_seqs.size() 
				 << " sequences from partition number " << part+1 << endl;
			Alignment *aln = new Alignment;
			aln->extractSubAlignment((*it), keep_seqs, 0);
			delete (*it);
			(*it) = aln;
			linkSubAlignment(part);
		}
		cout << __func__ << " num_states = " << (*it)->num_states << endl;
	}
	int wrong_seq = 0;
	for (i = 0; i < nseq; i++)
		if (gap_only_seq[i]) {
			cout << "ERROR: Sequence " << getSeqName(i) << " contains only gaps or missing data" << endl;
			wrong_seq++;
		}
	if (wrong_seq) {
		outError("Some sequences (see above) are problematic, please check your alignment again");
		}
}
*/
void SuperAlignment::getSitePatternIndex(IntVector &pattern_index) {
	int nptn = 0;
	for (vector<Alignment*>::iterator it = partitions.begin(); it != partitions.end(); it++) {
		int nsite = pattern_index.size();
		pattern_index.insert(pattern_index.end(), (*it)->site_pattern.begin(), (*it)->site_pattern.end());
		for (int i = nsite; i < pattern_index.size(); i++)
			pattern_index[i] += nptn;
		nptn += (*it)->getNPattern();
	}
}

void SuperAlignment::getPatternFreq(IntVector &pattern_freq) {
	if (!isSuperAlignment()) outError("Internal error: ", __func__);
	int offset = 0;
	if (!pattern_freq.empty()) pattern_freq.resize(0);
	for (vector<Alignment*>::iterator it = partitions.begin(); it != partitions.end(); it++) {
		IntVector freq;
		(*it)->getPatternFreq(freq);
		pattern_freq.insert(pattern_freq.end(), freq.begin(), freq.end());
		offset += freq.size();
	}
}

void SuperAlignment::createBootstrapAlignment(Alignment *aln, IntVector* pattern_freq, const char *spec) {
	if (!aln->isSuperAlignment()) outError("Internal error: ", __func__);
	if (pattern_freq) outError("Unsupported yet.", __func__);
	if (spec) outError("Unsupported yet.", __func__);
	Alignment::copyAlignment(aln);
	SuperAlignment *super_aln = (SuperAlignment*) aln;
	if (!partitions.empty()) outError("Internal error: ", __func__);
	for (vector<Alignment*>::iterator it = super_aln->partitions.begin(); it != super_aln->partitions.end(); it++) {
		Alignment *boot_aln = new Alignment;
		boot_aln->createBootstrapAlignment(*it);
		partitions.push_back(boot_aln);
	}
	taxa_index = super_aln->taxa_index;
}

void SuperAlignment::createBootstrapAlignment(IntVector &pattern_freq, const char *spec) {
	if (!isSuperAlignment()) outError("Internal error: ", __func__);
	int nptn = 0;
	for (vector<Alignment*>::iterator it = partitions.begin(); it != partitions.end(); it++) {
		nptn += (*it)->getNPattern();
	}
	pattern_freq.resize(0);
	int *internal_freq = new int[nptn];
	createBootstrapAlignment(internal_freq, spec);
	pattern_freq.insert(pattern_freq.end(), internal_freq, internal_freq + nptn);
	delete [] internal_freq;

/*	if (spec && strncmp(spec, "GENE", 4) != 0) outError("Unsupported yet.", __func__);

	int offset = 0;
	if (!pattern_freq.empty()) pattern_freq.resize(0);

	if (spec && strncmp(spec, "GENE", 4) == 0) {
		// resampling whole genes
		int nptn = 0;
		IntVector part_pos;
		for (vector<Alignment*>::iterator it = partitions.begin(); it != partitions.end(); it++) {
			part_pos.push_back(nptn);
			nptn += (*it)->getNPattern();
		}
		pattern_freq.resize(nptn, 0);
		for (int i = 0; i < partitions.size(); i++) {
			int part = random_int(partitions.size());
			for (int j = 0; j < partitions[part]->getNPattern(); j++)
				pattern_freq[j + part_pos[part]] += partitions[part]->at(j).frequency;
		}
	} else {
		// resampling sites within genes
		for (vector<Alignment*>::iterator it = partitions.begin(); it != partitions.end(); it++) {
			IntVector freq;
			(*it)->createBootstrapAlignment(freq);
			pattern_freq.insert(pattern_freq.end(), freq.begin(), freq.end());
			offset += freq.size();
		}
	}*/
}


void SuperAlignment::createBootstrapAlignment(int *pattern_freq, const char *spec) {
	if (!isSuperAlignment()) outError("Internal error: ", __func__);
	if (spec && strncmp(spec, "GENE", 4) != 0) outError("Unsupported yet. ", __func__);

	if (spec && strncmp(spec, "GENE", 4) == 0) {
		// resampling whole genes
		int nptn = 0;
		IntVector part_pos;
		for (vector<Alignment*>::iterator it = partitions.begin(); it != partitions.end(); it++) {
			part_pos.push_back(nptn);
			nptn += (*it)->getNPattern();
		}
		memset(pattern_freq, 0, nptn * sizeof(int));
		for (int i = 0; i < partitions.size(); i++) {
			int part = random_int(partitions.size());
			Alignment *aln = partitions[part];
			if (strncmp(spec,"GENESITE",8) == 0) {
				// then resampling sites in resampled gene
				for (int j = 0; j < aln->getNSite(); j++) {
					int ptn_id = aln->getPatternID(random_int(aln->getNPattern()));
					pattern_freq[ptn_id + part_pos[part]]++;
				}

			} else {
				for (int j = 0; j < aln->getNPattern(); j++)
					pattern_freq[j + part_pos[part]] += aln->at(j).frequency;
			}
		}
	} else {
		// resampling sites within genes
		int offset = 0;
		for (vector<Alignment*>::iterator it = partitions.begin(); it != partitions.end(); it++) {
			(*it)->createBootstrapAlignment(pattern_freq + offset);
			offset += (*it)->getNPattern();
		}
	}
}

/**
 * shuffle alignment by randomizing the order of sites
 */
void SuperAlignment::shuffleAlignment() {
	if (!isSuperAlignment()) outError("Internal error: ", __func__);
	for (vector<Alignment*>::iterator it = partitions.begin(); it != partitions.end(); it++) {
		(*it)->shuffleAlignment();
	}
}


double SuperAlignment::computeObsDist(int seq1, int seq2) {
	int site;
	int diff_pos = 0, total_pos = 0;
	for (site = 0; site < getNSite(); site++) {
		int id1 = taxa_index[seq1][site];
		int id2 = taxa_index[seq2][site];
		if (id1 < 0 || id2 < 0) continue;
		int num_states = partitions[site]->num_states;
		for (Alignment::iterator it = partitions[site]->begin(); it != partitions[site]->end(); it++) 
			if  ((*it)[id1] < num_states && (*it)[id2] < num_states) {
				total_pos += (*it).frequency;
				if ((*it)[id1] != (*it)[id2] )
					diff_pos += (*it).frequency;
			}
	}
	if (!total_pos) 
		return MAX_GENETIC_DIST; // return +INF if no overlap between two sequences
	return ((double)diff_pos) / total_pos;
}


double SuperAlignment::computeDist(int seq1, int seq2) {
	if (partitions.empty()) return 0.0;
	double obs_dist = computeObsDist(seq1, seq2);
    int num_states = partitions[0]->num_states;
    double z = (double)num_states / (num_states-1);
    double x = 1.0 - (z * obs_dist);

    if (x <= 0) {
        /*		string str = "Too long distance between two sequences ";
        		str += getSeqName(seq1);
        		str += " and ";
        		str += getSeqName(seq2);
        		outWarning(str);*/
        return MAX_GENETIC_DIST;
    }

    return -log(x) / z;
    //return computeObsDist(seq1, seq2);
	//  AVERAGE DISTANCE

	double dist = 0;
	int part = 0, num = 0;
	for (vector<Alignment*>::iterator it = partitions.begin(); it != partitions.end(); it++, part++) {
		int id1 = taxa_index[seq1][part];
		int id2 = taxa_index[seq2][part];
		if (id1 < 0 || id2 < 0) continue;
		dist += (*it)->computeDist(id1, id2);
	}
	if (num == 0) // two sequences are not overlapping at all!
		return MAX_GENETIC_DIST;
	return dist / num;
}

SuperAlignment::~SuperAlignment()
{
	for (vector<Alignment*>::reverse_iterator it = partitions.rbegin(); it != partitions.rend(); it++)
		delete (*it);
	partitions.clear();
}

void SuperAlignment::printCombinedAlignment(ostream &out, bool append) {
	vector<Alignment*>::iterator pit;
	int final_length = 0;
	for (pit = partitions.begin(); pit != partitions.end(); pit++)
		final_length += (*pit)->getNSite();

	out << getNSeq() << " " << final_length << endl;
	StrVector::iterator it;
	int max_len = getMaxSeqNameLength();
	if (max_len < 10) max_len = 10;
	int seq_id = 0;
	for (it = seq_names.begin(); it != seq_names.end(); it++, seq_id++) {
		out.width(max_len);
		out << left << (*it) << " ";
		int part = 0;
		for (pit = partitions.begin(); pit != partitions.end(); pit++, part++) {
			int part_seq_id = taxa_index[seq_id][part];
			int nsite = (*pit)->getNSite();
			if (part_seq_id >= 0) {
				for (int i = 0; i < nsite; i++)
					out << (*pit)->convertStateBackStr((*pit)->getPattern(i) [part_seq_id]);
			} else {
				string str(nsite, '?');
				out << str;
			}
		}
		out << endl;
	}
}

void SuperAlignment::printCombinedAlignment(const char *file_name, bool append) {
	vector<Alignment*>::iterator pit;
	int final_length = 0;
	for (pit = partitions.begin(); pit != partitions.end(); pit++)
		final_length += (*pit)->getNSite();
	try {
		ofstream out;
		out.exceptions(ios::failbit | ios::badbit);

		if (append)
			out.open(file_name, ios_base::out | ios_base::app);
		else
			out.open(file_name);
		out << getNSeq() << " " << final_length << endl;
		StrVector::iterator it;
		int max_len = getMaxSeqNameLength();
		if (max_len < 10) max_len = 10;
		int seq_id = 0;
		for (it = seq_names.begin(); it != seq_names.end(); it++, seq_id++) {
			out.width(max_len);
			out << left << (*it) << " ";
			int part = 0;
			for (pit = partitions.begin(); pit != partitions.end(); pit++, part++) {
				int part_seq_id = taxa_index[seq_id][part];
				int nsite = (*pit)->getNSite();
				if (part_seq_id >= 0) {
					for (int i = 0; i < nsite; i++)
						out << (*pit)->convertStateBackStr((*pit)->getPattern(i) [part_seq_id]);
				} else {
					string str(nsite, '?');
					out << str;
				}
			}
			out << endl;
		}
		out.close();
		cout << "Concatenated alignment was printed to " << file_name << endl;
	} catch (ios::failure) {
		outError(ERR_WRITE_OUTPUT, file_name);
	}	
}

void SuperAlignment::printSubAlignments(Params &params, vector<PartitionInfo> &part_info) {
	vector<Alignment*>::iterator pit;
	string filename;
	int part;
	assert(part_info.size() == partitions.size());
	for (pit = partitions.begin(), part = 0; pit != partitions.end(); pit++, part++) {
		if (params.aln_output)
			filename = params.aln_output;
		else
			filename = params.out_prefix;
		filename += "." + part_info[part].name;
		 if (params.aln_output_format == ALN_PHYLIP)
			(*pit)->printPhylip(filename.c_str(), false, NULL, params.aln_nogaps, false, NULL);
		else if (params.aln_output_format == ALN_FASTA)
			(*pit)->printFasta(filename.c_str(), false, NULL, params.aln_nogaps, false, NULL);
	}
}

double SuperAlignment::computeUnconstrainedLogL() {
	double logl = 0.0;
	vector<Alignment*>::iterator pit;
	for (pit = partitions.begin(); pit != partitions.end(); pit++)
		logl += (*pit)->computeUnconstrainedLogL();
	return logl;
}

double SuperAlignment::computeMissingData() {
	double ret = 0.0;
	int len = 0;
	vector<Alignment*>::iterator pit;
	for (pit = partitions.begin(); pit != partitions.end(); pit++) {
		ret += (*pit)->getNSeq() * (*pit)->getNSite();
		len += (*pit)->getNSite();
	}
	ret /= getNSeq() * len;
	return 1.0 - ret;

}

Alignment *SuperAlignment::concatenateAlignments(IntVector &ids) {
	string union_taxa;
	int nsites = 0, nstates = 0, i;
	SeqType sub_type = SEQ_UNKNOWN;
	for (i = 0; i < ids.size(); i++) {
		int id = ids[i];
		if (id < 0 || id >= partitions.size())
			outError("Internal error ", __func__);
		if (nstates == 0) nstates = partitions[id]->num_states;
		if (sub_type == SEQ_UNKNOWN) sub_type = partitions[id]->seq_type;
		if (sub_type != partitions[id]->seq_type)
			outError("Cannot concatenate sub-alignments of different type");
		if (nstates != partitions[id]->num_states)
			outError("Cannot concatenate sub-alignments of different #states");

		string taxa_set = getPattern(id);
		nsites += partitions[id]->getNSite();
		if (i == 0) union_taxa = taxa_set; else {
			for (int j = 0; j < union_taxa.length(); j++)
				if (taxa_set[j] == 1) union_taxa[j] = 1;
		}
	}

	Alignment *aln = new Alignment;
	for (i = 0; i < union_taxa.length(); i++)
		if (union_taxa[i] == 1) {
			aln->seq_names.push_back(getSeqName(i));
		}
	aln->num_states = nstates;
	aln->seq_type = sub_type;
	aln->site_pattern.resize(nsites, -1);
    aln->clear();
    aln->pattern_index.clear();
    aln->STATE_UNKNOWN = partitions[ids[0]]->STATE_UNKNOWN;

    int site = 0;
    for (i = 0; i < ids.size(); i++) {
    	int id = ids[i];
		string taxa_set = getPattern(id);
    	for (Alignment::iterator it = partitions[id]->begin(); it != partitions[id]->end(); it++) {
    		Pattern pat;
    		int part_seq = 0;
    		for (int seq = 0; seq < union_taxa.size(); seq++)
    			if (union_taxa[seq] == 1) {
    				char ch = aln->STATE_UNKNOWN;
    				if (taxa_set[seq] == 1) {
    					ch = (*it)[part_seq++];
    				}
    				pat.push_back(ch);
    			}
    		assert(part_seq == partitions[id]->getNSeq());
    		aln->addPattern(pat, site, (*it).frequency);
    		// IMPORTANT BUG FIX FOLLOW
    		int ptnindex = aln->pattern_index[pat];
            for (int j = 0; j < (*it).frequency; j++)
                aln->site_pattern[site++] = ptnindex;

    	}
    }
    aln->countConstSite();
    aln->countInformative();
    aln->buildSeqStates();

	return aln;
}
