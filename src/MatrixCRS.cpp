/**
* \file src/MatrixCRS.cpp
* \brief Implementation of a compressed row storage datastructure for sparse
*        matrices.
* 
* Copyright 2008, 2009 Tobias Preclik
* 
* This file is part of amgpp.
* 
* Amgpp is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* Amgpp is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with amgpp.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "MatrixCRS.h"
#include <omp.h>


bool IndexValuePair::operator<(const IndexValuePair& ivp) const {
	return index_ < ivp.index_;
}


/**
* \brief Constructor for a matrix of size \f$ M \times N \f$.
*
* \param m The number of rows of the matrix.
* \param n The number of columns of the matrix.
* \param capacities The number of elements which should be reserved for each row.
*
* The matrix is initialized to the zero matrix.
*/
MatrixCRS::MatrixCRS(std::size_t m, std::size_t n, const std::vector<std::size_t>& capacities) : cols_(n), values_(), row_indices_(m + 1), row_lengths_(m, 0) {
	assert(capacities.size() == m && "The number of rows does not match the size of the capacity vector.");

	std::size_t cap = 0;
	row_indices_[0] = 0;
	for (std::size_t i = 0; i < m; ++i) {
		row_indices_[i+1] = row_indices_[i] + capacities[i];
		cap += capacities[i]; 
	}

	values_.resize(cap);
}


/**
* \brief Tests if the two matrices are semantically equal.
*
* \return Returns true if the two matrices are semantically equal, otherwise false.
*/
bool MatrixCRS::operator==(const MatrixCRS& rhs) const {
	if (rows() != rhs.rows() || columns() != rhs.columns())
	return false;

	const MatrixCRS& lhs(*this);

	for (std::size_t i = 0; i < rows(); ++i) {
		std::size_t j_lhs(lhs.begin(i)), j_rhs(rhs.begin(i));
		std::size_t j_lhs_end(lhs.end(i)), j_rhs_end(rhs.end(i));

		while (j_lhs < j_lhs_end && j_rhs < j_rhs_end) {
			if (lhs[j_lhs].index_ < rhs[j_rhs].index_) {
				if (lhs[j_lhs++].value_ != 0.0)
				return false;
			}
			else if (lhs[j_lhs].index_ > rhs[j_rhs].index_) {
				if (rhs[j_rhs++].value_ != 0.0)
				return false;
			}
			else {
				if (lhs[j_lhs++].value_ != rhs[j_rhs++].value_)
				return false;
			}
		}

		while (j_lhs < j_lhs_end) {
			if (lhs[j_lhs++].value_ != 0.0)
			return false;
		}

		while (j_rhs < j_rhs_end) {
			if (rhs[j_rhs++].value_ != 0.0)
			return false;
		}
	}

	return true;
}


/**
* \brief Reserves at least capacity elements for the m-th row.
*
* \param m The index of the row.
* \param cap The minimum target capacity of the row.
*/
void MatrixCRS::reserveRowElements(std::size_t m, std::size_t cap) {
	if (capacity(m) >= cap)
	return;

	std::size_t additional = cap - capacity(m);

	// enlarge entry storage if necessary
	if (capacity() - row_indices_[rows()] < additional) {
		values_.resize(row_indices_[rows()] + additional);
	}

	// move successive rows
	std::copy_backward(&values_[row_indices_[m+1]], &values_[row_indices_[rows()]], &values_[row_indices_[rows()] + additional]);
	for (std::size_t i = m+1; i <= rows(); ++i)
	row_indices_[i] += additional;
}


/**
* \brief Frees reserved row elements for the m-th row.
* 
* \param m The index of the row.
* \param limit The maximum number of elements freed, by default INT_MAX.
* 
* At most limit unused reserved row elements from the m-th row will be transfered to the
* next row. This will take constant time unless the next row is already filled. In this case
* the process needs time in the order of the number of entries in the next row.
*/ 
void MatrixCRS::freeReservedRowElements(std::size_t m, std::size_t limit) {
	std::size_t additional = std::min(capacity(m) - nonzeros(m), limit);
	if (additional == 0)
	return;
	
	// shrink reserved storage
	std::copy(&values_[row_indices_[m+1]], &values_[row_indices_[m+1] + row_lengths_[m+1]], &values_[row_indices_[m+1] - additional]);
	row_indices_[m+1] -= additional;
}


/**
* \brief Appends an element to the m-th row.
*
* \param m The absolute index of the entry's row.
* \param n The absolute index of the entry's column.
* \param value The value of the entry.
*
* Appends an element to the m-th row under the condition that the column index is larger than
* the index of the previous element in the row. Furthermore there must be enough reserved space
* in the row. This allows an insertion in constant time.
*/
void MatrixCRS::appendRowElement(std::size_t m, std::size_t n, double value) {
	assert(end(m) < begin(m+1) && "Not enough reserved space in the current row.");
	assert((nonzeros(m) == 0 || n > (*this)(m, nonzeros(m)-1).index_) && "Index is not strictly increasing.");

	values_[end(m)].value_ = value;
	values_[end(m)].index_ = n;
	++row_lengths_[m];
}


/**
* \brief Inserts an element into the m-th row.
*
* \param m The absolute index of the entry's row.
* \param n The absolute index of the entry's column.
* \param value The value of the entry.
*
* Inserts an element to the m-th row and reallocates space if necessary.
*/
void MatrixCRS::insertRowElement(std::size_t m, std::size_t n, double value) {
	if (end(m) >= begin(m+1))
	reserveRowElements(m, capacity(m) + 1);

	std::size_t j = end(m);
	++row_lengths_[m];

	values_[j].value_ = value;
	values_[j].index_ = n;

	while (j != begin(m) && values_[j].index_ < values_[j-1].index_) {
		std::swap(values_[j], values_[j-1]);
		--j;
	}

	assert((j == begin(m) || values_[j-1].index_ < values_[j].index_) && "Double entry detected.");
}


/**
* \brief Calculation of the transpose of the matrix.
*
* \return The transpose of the matrix.
*/
const MatrixCRS MatrixCRS::getTranspose() const {
	// count the number of entries per column
	std::vector<std::size_t> column_lengths(columns(), 0);
	for (std::size_t i = 0; i < rows(); ++i)
	for (std::size_t j = begin(i); j < end(i); ++j)
	++column_lengths[values_[j].index_];

	// setup tranpose and reserve the correct number of entries per row
	MatrixCRS tmp(columns(), rows(), column_lengths);

	// append elements to rows of transpose
	for (std::size_t i = 0; i < rows(); ++i)
	for (std::size_t j = begin(i); j < end(i); ++j)
	tmp.appendRowElement(values_[j].index_, i, values_[j].value_);

	return tmp;
}


/**
* \brief Addition operator for the addition of two matrices (\f$ A=B+C \f$).
*
* \param lhs The left-hand side matrix for the matrix addition.
* \param rhs The right-hand side matrix to be added to the left-hand side matrix.
* \return The sum of the two matrices.
*/
const MatrixCRS operator+(const MatrixCRS& lhs, const MatrixCRS& rhs) {
	assert((lhs.rows() == rhs.rows() && rhs.columns() == rhs.columns()) && "Matrix sizes do not match");

	// analyze matrices to predict the required row storage capacities
	std::vector<std::size_t> capacities(lhs.rows(), 0);
	for (std::size_t i = 0; i < lhs.rows(); ++i) {
		std::size_t j_l(0), j_r(0);
		std::size_t accu(lhs.nonzeros(i) + rhs.nonzeros(i));

		while (j_l < lhs.nonzeros(i) && j_r < rhs.nonzeros(i)) {
			if (lhs(i, j_l).index_ < rhs(i, j_r).index_)
			++j_l;
			else if (lhs(i, j_l).index_ > rhs(i, j_r).index_)
			++j_r;
			else {
				--accu; ++j_l; ++j_r;
			}
		}

		capacities[i] = accu;
	}

	// merge matrices 
	MatrixCRS tmp(lhs.rows(), lhs.columns(), capacities);
	for (std::size_t i = 0; i < lhs.rows(); ++i) {
		std::size_t j_l(0), j_r(0);

		while (j_l < lhs.nonzeros(i) && j_r < rhs.nonzeros(i)) {
			if (lhs(i, j_l).index_ < rhs(i, j_r).index_) {
				tmp.appendRowElement(i, lhs(i, j_l).index_, lhs(i, j_l).value_);
				++j_l;
			}
			else if (lhs(i, j_l).index_ > rhs(i, j_r).index_) {
				tmp.appendRowElement(i, rhs(i, j_r).index_, rhs(i, j_r).value_);
				++j_r;
			}
			else {
				tmp.appendRowElement(i, lhs(i, j_l).index_, lhs(i, j_l).value_ + rhs(i, j_r).value_);
				++j_l; ++j_r;
			}
		}

		while (j_l < lhs.nonzeros(i)) {
			tmp.appendRowElement(i, lhs(i, j_l).index_, lhs(i, j_l).value_);
			++j_l;
		}

		while (j_r < rhs.nonzeros(i)) {
			tmp.appendRowElement(i, rhs(i, j_r).index_, rhs(i, j_r).value_);
			++j_r;
		}
	}

	return tmp;
}


/**
* \brief Subtraction operator for the subtraction of two matrices (\f$ A=B-C \f$).
*
* \param lhs The left-hand side matrix for the matrix subtraction.
* \param rhs The right-hand-side matrix to be subtracted from the left-hand side matrix.
* \return The difference of the two matrices.
*/
const MatrixCRS operator-(const MatrixCRS& lhs, const MatrixCRS& rhs) {
	assert((lhs.rows() == rhs.rows() && rhs.columns() == rhs.columns()) && "Matrix sizes do not match");

	// analyze matrices to predict the required row storage capacities
	std::vector<std::size_t> capacities(lhs.rows(), 0);
	for (std::size_t i = 0; i < lhs.rows(); ++i) {
		std::size_t j_l(0), j_r(0);
		std::size_t accu(lhs.nonzeros(i) + rhs.nonzeros(i));

		while (j_l < lhs.nonzeros(i) && j_r < rhs.nonzeros(i)) {
			if (lhs(i, j_l).index_ < rhs(i, j_r).index_)
			++j_l;
			else if (lhs(i, j_l).index_ > rhs(i, j_r).index_)
			++j_r;
			else {
				--accu; ++j_l; ++j_r;
			}
		}

		capacities[i] = accu;
	}

	// merge matrices 
	MatrixCRS tmp(lhs.rows(), lhs.columns(), capacities);
	for (std::size_t i = 0; i < lhs.rows(); ++i) {
		std::size_t j_l(0), j_r(0);

		while (j_l < lhs.nonzeros(i) && j_r < rhs.nonzeros(i)) {
			if (lhs(i, j_l).index_ < rhs(i, j_r).index_) {
				tmp.appendRowElement(i, lhs(i, j_l).index_, lhs(i, j_l).value_);
				++j_l;
			}
			else if (lhs(i, j_l).index_ > rhs(i, j_r).index_) {
				tmp.appendRowElement(i, rhs(i, j_r).index_, rhs(i, j_r).value_);
				++j_r;
			}
			else {
				tmp.appendRowElement(i, lhs(i, j_l).index_, lhs(i, j_l).value_ - rhs(i, j_r).value_);
				++j_l; ++j_r;
			}
		}

		while (j_l < lhs.nonzeros(i)) {
			tmp.appendRowElement(i, lhs(i, j_l).index_, lhs(i, j_l).value_);
			++j_l;
		}

		while (j_r < rhs.nonzeros(i)) {
			tmp.appendRowElement(i, rhs(i, j_r).index_, rhs(i, j_r).value_);
			++j_r;
		}
	}

	return tmp;
}



/**
* \brief Multiplication operator for the multiplication of two matrices (\f$ A=B*C \f$).
*
* \param lhs The left-hand side matrix for the multiplication.
* \param rhs The right-hand-side matrix for the multiplication.
* \return The resulting matrix.
*/
const MatrixCRS operator*(const MatrixCRS& lhs, const MatrixCRS& rhs) {
	assert(lhs.columns() == rhs.rows() && "Matrix sizes do not match");

	MatrixCRS                   rhs_trans = rhs.getTranspose();
	

	
	MatrixCRS tmp(lhs.rows(), rhs.columns());
	
	const std::size_t height = lhs.rows();
	const std::size_t width = rhs.columns();
	
	unsigned int *rowLengths = new unsigned int[lhs.rows()];
	
	const IndexValuePair* const lhsRowValues = &lhs.values_[0];
	const std::size_t* const lhsRowIndices = &lhs.row_indices_[0];
	const std::size_t* const lhsRowLengths = &lhs.row_lengths_[0];
	
	const IndexValuePair* const rhsColValues = &rhs_trans.values_[0];
	const std::size_t* const rhsColIndices = &rhs_trans.row_indices_[0];
	const std::size_t* const rhsColLengths = &rhs_trans.row_lengths_[0];
	
	
	const IndexValuePair* const rhsRowValues = &rhs.values_[0];
	const std::size_t* const rhsRowIndices = &rhs.row_indices_[0];
	const std::size_t* const rhsRowLengths = &rhs.row_lengths_[0];

	std::size_t maxNonZerosLeft = 0;
	std::size_t maxNonZerosRight = 0;
	for (std::size_t i = 0; i < height; ++i) 
	maxNonZerosLeft = std::max(lhsRowLengths[i], maxNonZerosLeft);		
	for (std::size_t j = 0; j < width; ++j)
	maxNonZerosRight = std::max(rhsColLengths[j], maxNonZerosRight);
	std::size_t maxWidth = std::min(maxNonZerosLeft * maxNonZerosRight, rhs.rows());
	
	
	//std::cout << "size of temp 2D array : " << (maxWidth * height) << std::endl;
	
	IndexValuePair** newMat;
	newMat = (IndexValuePair**)malloc(sizeof(IndexValuePair*) * height);
	newMat[0] = (IndexValuePair*)malloc(sizeof(IndexValuePair) * height * maxWidth);
	
	for(int i = 1; i < height; i++)
	newMat[i] = newMat[i-1] + maxWidth;
	
	#pragma acc data create(newMat[height][maxWidth])
	{
		#pragma acc data copyout(rowLengths[0:lhs.rows()]) \
			copyin(rhsColValues[0:rhs_trans.values_.size()]) \
			copyin(rhsColIndices[0:rhs_trans.row_indices_.size()]) \
			copyin(rhsColLengths[0:rhs_trans.row_lengths_.size()]) \
			copyin(lhsRowValues[0:lhs.values_.size()]) \
			copyin(lhsRowIndices[0:lhs.row_indices_.size()]) \
			copyin(lhsRowLengths[0:lhs.row_lengths_.size()])
		{
			#pragma acc kernels
			{
#pragma acc loop independent
#pragma omp parallel for
				for (std::size_t i = 0; i < height; ++i) {
					
					rowLengths[i] = 0;
					std::size_t lMax = lhsRowLengths[i];
					const IndexValuePair* lPtr = &lhsRowValues[lhsRowIndices[i]];
					
					
					
					int nextCol = -1;
					int rowPtrs[lMax];
					for(int n = 0; n < lMax; n++)
						rowPtrs[n] = 0;
					while(true){
						
						int minColAbove = std::numeric_limits<int>::max();
						//(k (to find min col) + 2k (to calc product)) * k^2 cols = k^3 (or k^4 without extra storage)
						for (std::size_t n = 0; n < lMax; ++n) {
							std::size_t rRowMax = rhsRowLengths[lPtr[n].index_];
							const IndexValuePair* rRowPtr = &rhsRowValues[rhsRowIndices[lPtr[n].index_]];
							
							
							for(std::size_t m = rowPtrs[n]; m < rRowMax; ++m){
								int col = rRowPtr[m].index_;
								if(col > nextCol){
									rowPtrs[n] = m;
									minColAbove = std::min(col, minColAbove);
									break;
								}
							}
						}
						if(minColAbove == std::numeric_limits<int>::max()) break;
						nextCol = minColAbove;
						
						std::size_t j = nextCol;
						//for (std::size_t j = 0; j < width; ++j) {

						std::size_t rMax = rhsColLengths[j];
						
						
						
						const IndexValuePair* rPtr = &rhsColValues[rhsColIndices[j]];
						
						if(lPtr[0].index_ > rPtr[rMax - 1].index_) continue;
						if(rPtr[0].index_ > lPtr[lMax - 1].index_) continue;
						
						std::size_t k_l = 0, k_r = 0;
						double accu = 0.0;
						
						while (k_l < lMax && k_r < rMax) {
							if(lPtr[k_l].index_ < rPtr[k_r].index_)
							++k_l;
							else if(lPtr[k_l].index_ > rPtr[k_r].index_)
							++k_r;
							else{
								accu += lPtr[k_l].value_ * rPtr[k_r].value_;
								++k_l; ++k_r;
							}
						}

						if (accu != 0)
						{
							newMat[i][rowLengths[i]].value_ = accu;
							newMat[i][rowLengths[i]].index_ = j;
							rowLengths[i]++;
						}

					}	
				}
				
				
			}	
		}
		
		//make room in the new matrix
		int totalSize = 0;
		for (std::size_t i = 0; i < height; ++i){
			tmp.row_indices_[i] = totalSize;
			tmp.row_lengths_[i] = rowLengths[i];
			totalSize += rowLengths[i];
		}
		tmp.values_.resize(totalSize);
		//	tmp.reserveRowElements(i, rowLengths[i]);
		
		IndexValuePair * const values = &tmp.values_[0];
		const std::size_t * const rowIndices = &tmp.row_indices_[0];
		
		#pragma acc data copyout(values[totalSize]) \
			pcopyin(rowLengths[height], rowIndices[height]) \
			present(newMat[height][maxWidth])
		{
			#pragma acc kernels
			{
				
				
				// merge sparse rows into the matrix
				#pragma acc loop independent
				#pragma omp parallel for
				for (std::size_t i = 0; i < height; ++i) {
					#pragma acc loop independent
					for (std::size_t j = 0; j < rowLengths[i]; ++j){
						values[rowIndices[i] + j] = newMat[i][j];
					}
				}
			}
		}
		
	}
	
	free(newMat[0]);
	free(newMat);
	delete[] rowLengths;
	return tmp;
}

