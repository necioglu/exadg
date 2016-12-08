/*
 * PreconditionerBase.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_PRECONDITIONERBASE_H_
#define INCLUDE_PRECONDITIONERBASE_H_

using namespace dealii;

#include <deal.II/lac/parallel_vector.h>

#include "MatrixOperatorBase.h"

template<typename value_type>
class PreconditionerBase
{
public:
  virtual ~PreconditionerBase(){}

  virtual void vmult(parallel::distributed::Vector<value_type>        &dst,
                     const parallel::distributed::Vector<value_type>  &src) const = 0;

  virtual void update(MatrixOperatorBase const *matrix_operator) = 0;
};


#endif /* INCLUDE_PRECONDITIONERBASE_H_ */
