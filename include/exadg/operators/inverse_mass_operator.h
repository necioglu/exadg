/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_
#define INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/operators.h>

// ExaDG
#include <exadg/matrix_free/integrators.h>
#include <exadg/operators/mass_operator.h>
#include <exadg/solvers_and_preconditioners/preconditioners/block_jacobi_preconditioner.h>

namespace ExaDG
{
template<int dim, int n_components, typename Number>
class InverseMassOperator
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef InverseMassOperator<dim, n_components, Number> This;

  typedef CellIntegrator<dim, n_components, Number> Integrator;

  // use a template parameter of -1 to select the precompiled version of this operator
  typedef dealii::MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, n_components, Number>
    CellwiseInverseMass;

  typedef std::pair<unsigned int, unsigned int> Range;

public:
  InverseMassOperator();

  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
             unsigned int const                      dof_index_in,
             unsigned int const                      quad_index_in);

  void
  initialize_inverse_mass_operator_with_block_jacobi();

  void
  apply(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_loop(dealii::MatrixFree<dim, Number> const &,
            VectorType &       dst,
            VectorType const & src,
            Range const &      cell_range) const;

  dealii::MatrixFree<dim, Number> const * matrix_free;

  unsigned int dof_index, quad_index;

  // Mass solver (used only if inverse mass operator is not avaliable)
  std::shared_ptr<Krylov::SolverBase<VectorType>> mass_solver;

  MassOperator<dim, n_components, Number> mass_operator;

  std::shared_ptr<PreconditionerBase<Number>> mass_preconditioner;

    MassOperator<dim, n_components, Number> mass_operator2;

  std::shared_ptr<PreconditionerBase<Number>> mass_preconditioner2;

  bool cellwise_inverse_mass_not_available;

  bool use_solver;
};



} // namespace ExaDG


#endif /* INCLUDE_OPERATORS_INVERSEMASSMATRIX_H_ */
