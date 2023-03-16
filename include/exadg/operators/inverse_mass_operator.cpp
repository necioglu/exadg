//
// Created by necioglu on 3/15/23.
//

#include <exadg/operators/inverse_mass_operator.h>

namespace ExaDG
{

template<int dim, int n_components, typename Number>
InverseMassOperator<dim, n_components, Number>::InverseMassOperator()
  : matrix_free(nullptr), dof_index(0), quad_index(0)
{
}
template<int dim, int n_components, typename Number>
void
InverseMassOperator<dim, n_components, Number>::initialize(
  const dealii::MatrixFree<dim, Number> & matrix_free_in,
  const unsigned int                      dof_index_in,
  const unsigned int                      quad_index_in)
{
  this->matrix_free = &matrix_free_in;
  dof_index         = dof_index_in;
  quad_index        = quad_index_in;

  cellwise_inverse_mass_not_available =
    matrix_free->get_dof_handler(dof_index).get_triangulation().all_reference_cells_are_simplex();

  // initialize mass operator
  dealii::AffineConstraints<Number> const & constraint =
    matrix_free->get_affine_constraints(dof_index);

  MassOperatorData<dim> mass_operator_data;
  mass_operator_data.dof_index  = dof_index;
  mass_operator_data.quad_index = quad_index;

  mass_operator_data.implement_block_diagonal_preconditioner_matrix_free = true;
  mass_operator_data.solver_block_diagonal         = Elementwise::Solver::GMRES;
  mass_operator_data.use_cell_based_loops          = true;
  mass_operator_data.preconditioner_block_diagonal = Elementwise::Preconditioner::InverseMassMatrix;
  mass_operator_data.solver_data_block_diagonal    = SolverData(1000, 1e-12, 1e-20, 1000);

  mass_operator.initialize(*matrix_free, constraint, mass_operator_data);

  // build a BlockJacobiPreconditioner and use the vmult(dst,src) for applying the inverse mass
  // operator on  source the vector
  mass_preconditioner =
    std::make_shared<BlockJacobiPreconditioner<MassOperator<dim, n_components, Number>>>(
      mass_operator);
}
template<int dim, int n_components, typename Number>
void
InverseMassOperator<dim, n_components, Number>::apply(
  InverseMassOperator::VectorType &       dst,
  const InverseMassOperator::VectorType & src) const
{
  VectorType dst_temp = dst;
  dst_temp.zero_out_ghost_values();
  dst.zero_out_ghost_values();

  VectorType temp = src;
  mass_preconditioner->vmult(dst_temp, temp);

  matrix_free->cell_loop(&This::cell_loop, this, dst, src);

  dst_temp -= dst;
  if(dst_temp.linfty_norm() / dst.linfty_norm() > 1e-14)
    std::cout << "difference between the methods is: " << dst_temp.linfty_norm() / dst.linfty_norm()
              << " size is: " << dst_temp.size() << std::endl;
}
template<int dim, int n_components, typename Number>
void
InverseMassOperator<dim, n_components, Number>::cell_loop(
  const dealii::MatrixFree<dim, Number> &,
  InverseMassOperator::VectorType &       dst,
  const InverseMassOperator::VectorType & src,
  const InverseMassOperator::Range &      cell_range) const
{
  Integrator          integrator(*matrix_free, dof_index, quad_index);
  CellwiseInverseMass inverse(integrator);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(src, 0);

    inverse.apply(integrator.begin_dof_values(), integrator.begin_dof_values());

    integrator.set_dof_values(dst, 0);
  }
}
// scalar
template class InverseMassOperator<2, 1, float>;
template class InverseMassOperator<2, 1, double>;

template class InverseMassOperator<3, 1, float>;
template class InverseMassOperator<3, 1, double>;

// dim components
template class InverseMassOperator<2, 2, float>;
template class InverseMassOperator<2, 2, double>;

template class InverseMassOperator<3, 3, float>;
template class InverseMassOperator<3, 3, double>;

// dim + 1 components
template class InverseMassOperator<2, 3, float>;
template class InverseMassOperator<2, 3, double>;

template class InverseMassOperator<3, 4, float>;
template class InverseMassOperator<3, 4, double>;

// compressible Navier-Stokes merged operators
template class InverseMassOperator<2, 4, float>;
template class InverseMassOperator<2, 4, double>;

template class InverseMassOperator<3, 5, float>;
template class InverseMassOperator<3, 5, double>;

} // namespace ExaDG