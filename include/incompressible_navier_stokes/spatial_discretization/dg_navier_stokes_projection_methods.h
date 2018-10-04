/*
 * DGNavierStokesProjectionMethods.h
 *
 *  Created on: Nov 7, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_PROJECTION_METHODS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_PROJECTION_METHODS_H_

#include "../../incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_base.h"
#include "../../incompressible_navier_stokes/spatial_discretization/projection_operators_and_solvers.h"
#include "../../poisson/spatial_discretization/laplace_operator.h"
#include "../../solvers_and_preconditioners/solvers/iterative_solvers.h"

namespace IncNS
{
/*
 * Base class for projection type splitting methods such as the high-order dual splitting scheme
 * (velocity-correction) or pressure correction schemes
 */
template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
class DGNavierStokesProjectionMethods
  : public DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>
{
public:
  typedef DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> BASE;

  typedef typename BASE::VectorType VectorType;

  DGNavierStokesProjectionMethods(parallel::distributed::Triangulation<dim> const & triangulation,
                                  InputParameters<dim> const &                      parameters_in,
                                  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor_in)
    : BASE(triangulation, parameters_in, postprocessor_in),
      use_optimized_projection_operator(false) // TODO
  {
    AssertThrow(fe_degree_p > 0,
                ExcMessage("Polynomial degree of pressure shape functions has to be larger than "
                           "zero for dual splitting scheme and pressure-correction scheme."));
  }

  virtual ~DGNavierStokesProjectionMethods()
  {
  }

  virtual void
  setup_solvers(double const & time_step_size,
                double const & scaling_factor_time_derivative_term) = 0;

  // velocity divergence
  void
  evaluate_velocity_divergence_term(VectorType &       dst,
                                    VectorType const & src,
                                    double const       evaluation_time) const;

  // pressure gradient term
  void
  evaluate_pressure_gradient_term(VectorType &       dst,
                                  VectorType const & src,
                                  double const       evaluation_time) const;

  // rhs viscous term (add)
  void
  rhs_add_viscous_term(VectorType & dst, double const evaluation_time) const;

  // rhs pressure Poisson equation: inhomogeneous parts of boundary face
  // integrals of negative Laplace operator
  void
  rhs_ppe_laplace_add(VectorType & dst, double const & evaluation_time) const;

  // solve pressure step
  unsigned int
  solve_pressure(VectorType & dst, VectorType const & src) const;

  // solve projection step
  unsigned int
  solve_projection(VectorType &       dst,
                   VectorType const & src,
                   VectorType const & velocity,
                   double const       time_step_size) const;

  // apply projection operator
  void
  apply_projection_operator(VectorType & dst, VectorType const & src) const;

  // Evaluate residual of steady, coupled incompressible Navier-Stokes equations
  void
  evaluate_nonlinear_residual_steady(VectorType &       dst_u,
                                     VectorType &       dst_p,
                                     VectorType const & src_u,
                                     VectorType const & src_p,
                                     double const &     evaluation_time);

  // apply homogeneous Laplace operator
  void
  apply_laplace_operator(VectorType & dst, VectorType const & src) const;

protected:
  virtual void
  setup_pressure_poisson_solver(double const time_step_size);

  void
  setup_projection_solver();

  // Pressure Poisson equation
  Poisson::LaplaceOperator<dim, fe_degree_p, Number> laplace_operator;

  std::shared_ptr<PreconditionerBase<Number>> preconditioner_pressure_poisson;

  std::shared_ptr<IterativeSolverBase<VectorType>> pressure_poisson_solver;

  // Projection method

  // div-div-penalty and continuity penalty operator
  std::shared_ptr<DivergencePenaltyOperator<dim,
                                            fe_degree,
                                            fe_degree_p,
                                            fe_degree_xwall,
                                            xwall_quad_rule,
                                            Number>>
    divergence_penalty_operator;

  std::shared_ptr<ContinuityPenaltyOperator<dim,
                                            fe_degree,
                                            fe_degree_p,
                                            fe_degree_xwall,
                                            xwall_quad_rule,
                                            Number>>
    continuity_penalty_operator;

  // apply all operators in one matrix-free loop
  bool use_optimized_projection_operator;

  // projection operator
  std::shared_ptr<ProjectionOperatorBase<dim>> projection_operator;

  // projection solver
  std::shared_ptr<IterativeSolverBase<VectorType>> projection_solver;
  std::shared_ptr<PreconditionerBase<Number>>      preconditioner_projection;
};

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesProjectionMethods<dim,
                                fe_degree,
                                fe_degree_p,
                                fe_degree_xwall,
                                xwall_quad_rule,
                                Number>::setup_pressure_poisson_solver(double const time_step_size)
{
  // setup Laplace operator
  Poisson::LaplaceOperatorData<dim> laplace_operator_data;
  laplace_operator_data.dof_index  = this->get_dof_index_pressure();
  laplace_operator_data.quad_index = this->get_quad_index_pressure();
  laplace_operator_data.IP_factor  = this->param.IP_factor_pressure;

  // TODO: do this in derived classes
  if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    laplace_operator_data.operator_is_singular = this->param.pure_dirichlet_bc;
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    // One can show that the linear system of equations of the PPE is consistent
    // in case of the pressure-correction scheme if the velocity Dirichlet BC is consistent.
    // So there should be no need to solve a tranformed linear system of equations.
    //    laplace_operator_data.operator_is_singular = false;

    // In principle, it works (since the linear system of equations is consistent)
    // but we detected no convergence for some test cases and specific parameters.
    // Hence, for reasons of robustness we also solve a transformed linear system of equations
    // in case of the pressure-correction scheme.
    laplace_operator_data.operator_is_singular = this->param.pure_dirichlet_bc;
  }

  if(this->param.use_approach_of_ferrer == true)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout
      << "Approach of Ferrer et al. is applied: IP_factor_pressure is scaled by time_step_size/time_step_size_ref!"
      << std::endl;

    // only makes sense in case of constant time step sizes
    laplace_operator_data.IP_factor =
      this->param.IP_factor_pressure / time_step_size * this->param.deltat_ref;
  }

  laplace_operator_data.bc = this->boundary_descriptor_laplace;

  laplace_operator_data.periodic_face_pairs_level0 = this->periodic_face_pairs;
  laplace_operator.initialize(this->mapping, this->data, laplace_operator_data);

  // setup preconditioner
  if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::Jacobi)
  {
    preconditioner_pressure_poisson.reset(
      new JacobiPreconditioner<Poisson::LaplaceOperator<dim, fe_degree_p, Number>>(
        laplace_operator));
  }
  else if(this->param.preconditioner_pressure_poisson ==
          PreconditionerPressurePoisson::GeometricMultigrid)
  {
    MultigridData mg_data;
    mg_data = this->param.multigrid_data_pressure_poisson;

    // use single precision for multigrid
    typedef float MultigridNumber;

    typedef MyMultigridPreconditionerDG<dim,
                                        Number,
                                        Poisson::LaplaceOperator<dim, fe_degree_p, MultigridNumber>>
      MULTIGRID;

    preconditioner_pressure_poisson.reset(new MULTIGRID());

    std::shared_ptr<MULTIGRID> mg_preconditioner =
      std::dynamic_pointer_cast<MULTIGRID>(preconditioner_pressure_poisson);

    // TODO: not necessary
    //    typedef typename Triangulation<dim>::cell_iterator    TriIterator;
    //    std::vector<GridTools::PeriodicFacePair<TriIterator>> periodic_face_pairs;

    mg_preconditioner->initialize(mg_data,
                                  this->dof_handler_p,
                                  this->mapping,
                                  laplace_operator.get_operator_data().bc->dirichlet_bc,
                                  (void *)&laplace_operator.get_operator_data());
  }
  else
  {
    AssertThrow(
      this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::None ||
        this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::Jacobi ||
        this->param.preconditioner_pressure_poisson ==
          PreconditionerPressurePoisson::GeometricMultigrid,
      ExcMessage("Specified preconditioner for pressure Poisson equation not implemented"));
  }

  if(this->param.solver_pressure_poisson == SolverPressurePoisson::PCG)
  {
    // setup solver data
    CGSolverData solver_data;
    // use default value of max_iter
    solver_data.solver_tolerance_abs = this->param.abs_tol_pressure;
    solver_data.solver_tolerance_rel = this->param.rel_tol_pressure;
    // default value of use_preconditioner = false
    if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::Jacobi ||
       this->param.preconditioner_pressure_poisson ==
         PreconditionerPressurePoisson::GeometricMultigrid)
    {
      solver_data.use_preconditioner = true;
    }

    // setup solver
    pressure_poisson_solver.reset(
      new CGSolver<Poisson::LaplaceOperator<dim, fe_degree_p, Number>,
                   PreconditionerBase<Number>,
                   VectorType>(laplace_operator, *preconditioner_pressure_poisson, solver_data));
  }
  else if(this->param.solver_pressure_poisson == SolverPressurePoisson::FGMRES)
  {
    FGMRESSolverData solver_data;
    // use default value of max_iter
    solver_data.solver_tolerance_abs = this->param.abs_tol_pressure;
    solver_data.solver_tolerance_rel = this->param.rel_tol_pressure;
    solver_data.max_n_tmp_vectors    = this->param.max_n_tmp_vectors_pressure_poisson;
    // use default value of update_preconditioner (=false)

    if(this->param.preconditioner_pressure_poisson == PreconditionerPressurePoisson::Jacobi ||
       this->param.preconditioner_pressure_poisson ==
         PreconditionerPressurePoisson::GeometricMultigrid)
    {
      solver_data.use_preconditioner = true;
    }

    pressure_poisson_solver.reset(
      new FGMRESSolver<Poisson::LaplaceOperator<dim, fe_degree_p, Number>,
                       PreconditionerBase<Number>,
                       VectorType>(laplace_operator,
                                   *preconditioner_pressure_poisson,
                                   solver_data));
  }
  else
  {
    AssertThrow(
      this->param.solver_viscous == SolverViscous::PCG ||
        this->param.solver_viscous == SolverViscous::FGMRES,
      ExcMessage(
        "Specified  solver for pressure Poisson equation not implemented - possibilities are PCG and FGMRES"));
  }
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesProjectionMethods<dim,
                                fe_degree,
                                fe_degree_p,
                                fe_degree_xwall,
                                xwall_quad_rule,
                                Number>::setup_projection_solver()
{
  // setup divergence and continuity penalty operators
  if(this->param.use_divergence_penalty == true)
  {
    DivergencePenaltyOperatorData div_penalty_data;
    div_penalty_data.type_penalty_parameter = this->param.type_penalty_parameter;
    div_penalty_data.viscosity              = this->param.viscosity;
    div_penalty_data.penalty_parameter      = this->param.divergence_penalty_factor;

    divergence_penalty_operator.reset(
      new DivergencePenaltyOperator<dim,
                                    fe_degree,
                                    fe_degree_p,
                                    fe_degree_xwall,
                                    xwall_quad_rule,
                                    Number>(this->data,
                                            this->get_dof_index_velocity(),
                                            this->get_quad_index_velocity_linear(),
                                            div_penalty_data));
  }

  if(this->param.use_continuity_penalty == true)
  {
    ContinuityPenaltyOperatorData<dim> conti_penalty_data;
    conti_penalty_data.type_penalty_parameter = this->param.type_penalty_parameter;
    conti_penalty_data.viscosity              = this->param.viscosity;
    conti_penalty_data.penalty_parameter      = this->param.continuity_penalty_factor;
    conti_penalty_data.which_components       = this->param.continuity_penalty_components;
    // The projected velocity field does not fulfill the velocity Dirichlet boundary conditions.
    // Hence, do not use the prescribed boundary data when applying the continuity penalty operator.
    conti_penalty_data.use_boundary_data = false;
    conti_penalty_data.bc                = this->boundary_descriptor_velocity;

    continuity_penalty_operator.reset(
      new ContinuityPenaltyOperator<dim,
                                    fe_degree,
                                    fe_degree_p,
                                    fe_degree_xwall,
                                    xwall_quad_rule,
                                    Number>(this->data,
                                            this->get_dof_index_velocity(),
                                            this->get_quad_index_velocity_linear(),
                                            conti_penalty_data));
  }

  // setup projection operator and projection solver

  // no penalty terms
  if(this->param.use_divergence_penalty == false && this->param.use_continuity_penalty == false)
  {
    projection_solver.reset(new ProjectionSolverNoPenalty<dim, fe_degree, Number>(
      this->data, this->get_dof_index_velocity(), this->get_quad_index_velocity_linear()));
  }
  // divergence penalty only
  else if(this->param.use_divergence_penalty == true && this->param.use_continuity_penalty == false)
  {
    // use direct solver
    if(this->param.solver_projection == SolverProjection::LU)
    {
      AssertThrow(divergence_penalty_operator.get() != 0,
                  ExcMessage("Divergence penalty operator has not been initialized."));

      // projection operator
      typedef ProjectionOperatorDivergencePenaltyDirect<dim,
                                                        fe_degree,
                                                        fe_degree_p,
                                                        fe_degree_xwall,
                                                        xwall_quad_rule,
                                                        Number>
        PROJ_OPERATOR;

      projection_operator.reset(new PROJ_OPERATOR(*divergence_penalty_operator));

      typedef DirectProjectionSolverDivergencePenalty<dim,
                                                      fe_degree,
                                                      fe_degree_p,
                                                      fe_degree_xwall,
                                                      xwall_quad_rule,
                                                      Number>
        PROJ_SOLVER;

      projection_solver.reset(
        new PROJ_SOLVER(std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator)));
    }
    // use iterative solver (PCG)
    else if(this->param.solver_projection == SolverProjection::PCG)
    {
      AssertThrow(divergence_penalty_operator.get() != 0,
                  ExcMessage("Divergence penalty operator has not been initialized."));

      // projection operator
      typedef ProjectionOperatorDivergencePenaltyIterative<dim,
                                                           fe_degree,
                                                           fe_degree_p,
                                                           fe_degree_xwall,
                                                           xwall_quad_rule,
                                                           Number>
        PROJ_OPERATOR;

      projection_operator.reset(new PROJ_OPERATOR(*divergence_penalty_operator));

      // solver
      ProjectionSolverData projection_solver_data;
      projection_solver_data.solver_tolerance_abs = this->param.abs_tol_projection;
      projection_solver_data.solver_tolerance_rel = this->param.rel_tol_projection;

      typedef IterativeProjectionSolverDivergencePenalty<dim,
                                                         fe_degree,
                                                         fe_degree_p,
                                                         fe_degree_xwall,
                                                         xwall_quad_rule,
                                                         Number>
        PROJ_SOLVER;

      projection_solver.reset(
        new PROJ_SOLVER(*std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator),
                        projection_solver_data));
    }
    else
    {
      AssertThrow(this->param.solver_projection == SolverProjection::LU ||
                    this->param.solver_projection == SolverProjection::PCG,
                  ExcMessage("Specified projection solver not implemented."));
    }
  }
  // both divergence and continuity penalty terms
  else if(this->param.use_divergence_penalty == true &&
          this->param.use_continuity_penalty == true && use_optimized_projection_operator == false)
  {
    AssertThrow(divergence_penalty_operator.get() != 0,
                ExcMessage("Divergence penalty operator has not been initialized."));

    AssertThrow(continuity_penalty_operator.get() != 0,
                ExcMessage("Continuity penalty operator has not been initialized."));

    // projection operator consisting of mass matrix operator,
    // divergence penalty operator, and continuity penalty operator
    typedef ProjectionOperatorDivergenceAndContinuityPenalty<dim,
                                                             fe_degree,
                                                             fe_degree_p,
                                                             fe_degree_xwall,
                                                             xwall_quad_rule,
                                                             Number>
      PROJ_OPERATOR;

    projection_operator.reset(new PROJ_OPERATOR(this->mass_matrix_operator,
                                                *this->divergence_penalty_operator,
                                                *this->continuity_penalty_operator));

    // preconditioner
    if(this->param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix)
    {
      preconditioner_projection.reset(new InverseMassMatrixPreconditioner<dim, fe_degree, Number>(
        this->data, this->get_dof_index_velocity(), this->get_quad_index_velocity_linear()));
    }
    else if(this->param.preconditioner_projection == PreconditionerProjection::PointJacobi)
    {
      // Note that at this point (when initializing the Jacobi preconditioner and calculating the
      // diagonal) the penalty parameter of the projection operator has not been calculated and the
      // time step size has not been set. Hence, update_preconditioner = true should be used for the
      // Jacobi preconditioner in order to use to correct diagonal for preconditioning.
      preconditioner_projection.reset(new JacobiPreconditioner<PROJ_OPERATOR>(
        *std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator)));
    }
    else if(this->param.preconditioner_projection == PreconditionerProjection::BlockJacobi)
    {
      // Note that at this point (when initializing the Jacobi preconditioner)
      // the penalty parameter of the projection operator has not been calculated and the time step
      // size has not been set. Hence, update_preconditioner = true should be used for the Jacobi
      // preconditioner in order to use to correct diagonal blocks for preconditioning.
      preconditioner_projection.reset(new BlockJacobiPreconditioner<PROJ_OPERATOR>(
        *std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator)));
    }
    else
    {
      // clang-format off
      AssertThrow(this->param.preconditioner_projection == PreconditionerProjection::None ||
                    this->param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix ||
                    this->param.preconditioner_projection == PreconditionerProjection::PointJacobi ||
                    this->param.preconditioner_projection == PreconditionerProjection::BlockJacobi,
                  ExcMessage("Specified preconditioner of projection solver not implemented."));
      // clang-format on
    }

    // solver
    if(this->param.solver_projection == SolverProjection::PCG)
    {
      // setup solver data
      CGSolverData projection_solver_data;
      // use default value of max_iter
      projection_solver_data.solver_tolerance_abs = this->param.abs_tol_projection;
      projection_solver_data.solver_tolerance_rel = this->param.rel_tol_projection;
      // default value of use_preconditioner is false
      if(this->param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix ||
         this->param.preconditioner_projection == PreconditionerProjection::PointJacobi ||
         this->param.preconditioner_projection == PreconditionerProjection::BlockJacobi)
      {
        projection_solver_data.use_preconditioner    = true;
        projection_solver_data.update_preconditioner = this->param.update_preconditioner_projection;
      }
      else
      {
        AssertThrow(
          this->param.preconditioner_projection == PreconditionerProjection::None ||
            this->param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix ||
            this->param.preconditioner_projection == PreconditionerProjection::PointJacobi ||
            this->param.preconditioner_projection == PreconditionerProjection::BlockJacobi,
          ExcMessage("Specified preconditioner of projection solver not implemented."));
      }

      // setup solver
      projection_solver.reset(new CGSolver<PROJ_OPERATOR, PreconditionerBase<Number>, VectorType>(
        *std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator),
        *preconditioner_projection,
        projection_solver_data));
    }
    else
    {
      AssertThrow(this->param.solver_projection == SolverProjection::PCG,
                  ExcMessage("Specified projection solver not implemented."));
    }
  }
  // TODO:
  // both divergence and continuity penalty terms
  else if(this->param.use_divergence_penalty == true &&
          this->param.use_continuity_penalty == true && use_optimized_projection_operator == true)
  {
    OptimizedProjectionOperatorData<dim> proj_op_data;
    proj_op_data.type_penalty_parameter = this->param.type_penalty_parameter;
    proj_op_data.viscosity              = this->param.viscosity;
    proj_op_data.penalty_parameter      = this->param.continuity_penalty_factor;
    proj_op_data.which_components       = this->param.continuity_penalty_components;
    proj_op_data.use_boundary_data      = false;
    proj_op_data.bc                     = this->boundary_descriptor_velocity;

    typedef ProjectionOperatorOptimized<dim,
                                        fe_degree,
                                        fe_degree_p,
                                        fe_degree_xwall,
                                        xwall_quad_rule,
                                        Number>
      PROJ_OPERATOR;

    projection_operator.reset(new PROJ_OPERATOR(this->data,
                                                this->get_dof_index_velocity(),
                                                this->get_quad_index_velocity_linear(),
                                                proj_op_data));

    // preconditioner
    if(this->param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix)
    {
      preconditioner_projection.reset(new InverseMassMatrixPreconditioner<dim, fe_degree, Number>(
        this->data, this->get_dof_index_velocity(), this->get_quad_index_velocity_linear()));
    }
    else
    {
      AssertThrow(this->param.preconditioner_projection == PreconditionerProjection::None ||
                    this->param.preconditioner_projection ==
                      PreconditionerProjection::InverseMassMatrix,
                  ExcMessage("Specified preconditioner of projection solver not implemented."));
    }

    // solver
    if(this->param.solver_projection == SolverProjection::PCG)
    {
      // setup solver data
      CGSolverData projection_solver_data;
      // use default value of max_iter
      projection_solver_data.solver_tolerance_abs = this->param.abs_tol_projection;
      projection_solver_data.solver_tolerance_rel = this->param.rel_tol_projection;
      // default value of use_preconditioner = false
      if(this->param.preconditioner_projection == PreconditionerProjection::InverseMassMatrix)
      {
        projection_solver_data.use_preconditioner    = true;
        projection_solver_data.update_preconditioner = this->param.update_preconditioner_projection;
      }
      else
      {
        AssertThrow(this->param.preconditioner_projection == PreconditionerProjection::None ||
                      this->param.preconditioner_projection ==
                        PreconditionerProjection::InverseMassMatrix,
                    ExcMessage("Specified preconditioner of projection solver not implemented."));
      }

      // setup solver
      projection_solver.reset(new CGSolver<PROJ_OPERATOR, PreconditionerBase<Number>, VectorType>(
        *std::dynamic_pointer_cast<PROJ_OPERATOR>(projection_operator),
        *preconditioner_projection,
        projection_solver_data));
    }
    else
    {
      AssertThrow(this->param.solver_projection == SolverProjection::PCG,
                  ExcMessage("Specified projection solver not implemented."));
    }
  }
  else
  {
    AssertThrow(
      false,
      ExcMessage(
        "Specified combination of divergence and continuity penalty operators not implemented."));
  }
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesProjectionMethods<dim,
                                fe_degree,
                                fe_degree_p,
                                fe_degree_xwall,
                                xwall_quad_rule,
                                Number>::evaluate_velocity_divergence_term(VectorType &       dst,
                                                                           VectorType const & src,
                                                                           double const
                                                                             evaluation_time) const
{
  this->divergence_operator.evaluate(dst, src, evaluation_time);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesProjectionMethods<dim,
                                fe_degree,
                                fe_degree_p,
                                fe_degree_xwall,
                                xwall_quad_rule,
                                Number>::evaluate_pressure_gradient_term(VectorType &       dst,
                                                                         VectorType const & src,
                                                                         double const
                                                                           evaluation_time) const
{
  this->gradient_operator.evaluate(dst, src, evaluation_time);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesProjectionMethods<dim,
                                fe_degree,
                                fe_degree_p,
                                fe_degree_xwall,
                                xwall_quad_rule,
                                Number>::rhs_add_viscous_term(VectorType & dst,
                                                              double const evaluation_time) const
{
  this->viscous_operator.rhs_add(dst, evaluation_time);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesProjectionMethods<dim,
                                fe_degree,
                                fe_degree_p,
                                fe_degree_xwall,
                                xwall_quad_rule,
                                Number>::rhs_ppe_laplace_add(VectorType &   dst,
                                                             double const & evaluation_time) const
{
  const Poisson::LaplaceOperatorData<dim> & data = this->laplace_operator.get_operator_data();

  // Set correct time for evaluation of functions on pressure Dirichlet boundaries
  // (not needed for pressure Neumann boundaries because all functions are ZeroFunction in Neumann
  // BC map!)
  for(auto & it : data.bc->dirichlet_bc)
  {
    it.second->set_time(evaluation_time);
  }

  this->laplace_operator.rhs_add(dst);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
unsigned int
DGNavierStokesProjectionMethods<dim,
                                fe_degree,
                                fe_degree_p,
                                fe_degree_xwall,
                                xwall_quad_rule,
                                Number>::solve_pressure(VectorType &       dst,
                                                        VectorType const & src) const
{
  //  typedef float MultigridNumber;
  //  typedef MyMultigridPreconditionerLaplace<dim, Number,
  //      LaplaceOperator<dim, fe_degree_p, MultigridNumber>, LaplaceOperatorData<dim> > MULTIGRID;
  //
  //  std::shared_ptr<MULTIGRID> mg_preconditioner
  //    = std::dynamic_pointer_cast<MULTIGRID>(preconditioner_pressure_poisson);
  //
  //  CheckMultigrid<dim,Number,LaplaceOperator<dim,fe_degree_p, Number>,MULTIGRID>
  //    check_multigrid(this->laplace_operator,mg_preconditioner);
  //  check_multigrid.check();

  unsigned int n_iter = this->pressure_poisson_solver->solve(dst, src);

  return n_iter;
}


template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesProjectionMethods<dim,
                                fe_degree,
                                fe_degree_p,
                                fe_degree_xwall,
                                xwall_quad_rule,
                                Number>::apply_laplace_operator(VectorType &       dst,
                                                                VectorType const & src) const
{
  this->laplace_operator.vmult(dst, src);
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
unsigned int
DGNavierStokesProjectionMethods<dim,
                                fe_degree,
                                fe_degree_p,
                                fe_degree_xwall,
                                xwall_quad_rule,
                                Number>::solve_projection(VectorType &       dst,
                                                          VectorType const & src,
                                                          VectorType const & velocity,
                                                          double const       time_step_size) const
{
  if(use_optimized_projection_operator == false)
  {
    // Update projection operator, i.e., the penalty parameters that depend on
    // the current solution (velocity field).
    if(this->param.use_divergence_penalty == true)
    {
      divergence_penalty_operator->calculate_array_penalty_parameter(velocity);
    }
    if(this->param.use_continuity_penalty == true)
    {
      continuity_penalty_operator->calculate_array_penalty_parameter(velocity);
    }
  }
  // TODO
  else // use_optimized_projection_operator == true
  {
    typedef ProjectionOperatorOptimized<dim,
                                        fe_degree,
                                        fe_degree_p,
                                        fe_degree_xwall,
                                        xwall_quad_rule,
                                        Number>
      PROJ_OPERATOR;

    std::shared_ptr<PROJ_OPERATOR> proj_op =
      std::dynamic_pointer_cast<PROJ_OPERATOR>(this->projection_operator);
    AssertThrow(proj_op.get() != 0,
                ExcMessage("Projection operator is not initialized correctly."));

    proj_op->calculate_array_penalty_parameter(velocity);
  }

  // Set the correct time step size.
  if(projection_operator.get() != 0)
    projection_operator->set_time_step_size(time_step_size);

  // Solve projection equation.
  unsigned int n_iter = this->projection_solver->solve(dst, src);

  return n_iter;
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesProjectionMethods<dim,
                                fe_degree,
                                fe_degree_p,
                                fe_degree_xwall,
                                xwall_quad_rule,
                                Number>::apply_projection_operator(VectorType &       dst,
                                                                   VectorType const & src) const
{
  if(use_optimized_projection_operator == false)
  {
    typedef ProjectionOperatorDivergenceAndContinuityPenalty<dim,
                                                             fe_degree,
                                                             fe_degree_p,
                                                             fe_degree_xwall,
                                                             xwall_quad_rule,
                                                             Number>
      PROJ_OPERATOR;

    std::shared_ptr<PROJ_OPERATOR> proj_op =
      std::dynamic_pointer_cast<PROJ_OPERATOR>(this->projection_operator);
    AssertThrow(proj_op.get() != 0,
                ExcMessage("Projection operator is not initialized correctly."));

    proj_op->vmult(dst, src);
  }
  // TODO
  else // use_optimized_projection_operator == true
  {
    typedef ProjectionOperatorOptimized<dim,
                                        fe_degree,
                                        fe_degree_p,
                                        fe_degree_xwall,
                                        xwall_quad_rule,
                                        Number>
      PROJ_OPERATOR;

    std::shared_ptr<PROJ_OPERATOR> proj_op =
      std::dynamic_pointer_cast<PROJ_OPERATOR>(this->projection_operator);
    AssertThrow(proj_op.get() != 0,
                ExcMessage("Projection operator is not initialized correctly."));

    proj_op->vmult(dst, src);
  }
}

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
void
DGNavierStokesProjectionMethods<
  dim,
  fe_degree,
  fe_degree_p,
  fe_degree_xwall,
  xwall_quad_rule,
  Number>::evaluate_nonlinear_residual_steady(VectorType &       dst_u,
                                              VectorType &       dst_p,
                                              VectorType const & src_u,
                                              VectorType const & src_p,
                                              double const &     evaluation_time)
{
  // velocity-block

  // set dst_u to zero. This is necessary since subsequent operators
  // call functions of type ..._add
  dst_u = 0.0;

  if(this->param.right_hand_side == true)
  {
    this->body_force_operator.evaluate(dst_u, evaluation_time);
    // Shift body force term to the left-hand side of the equation.
    // This works since body_force_operator is the first operator
    // that is evaluated.
    dst_u *= -1.0;
  }

  if(this->param.equation_type == EquationType::NavierStokes)
    this->convective_operator.evaluate_add(dst_u, src_u, evaluation_time);

  this->viscous_operator.evaluate_add(dst_u, src_u, evaluation_time);

  // gradient operator scaled by scaling_factor_continuity
  this->gradient_operator.evaluate_add(dst_u, src_p, evaluation_time);

  // pressure-block

  this->divergence_operator.evaluate(dst_p, src_u, evaluation_time);
  // multiply by -1.0 since we use a formulation with symmetric saddle point matrix
  // with respect to pressure gradient term and velocity divergence term
  dst_p *= -1.0;
}


} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_DG_NAVIER_STOKES_PROJECTION_METHODS_H_ \
        */
