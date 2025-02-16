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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_BOX_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_BOX_H_

// ExaDG
#include <exadg/grid/periodic_box.h>

namespace ExaDG
{
namespace IncNS
{
enum class MeshType
{
  Cartesian,
  Curvilinear
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("MeshType",  mesh_type_string, "Type of mesh (Cartesian versus curvilinear).", dealii::Patterns::Selection("Cartesian|Curvilinear"));
    prm.leave_subsection();
    // clang-format on
  }

private:
  void
  parse_parameters() final
  {
    ApplicationBase<dim, Number>::parse_parameters();

    Utilities::string_to_enum(mesh_type, mesh_type_string);
  }

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type                = ProblemType::Unsteady;
    this->param.equation_type               = EquationType::NavierStokes;
    this->param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    this->param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    this->param.right_hand_side             = false;

    // PHYSICAL QUANTITIES
    this->param.start_time = 0.0;
    this->param.end_time   = 1.0;
    this->param.viscosity  = 1.0;

    // TEMPORAL DISCRETIZATION
    this->param.solver_type = SolverType::Unsteady;
    this->param.temporal_discretization =
      TemporalDiscretization::BDFDualSplittingScheme; // BDFPressureCorrection;
                                                      // //BDFCoupledSolution;
    this->param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Explicit;
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.cfl                           = 1.0;
    this->param.max_velocity                  = 1.0;

    // NUMERICAL PARAMETERS
    this->param.quad_rule_linearization =
      QuadratureRuleLinearization::Standard; // Overintegration32k;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = 1; // this->param.degree_u;
    // use EqualOrder so that we can also start with k=1 for the velocity!
    this->param.degree_p = DegreePressure::MixedOrder;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // div-div and continuity penalty
    this->param.use_divergence_penalty                     = true;
    this->param.divergence_penalty_factor                  = 1.0e0;
    this->param.use_continuity_penalty                     = true;
    this->param.continuity_penalty_factor                  = this->param.divergence_penalty_factor;
    this->param.apply_penalty_terms_in_postprocessing_step = true;

    // TURBULENCE
    this->param.turbulence_model_data.is_active        = false;
    this->param.turbulence_model_data.turbulence_model = TurbulenceEddyViscosityModel::Sigma;
    // Smagorinsky: 0.165
    // Vreman: 0.28
    // WALE: 0.50
    // Sigma: 1.35
    this->param.turbulence_model_data.constant = 1.35;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    this->param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::None;

    // projection step
    this->param.solver_projection         = SolverProjection::CG;
    this->param.preconditioner_projection = PreconditionerProjection::None;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // viscous step
    this->param.solver_viscous         = SolverViscous::CG;
    this->param.preconditioner_viscous = PreconditionerViscous::None;

    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // linear solver
    this->param.solver_momentum         = SolverMomentum::GMRES;
    this->param.preconditioner_momentum = MomentumPreconditioner::None;

    // COUPLED NAVIER-STOKES SOLVER

    // linear solver
    this->param.solver_coupled = SolverCoupled::GMRES;

    // preconditioning linear solver
    this->param.preconditioner_coupled = PreconditionerCoupled::None;

    this->param.preconditioner_velocity_block = MomentumPreconditioner::None;
    this->param.preconditioner_pressure_block = SchurComplementPreconditioner::None;
  }

  void
  create_grid() final
  {
    double const left = -1.0, right = 1.0;
    double const deformation = 0.1;

    bool curvilinear_mesh = false;
    if(mesh_type == MeshType::Cartesian)
    {
      // do nothing
    }
    else if(mesh_type == MeshType::Curvilinear)
    {
      curvilinear_mesh = true;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    create_periodic_box(this->grid->triangulation,
                        this->param.grid.n_refine_global,
                        this->grid->periodic_face_pairs,
                        this->n_subdivisions_1d_hypercube,
                        left,
                        right,
                        curvilinear_mesh,
                        deformation);
  }

  void
  set_boundary_descriptor()
  {
    // test case with purely periodic boundary conditions
    // boundary descriptors remain empty for velocity and pressure
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // no postprocessing

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  std::string mesh_type_string = "Cartesian";
  MeshType    mesh_type        = MeshType::Cartesian;
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_BOX_H_ */
