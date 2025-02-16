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

#include <exadg/incompressible_navier_stokes/driver_precursor.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/create_operator.h>
#include <exadg/incompressible_navier_stokes/time_integration/create_time_integrator.h>
#include <exadg/time_integration/time_step_calculation.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
DriverPrecursor<dim, Number>::DriverPrecursor(
  MPI_Comm const &                                       comm,
  std::shared_ptr<ApplicationBasePrecursor<dim, Number>> app,
  bool const                                             is_test)
  : mpi_comm(comm),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    is_test(is_test),
    application(app),
    use_adaptive_time_stepping(false)
{
  print_general_info<Number>(pcout, mpi_comm, is_test);
}

template<int dim, typename Number>
void
DriverPrecursor<dim, Number>::set_start_time() const
{
  // Setup time integrator and get time step size
  double const start_time = std::min(application->get_parameters_precursor().start_time,
                                     application->get_parameters().start_time);

  // Set the same time step size for both time integrators
  time_integrator_pre->reset_time(start_time);
  time_integrator->reset_time(start_time);
}

template<int dim, typename Number>
void
DriverPrecursor<dim, Number>::synchronize_time_step_size() const
{
  double const EPSILON = 1.e-10;

  // Setup time integrator and get time step size
  double time_step_size_pre = std::numeric_limits<double>::max();
  double time_step_size     = std::numeric_limits<double>::max();

  // get time step sizes
  if(use_adaptive_time_stepping == true)
  {
    if(time_integrator_pre->get_time() >
       application->get_parameters_precursor().start_time - EPSILON)
      time_step_size_pre = time_integrator_pre->get_time_step_size();

    if(time_integrator->get_time() > application->get_parameters().start_time - EPSILON)
      time_step_size = time_integrator->get_time_step_size();
  }
  else
  {
    time_step_size_pre = time_integrator_pre->get_time_step_size();
    time_step_size     = time_integrator->get_time_step_size();
  }

  // take the minimum
  time_step_size = std::min(time_step_size_pre, time_step_size);

  // decrease time_step in order to exactly hit end_time
  if(use_adaptive_time_stepping == false)
  {
    // assume that the precursor domain is the first to start and the last to end
    time_step_size =
      adjust_time_step_to_hit_end_time(application->get_parameters_precursor().start_time,
                                       application->get_parameters_precursor().end_time,
                                       time_step_size);

    pcout << std::endl
          << "Combined time step size for both domains: " << time_step_size << std::endl;
  }

  // set the time step size
  time_integrator_pre->set_current_time_step_size(time_step_size);
  time_integrator->set_current_time_step_size(time_step_size);
}

template<int dim, typename Number>
void
DriverPrecursor<dim, Number>::setup()
{
  dealii::Timer timer;
  timer.restart();

  pcout << std::endl << "Setting up incompressible Navier-Stokes solver:" << std::endl;

  application->setup();

  // constant vs. adaptive time stepping
  use_adaptive_time_stepping = application->get_parameters_precursor().adaptive_time_stepping;

  // initialize pde_operator_pre (precursor domain)
  pde_operator_pre = create_operator<dim, Number>(application->get_grid_precursor(),
                                                  nullptr /* grid motion */,
                                                  application->get_boundary_descriptor_precursor(),
                                                  application->get_field_functions_precursor(),
                                                  application->get_parameters_precursor(),
                                                  "fluid",
                                                  mpi_comm);

  // initialize operator_base (actual domain)
  pde_operator = create_operator<dim, Number>(application->get_grid(),
                                              nullptr /* grid motion */,
                                              application->get_boundary_descriptor(),
                                              application->get_field_functions(),
                                              application->get_parameters(),
                                              "fluid",
                                              mpi_comm);


  // initialize matrix_free precursor
  matrix_free_data_pre = std::make_shared<MatrixFreeData<dim, Number>>();
  matrix_free_data_pre->append(pde_operator_pre);

  matrix_free_pre = std::make_shared<dealii::MatrixFree<dim, Number>>();
  if(application->get_parameters_precursor().use_cell_based_face_loops)
    Categorization::do_cell_based_loops(*application->get_grid_precursor()->triangulation,
                                        matrix_free_data_pre->data);
  matrix_free_pre->reinit(*application->get_grid_precursor()->mapping,
                          matrix_free_data_pre->get_dof_handler_vector(),
                          matrix_free_data_pre->get_constraint_vector(),
                          matrix_free_data_pre->get_quadrature_vector(),
                          matrix_free_data_pre->data);

  // initialize matrix_free
  matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
  matrix_free_data->append(pde_operator);

  matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
  if(application->get_parameters().use_cell_based_face_loops)
    Categorization::do_cell_based_loops(*application->get_grid()->triangulation,
                                        matrix_free_data->data);
  matrix_free->reinit(*application->get_grid()->mapping,
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);


  // setup Navier-Stokes operator
  pde_operator_pre->setup(matrix_free_pre, matrix_free_data_pre);
  pde_operator->setup(matrix_free, matrix_free_data);

  // setup postprocessor
  postprocessor_pre = application->create_postprocessor_precursor();
  postprocessor_pre->setup(*pde_operator_pre);

  postprocessor = application->create_postprocessor();
  postprocessor->setup(*pde_operator);


  // Setup time integrator
  time_integrator_pre = create_time_integrator<dim, Number>(pde_operator_pre,
                                                            application->get_parameters_precursor(),
                                                            mpi_comm,
                                                            is_test,
                                                            postprocessor_pre);

  time_integrator = create_time_integrator<dim, Number>(
    pde_operator, application->get_parameters(), mpi_comm, is_test, postprocessor);

  // setup time integrator before calling setup_solvers (this is necessary since the setup of the
  // solvers depends on quantities such as the time_step_size or gamma0!)
  time_integrator_pre->setup(application->get_parameters_precursor().restarted_simulation);
  time_integrator->setup(application->get_parameters().restarted_simulation);

  // setup solvers
  pde_operator_pre->setup_solvers(time_integrator_pre->get_scaling_factor_time_derivative_term(),
                                  time_integrator_pre->get_velocity());

  pde_operator->setup_solvers(time_integrator->get_scaling_factor_time_derivative_term(),
                              time_integrator->get_velocity());

  timer_tree.insert({"Incompressible flow", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
DriverPrecursor<dim, Number>::solve() const
{
  set_start_time();

  synchronize_time_step_size();

  // time loop
  do
  {
    // advance one time step for precursor domain
    time_integrator_pre->advance_one_timestep();

    // Note that the coupling of both solvers via the inflow boundary conditions is
    // performed in the postprocessing step of the solver for the precursor domain,
    // overwriting the data global structures which are subsequently used by the
    // solver for the actual domain to evaluate the boundary conditions.

    // advance one time step for actual domain
    time_integrator->advance_one_timestep();

    // Both domains have already calculated the new, adaptive time step size individually in
    // function advance_one_timestep(). Here, we have to synchronize the time step size for
    // both domains.
    if(use_adaptive_time_stepping == true)
      synchronize_time_step_size();
  } while(not(time_integrator_pre->finished()) or not(time_integrator->finished()));
}

template<int dim, typename Number>
void
DriverPrecursor<dim, Number>::print_performance_results(double const total_time) const
{
  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;

  // Iterations
  pcout << std::endl
        << "Average number of iterations for incompressible Navier-Stokes solver:" << std::endl;

  pcout << std::endl << "Precursor:" << std::endl;

  time_integrator_pre->print_iterations();

  pcout << std::endl << "Main:" << std::endl;

  time_integrator->print_iterations();

  // Wall times
  pcout << std::endl << "Wall times for incompressible Navier-Stokes solver:" << std::endl;

  timer_tree.insert({"Incompressible flow"}, total_time);

  timer_tree.insert({"Incompressible flow"},
                    time_integrator_pre->get_timings(),
                    "Timeloop precursor");

  timer_tree.insert({"Incompressible flow"}, time_integrator->get_timings(), "Timeloop main");

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Computational costs in CPUh
  unsigned int const N_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

  dealii::Utilities::MPI::MinMaxAvg total_time_data =
    dealii::Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const total_time_avg = total_time_data.avg;

  print_costs(pcout, total_time_avg, N_mpi_processes);

  pcout << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

template class DriverPrecursor<2, float>;
template class DriverPrecursor<3, float>;

template class DriverPrecursor<2, double>;
template class DriverPrecursor<3, double>;

} // namespace IncNS
} // namespace ExaDG
