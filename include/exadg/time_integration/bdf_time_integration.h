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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_BDF_TIME_INTEGRATION_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_BDF_TIME_INTEGRATION_H_

// C/C++
#include <vector>

// deal.II
#include <deal.II/base/conditional_ostream.h>

// ExaDG
#include <exadg/time_integration/time_integration_constants_base.h>

namespace ExaDG
{
class BDFTimeIntegratorConstants : public TimeIntegratorConstantsBase
{
public:
  BDFTimeIntegratorConstants(unsigned int const order, bool const start_with_low_order);

  double
  get_gamma0() const;

  double
  get_alpha(unsigned int const i) const;

  void
  print(dealii::ConditionalOStream & pcout) const final;

private:
  void
  set_constant_time_step(unsigned int const current_order) final;

  void
  set_adaptive_time_step(unsigned int const          current_order,
                         std::vector<double> const & time_steps) final;

  /*
   *  BDF time integrator constants:
   *
   *  du/dt = (gamma_0 u^{n+1} - alpha_0 u^{n} - alpha_1 u^{n-1} ... - alpha_{J-1} u^{n-J+1})/dt
   */
  double gamma0;

  std::vector<double> alpha;
};

/*
 * Calculates the time derivative
 *
 *  derivative = du/dt = (gamma_0 u^{n+1} - alpha_0 u^{n} - alpha_1 u^{n-1} ... - alpha_{J-1}
 * u^{n-J+1})/dt
 */
template<typename VectorType>
void
compute_bdf_time_derivative(VectorType &                       derivative,
                            VectorType const &                 solution_np,
                            std::vector<VectorType> const &    previous_solutions,
                            BDFTimeIntegratorConstants const & bdf,
                            double const &                     time_step_size)
{
  derivative.equ(bdf.get_gamma0() / time_step_size, solution_np);

  for(unsigned int i = 0; i < previous_solutions.size(); ++i)
    derivative.add(-bdf.get_alpha(i) / time_step_size, previous_solutions[i]);
}

} // namespace ExaDG

#endif /* INCLUDE_EXADG_TIME_INTEGRATION_BDF_TIME_INTEGRATION_H_ */
