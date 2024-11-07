# https://github.com/GRTLCollaboration/GRChombo

```console
Tests/BSSNMatterTest/BSSNMatterTest.cpp:    CCZ4_params_t<MovingPunctureGauge::params_t> params;
Tests/BSSNMatterTest/BSSNMatterTest.cpp:    BoxLoops::loop(MatterCCZ4RHS<ScalarFieldWithPotential, MovingPunctureGauge,
Tests/CCZ4Test/CCZ4Test.cpp:    CCZ4_params_t<MovingPunctureGauge::params_t> params;
Tests/CCZ4Test/CCZ4Test.cpp:        CCZ4RHS<MovingPunctureGauge, FourthOrderDerivatives>(params, dx, sigma),
Examples/BinaryBH/BinaryBHLevel.cpp:        BoxLoops::loop(CCZ4RHS<MovingPunctureGauge, FourthOrderDerivatives>(
Examples/BinaryBH/BinaryBHLevel.cpp:        BoxLoops::loop(CCZ4RHS<MovingPunctureGauge, SixthOrderDerivatives>(
Examples/KerrBH/KerrBHLevel.cpp:        BoxLoops::loop(CCZ4RHS<MovingPunctureGauge, FourthOrderDerivatives>(
Examples/KerrBH/KerrBHLevel.cpp:        BoxLoops::loop(CCZ4RHS<MovingPunctureGauge, SixthOrderDerivatives>(
Examples/ScalarField/ScalarFieldLevel.cpp:        MatterCCZ4RHS<ScalarFieldWithPotential, MovingPunctureGauge,
Examples/ScalarField/ScalarFieldLevel.cpp:        MatterCCZ4RHS<ScalarFieldWithPotential, MovingPunctureGauge,
Source/Matter/MatterCCZ4RHS.hpp:#include "MovingPunctureGauge.hpp"
Source/Matter/MatterCCZ4RHS.hpp:template <class matter_t, class gauge_t = MovingPunctureGauge,
Source/Matter/MatterCCZ4.hpp:#include "MovingPunctureGauge.hpp"
Source/Matter/MatterCCZ4.hpp:template <class matter_t, class gauge_t = MovingPunctureGauge,
Source/CCZ4/MovingPunctureGauge.hpp:#ifndef MOVINGPUNCTUREGAUGE_HPP_
Source/CCZ4/MovingPunctureGauge.hpp:#define MOVINGPUNCTUREGAUGE_HPP_
Source/CCZ4/MovingPunctureGauge.hpp:class MovingPunctureGauge
Source/CCZ4/MovingPunctureGauge.hpp:    MovingPunctureGauge(const params_t &a_params) : m_params(a_params) {}
Source/CCZ4/MovingPunctureGauge.hpp:#endif /* MOVINGPUNCTUREGAUGE_HPP_ */
Source/CCZ4/IntegratedMovingPunctureGauge.hpp:#ifndef INTEGRATEDMOVINGPUNCTUREGAUGE_HPP_
Source/CCZ4/IntegratedMovingPunctureGauge.hpp:#define INTEGRATEDMOVINGPUNCTUREGAUGE_HPP_
Source/CCZ4/IntegratedMovingPunctureGauge.hpp:#include "MovingPunctureGauge.hpp"
Source/CCZ4/IntegratedMovingPunctureGauge.hpp:class IntegratedMovingPunctureGauge
Source/CCZ4/IntegratedMovingPunctureGauge.hpp:    using params_t = MovingPunctureGauge::params_t;
Source/CCZ4/IntegratedMovingPunctureGauge.hpp:    IntegratedMovingPunctureGauge(const params_t &a_params) : m_params(a_params)
Source/CCZ4/IntegratedMovingPunctureGauge.hpp:    // BoxLoops::loop(IntegratedMovingPunctureGauge(m_p.ccz4_params),
Source/CCZ4/IntegratedMovingPunctureGauge.hpp:#endif /* INTEGRATEDMOVINGPUNCTUREGAUGE_HPP_ */
Source/CCZ4/CCZ4.hpp:#include "MovingPunctureGauge.hpp"
Source/CCZ4/CCZ4.hpp:template <class gauge_params_t = MovingPunctureGauge::params_t>
Source/CCZ4/CCZ4.hpp:template <class gauge_t = MovingPunctureGauge,
Source/CCZ4/CCZ4RHS.hpp:#include "MovingPunctureGauge.hpp"
Source/CCZ4/CCZ4RHS.hpp:template <class gauge_params_t = MovingPunctureGauge::params_t>
Source/CCZ4/CCZ4RHS.hpp:template <class gauge_t = MovingPunctureGauge,
Source/GRChomboCore/GRAMRLevel.cpp:    Vector<int> procMap;
Source/GRChomboCore/GRAMRLevel.cpp:    LoadBalance(procMap, a_grids);
Source/GRChomboCore/GRAMRLevel.cpp:            pout() << igrid << ": " << procMap[igrid] << "  " << endl;
Source/GRChomboCore/GRAMRLevel.cpp:    DisjointBoxLayout dbl(a_grids, procMap, m_problem_domain);
Source/GRChomboCore/SimulationParametersBase.hpp:    // Note the gauge parameters are specific to MovingPunctureGauge

```
