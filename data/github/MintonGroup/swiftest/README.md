# https://github.com/MintonGroup/swiftest

```console
paper/paper.bib:	title = {{GENGA}. {II}. {GPU} {Planetary} {N}-body {Simulations} with {Non}-{Newtonian} {Forces} and {High} {Number} of {Particles}},
paper/paper.bib:	title = {{QYMSYM}: {A} {GPU}-accelerated hybrid symplectic integrator that permits close encounters},
paper/paper.md:The `SyMBA` integrator included in `Swiftest` is most similar to the hybrid symplectic integrator `MERCURY6` [@Chambers:1999], the `MERCURIUS` integrator of `REBOUND` [@Rein:2012;@Rein:2015], and the GPU-enabled hybrid symplectic integrators such as  `QYMSYM` [@Moore:2011] and `GENGA II` [@Grimm:2022], with some important distinctions. The hybrid symplectic integrators typically employ a symplectic method, such as the original WHM method in Jacobi coordinates or the modified method that uses the Democratic Heliocentric coordinates, only when bodies are far from each other relative to their gravitational spheres of influence (some multiple of a Hill's sphere). When bodies approach each other, the integration is smoothly switched to a non-symplectic method, such as Bulirsch-Stoer or IAS15. In contrast, `SyMBA` is a multi-step method that recursively subdivides the step size of bodies undergoing close approach with each other. 

```
