# Numerical Experiments

This directory contains code to reproduce the numerical experiments
described in the manuscript.

This code is developed with Julia version 1.10.10. To reproduce the
results, start Julia in this directory and execute the following
commands in the Julia REPL to create the figures shown in the paper.

```julia
julia> include("code.jl")

julia> semidiscrete_conservation() # takes a few seconds

julia> fully_discrete_conservation_two_waves() # takes a few seconds

julia> fully_discrete_conservation_one_wave() # takes a few seconds

julia> error_growth_multiple_solitons() # takes a few minutes

julia> error_growth_multiple_solitons_collocation() # takes roughly a minute

julia> change_of_invariants_three_solitons_nls() # takes roughly a minute

julia> error_growth_gray_solitons() # takes roughly a minute

julia> change_of_invariants_gray_solitons() # takes roughly a minute

julia> hyperbolized_nls() # takes roughly a minute

```


## Performance Comparison with Bai et. al. (2024)

For the performance comparison with the work of
[Bai et. al. (2024)](https://doi.org/10.1137/22M152178X), to
perform the simulation with our code:

```julia
julia> include("code.jl") # if you have not done so already

julia> comparison_bai_et_al(semidiscretization = FourierCollocationConserveMassEnergy(),
                            relaxation = MassEnergyRelaxation())
  0.075479 seconds (437 allocations: 419.578 KiB)
┌ Info: Errors at the final time
│   l2_error = 9.60114055621012e-12
└   h1_error = 9.60114055621012e-12

julia> comparison_bai_et_al(semidiscretization = FourierGalerkin(),
                            relaxation = MassEnergyRelaxation())
  0.127541 seconds (458 allocations: 417.938 KiB)
┌ Info: Errors at the final time
│   l2_error = 9.599523042188979e-12
└   h1_error = 9.599523042188979e-12

julia> comparison_bai_et_al(semidiscretization = FourierGalerkin(),
                            relaxation = ProjectionEnergyRelaxation())
  0.137332 seconds (458 allocations: 417.938 KiB)
┌ Info: Errors at the final time
│   l2_error = 1.0509262123032025e-11
└   h1_error = 1.0509262123032025e-11
```

To perform the simulation with their code, clone the repository at
<https://github.com/jiashhu/ME-Conserved-NLS> and install all the required
dependencies. Then

```bash
cd ME-Conserved-NLS/Numerical_Tests
```

Launch Python (or IPython) and enter the following commands:

```python
import NLS_Collo_1d
NLS_Collo_1d.Main(1,40,1024,512,1,2,3,"Err","Standard_Soliton",20,"test",True)
```

The $L_2$ error information is sent to a file, but it can also be printed to the screen
by adding this line to the script `NLS_Collo_1d.py` at line 88:

```python
print(myObj.endt_L2_ex_err_set[-1].real)
```


## Performance Comparison with Andrews and Farrell (2025)

For the performance comparison with the work of
[Andrews and Farrell (2025)](https://doi.org/10.48550/arXiv.2511.23266), to
perform the simulation with our code:

```julia
julia> include("code.jl") # if you have not done so already

julia> comparison_andrews_farrell(N = 100)
  0.146092 seconds (462 allocations: 1.052 MiB)
```

To run their code, install all dependencies (including Firedrake and PETsc).
Download their code from <https://doi.org/10.5281/zenodo.17750373>, unzip
it, and go to that directory. Then

```bash
python benjamin_bona_mahony/avfet.py
```

For the comparison in the paper, we reduced the runtime of their code
somewhat by eliminating most of the information that is printed to stdout.
This can be done by commenting out some lines in the `sp` dict that
manages PETSc output, e.g.,

```python
    "snes_converged_reason"     : None,
    "snes_linesearch_monitor"   : None,
    "snes_monitor"              : None,
    "ksp_monitor_true_residual" : None,
```
