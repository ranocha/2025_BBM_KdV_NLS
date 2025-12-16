# Install packages
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Load packages
using LinearAlgebra: LinearAlgebra, Diagonal, UniformScaling, I, det, diag, diagind, mul!, ldiv!, lu, lu!, norm
using SparseArrays: SparseArrays, sparse, issparse, dropzeros!
using Printf: @sprintf

using StaticArrays: StaticArrays, @SMatrix, SVector

using SimpleNonlinearSolve

using SummationByPartsOperators
using SummationByPartsOperators: xmin, xmax

using TowerOfEnzyme: nth_derivative # for AD to compute N-solitons of KdV nicely
using JacobiElliptic: JacobiElliptic

using LaTeXStrings
using CairoMakie
# using GLMakie
set_theme!(theme_latexfonts();
           fontsize = 26,
           linewidth = 3,
           markersize = 16,
           Lines = (cycle = Cycle([:color, :linestyle], covary = true),),
           Scatter = (cycle = Cycle([:color, :marker], covary = true),))


const FIGDIR = joinpath(dirname(@__DIR__), "figures")
if !isdir(FIGDIR)
    mkdir(FIGDIR)
end


#####################################################################
# Utility functions

function compute_eoc(Ns, errors)
    eoc = similar(errors)
    eoc[begin] = NaN # no EOC defined for the first grid
    for idx in Iterators.drop(eachindex(errors, Ns, eoc), 1)
        eoc[idx] = -( log(errors[idx] / errors[idx - 1]) /
                      log(Ns[idx] / Ns[idx - 1]) )
    end
    return eoc
end

function fit_order(Ns, errors)
    A = hcat(ones(length(Ns)), log.(Ns))
    c = A \ log.(errors)
    return -c[2]
end

change(x) = x .- first(x)
function relative_change(x)
    return (x .- first(x)) ./ abs(first(x))
end
function absolute_change(x)
    y = abs.(x .- first(x))
    e = eps(eltype(y))
    @. y = max(y, e)
    return y
end

"""
    dealiased_number_of_nodes(p, N)

Given `N` nodes in physical space, this function computes a
number of nodes `M >= N` that is sufficient to de-alias a
polynomial degree `p` nonlinearity. The resulting number of
nodes `M` is chosen such that an FFT can be performed with
reasonable efficiency.
"""
function dealiased_number_of_nodes(p::Integer, N::Integer)
    if !(p >= 1 && N >= 1)
        throw(ArgumentError("p and N must be positive; got p = $p, N = $N"))
    end

    # For a quadratic nonlinearity and a general quadratic nonlinearity,
    # Kopriva (2009, Implementing Spectral Methods, Section 4.3.2)
    # shows that one must use M > 3 N / 2 nodes, assuming that both N
    # and M are even.
    # For a general polynomial degree p nonlinearity, this becomes
    # M > (p + 1) N / 2.
    if iseven(N)
        M_min = (p + 1) * (N ÷ 2) + 1
    else
        M_min = (p + 1) * ((N + 1) ÷ 2) + 1
    end

    if isodd(M_min)
        M_min += 1
    end

    # For efficiency, we choose M to be a product of small primes.
    primes = (2, 3, 5, 7)
    M = nextprod(primes, M_min)

    return M
end



#####################################################################
# High-level interface of the equations and IMEX ode solver

rhs_stiff!(du, u, parameters, t) = rhs_stiff!(du, u, parameters.equation, parameters.semidiscretization, parameters, t)
rhs_nonstiff!(du, u, parameters, t) = rhs_nonstiff!(du, u, parameters.equation, parameters.semidiscretization, parameters, t)
operator(rhs_stiff!, parameters) = operator(rhs_stiff!, parameters.equation, parameters.semidiscretization, parameters)
mass(q, parameters) = mass(q, parameters.equation, parameters.semidiscretization, parameters)
momentum(q, parameters) = momentum(q, parameters.equation, parameters.semidiscretization, parameters)
energy(q, parameters) = energy(q, parameters.equation, parameters.semidiscretization, parameters)
mass_momentum(q, parameters) = mass_momentum(q, parameters.equation, parameters.semidiscretization, parameters)


# IMEX Coefficients
"""
    ARS111(T = Float64)

A first-order, globally stiffly accurate (GSA) type II IMEX method developed by
Ascher, Ruuth, and Spiteri (1997).

## References

- Uri M. Ascher, Steven J. Ruuth, and Raymond J Spiteri (1997)
  Implicit-explicit Runge-Kutta methods for time-dependent
  partial differential equations.
  [DOI: 10.1016/S0168-9274(97)00056-1](https://doi.org/10.1016/S0168-9274(97)00056-1)
- Sebastiano Boscarino and Giovanni Russo (2024)
  Asymptotic preserving methods for quasilinear hyperbolic systems with
  stiff relaxation: a review.
  [DOI: 10.1007/s40324-024-00351-x](https://doi.org/10.1007/s40324-024-00351-x)
- Sebastiano  Boscarino, Lorenzo Pareschi, and Giovanni Russo (2025)
  Implicit-explicit methods for evolutionary partial differential equations.
  [DOI: 10.1137/1.9781611978209](https://doi.org/10.1137/1.9781611978209)
"""
struct ARS111{T} end
ARS111(T = Float64) = ARS111{T}()
function coefficients(::ARS111{T}) where T
    l = one(T)

    A_stiff = [0 0;
               0 l]
    b_stiff = [0, l]
    c_stiff = [0, l]
    A_nonstiff = [0 0;
                  l 0]
    b_nonstiff = [l, 0]
    c_nonstiff = [0, l]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end


"""
    ARS222(T = Float64)

A second-order, globally stiffly accurate (GSA) type II IMEX method developed by
Ascher, Ruuth, and Spiteri (1997).

## References

- Uri M. Ascher, Steven J. Ruuth, and Raymond J Spiteri (1997)
  Implicit-explicit Runge-Kutta methods for time-dependent
  partial differential equations.
  [DOI: 10.1016/S0168-9274(97)00056-1](https://doi.org/10.1016/S0168-9274(97)00056-1)
- Sebastiano Boscarino and Giovanni Russo (2024)
  Asymptotic preserving methods for quasilinear hyperbolic systems with
  stiff relaxation: a review.
  [DOI: 10.1007/s40324-024-00351-x](https://doi.org/10.1007/s40324-024-00351-x)
- Sebastiano  Boscarino, Lorenzo Pareschi, and Giovanni Russo (2025)
  Implicit-explicit methods for evolutionary partial differential equations.
  [DOI: 10.1137/1.9781611978209](https://doi.org/10.1137/1.9781611978209)
"""
struct ARS222{T} end
ARS222(T = Float64) = ARS222{T}()
function coefficients(::ARS222{T}) where T
    two = convert(T, 2)
    γ = 1 - 1 / sqrt(two)
    δ = 1 - 1 / (2 * γ)

    A_stiff = [0 0 0;
               0 γ 0;
               0 1-γ γ]
    b_stiff = [0, 1-γ, γ]
    c_stiff = [0, γ, 1]
    A_nonstiff = [0 0 0;
                  γ 0 0;
                  δ 1-δ 0]
    b_nonstiff = [δ, 1-δ, 0]
    c_nonstiff = [0, γ, 1]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

"""
    ARS443(T = Float64)

A third-order, globally stiffly accurate (GSA) type II IMEX method developed by
Ascher, Ruuth, and Spiteri (1997).

## References

- Uri M. Ascher, Steven J. Ruuth, and Raymond J Spiteri (1997)
  Implicit-explicit Runge-Kutta methods for time-dependent
  partial differential equations.
  [DOI: 10.1016/S0168-9274(97)00056-1](https://doi.org/10.1016/S0168-9274(97)00056-1)
- Sebastiano Boscarino and Giovanni Russo (2024)
  Asymptotic preserving methods for quasilinear hyperbolic systems with
  stiff relaxation: a review.
  [DOI: 10.1007/s40324-024-00351-x](https://doi.org/10.1007/s40324-024-00351-x)
- Sebastiano  Boscarino, Lorenzo Pareschi, and Giovanni Russo (2025)
  Implicit-explicit methods for evolutionary partial differential equations.
  [DOI: 10.1137/1.9781611978209](https://doi.org/10.1137/1.9781611978209)
"""
struct ARS443{T} end
ARS443(T = Float64) = ARS443{T}()
function coefficients(::ARS443{T}) where T
    l = one(T)

    A_stiff = [0 0 0 0 0;
               0 l/2 0 0 0;
               0 l/6 l/2 0 0;
               0 -l/2 l/2 l/2 0;
               0 3*l/2 -3*l/2 l/2 l/2]
    b_stiff = [0, 3*l/2, -3*l/2, l/2, l/2]
    c_stiff = [0, l/2, 2*l/3, l/2, l]
    A_nonstiff = [0 0 0 0 0;
                  l/2 0 0 0 0;
                  11*l/18 l/18 0 0 0;
                  5*l/6 -5*l/6 l/2 0 0;
                  l/4 7*l/4 3*l/4 -7*l/4 0]
    b_nonstiff = [l/4 7*l/4 3*l/4 -7*l/4 0]
    c_nonstiff = [0, l/2, 2*l/3, l/2, l]
    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

"""
    ARS443explicit(T = Float64)

Explicit part of the third-order, type-II, globally stiffly accurate method of Ascher, Ruuth, and Spiteri (1997).
"""
struct ARS443explicit{T} end
ARS443explicit(T = Float64) = ARS443explicit{T}()
function coefficients(::ARS443explicit{T}) where T
    l = one(T)

    A_nonstiff = [0 0 0 0 0;
                  l/2 0 0 0 0;
                  11*l/18 l/18 0 0 0;
                  5*l/6 -5*l/6 l/2 0 0;
                  l/4 7*l/4 3*l/4 -7*l/4 0]
    b_nonstiff = [l/4 7*l/4 3*l/4 -7*l/4 0]
    c_nonstiff = [0, l/2, 2*l/3, l/2, l]

    A_stiff = copy(A_nonstiff)
    b_stiff = copy(b_nonstiff)
    c_stiff = copy(c_nonstiff)

    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

"""
    KenCarpARK437(T = Float64)

A fourth-order type II IMEX method developed by Kennedy and Carpenter (2019).
The implicit method is stiffly accurate.

## References

- Christopher A. Kennedy  and Mark H. Carpenter (2019)
  Higher-order additive Runge–Kutta schemes for ordinary differential equations.
  [DOI: 10.1016/j.apnum.2018.10.007](https://doi.org/10.1016/j.apnum.2018.10.007)
"""
struct KenCarpARK437{T} end
KenCarpARK437(T = Float64) = KenCarpARK437{T}()
function coefficients(::KenCarpARK437{T}) where T
    γ = T(1235) / T(10_000)

    c2 = T(247) / T(1000)
    c3 = T(4276536705230) / T(10142255878289)
    c4 = T(67) / T(200)
    c5 = T(3) / T(40)
    c6 = T(7) / T(10)
    c7 = T(1)
    c_nonstiff = [0, c2, c3, c4, c5, c6, c7]
    b1 = T(0)
    b2 = b1
    b3 = T(9164257142617) / T(17756377923965)
    b4 = T(-10812980402763) / T(74029279521829)
    b5 = T(1335994250573) / T(5691609445217)
    b6 = T(2273837961795) / T(8368240463276)
    b7 = T(247) / T(2000)
    b_nonstiff = [b1, b2, b3, b4, b5, b6, b7]
    a21 = c2
    a31 = T(247) / T(4000)
    a32 = T(2694949928731) / T(7487940209513)
    a41 = T(464650059369) / T(8764239774964)
    a42 = T(878889893998) / T(2444806327765)
    a43 = T(-952945855348) / T(12294611323341)
    a51 = T(476636172619) / T(8159180917465)
    a52 = T(-1271469283451) / T(7793814740893)
    a53 = T(-859560642026) / T(4356155882851)
    a54 = T(1723805262919) / T(4571918432560)
    a61 = T(6338158500785) / T(11769362343261)
    a62 = T(-4970555480458) / T(10924838743837)
    a63 = T(3326578051521) / T(2647936831840)
    a64 = T(-880713585975) / T(1841400956686)
    a65 = T(-1428733748635) / T(8843423958496)
    a71 = T(760814592956) / T(3276306540349)
    a72 = a71
    a73 = T(-47223648122716) / T(6934462133451)
    a74 = T(71187472546993) / T(9669769126921)
    a75 = T(-13330509492149) / T(9695768672337)
    a76 = T(11565764226357) / T(8513123442827)
    A_nonstiff = [0 0 0 0 0 0 0;
                  a21 0 0 0 0 0 0;
                  a31 a32 0 0 0 0 0;
                  a41 a42 a43 0 0 0 0;
                  a51 a52 a53 a54 0 0 0;
                  a61 a62 a63 a64 a65 0 0;
                  a71 a72 a73 a74 a75 a76 0]
    @assert c_nonstiff ≈ sum(A_nonstiff, dims = 2)

    a21 = γ
    a31 = T(624185399699) / T(4186980696204)
    a32 = a31
    a41 = T(1258591069120) / T(10082082980243)
    a42 = a41
    a43 = T(-322722984531) / T(8455138723562)
    a51 = T(-436103496990) / T(5971407786587)
    a52 = a51
    a53 = T(-2689175662187) / T(11046760208243)
    a54 = T(4431412449334) / T(12995360898505)
    a61 = T(-2207373168298) / T(14430576638973)
    a62 = a61
    a63 = T(242511121179) / T(3358618340039)
    a64 = T(3145666661981) / T(7780404714551)
    a65 = T(5882073923981) / T(14490790706663)
    a71 = T(0)
    a72 = a71
    a73 = T(9164257142617) / T(17756377923965)
    a74 = T(-10812980402763) / T(74029279521829)
    a75 = T(1335994250573) / T(5691609445217)
    a76 = T(2273837961795) / T(8368240463276)
    A_stiff = [0 0 0 0 0 0 0;
               a21 γ 0 0 0 0 0;
               a31 a32 γ 0 0 0 0;
               a41 a42 a43 γ 0 0 0;
               a51 a52 a53 a54 γ 0 0;
               a61 a62 a63 a64 a65 γ 0;
               a71 a72 a73 a74 a75 a76 γ]
    b_stiff = copy(b_nonstiff)
    c_stiff = copy(c_nonstiff)
    @assert c_stiff ≈ sum(A_stiff, dims = 2)

    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

"""
    KenCarpARK548(T = Float64)

A fifth-order type II IMEX method developed by Kennedy and Carpenter (2019).
The implicit method is stiffly accurate.

## References

- Christopher A. Kennedy  and Mark H. Carpenter (2019)
  Higher-order additive Runge–Kutta schemes for ordinary differential equations.
  [DOI: 10.1016/j.apnum.2018.10.007](https://doi.org/10.1016/j.apnum.2018.10.007)
"""
struct KenCarpARK548{T} end
KenCarpARK548(T = Float64) = KenCarpARK548{T}()
function coefficients(::KenCarpARK548{T}) where T
    γ = T(2) / T(9)

    c2 = T(4) / T(9)
    c3 = T(6456083330201) / T(8509243623797)
    c4 = T(1632083962415) / T(14158861528103)
    c5 = T(6365430648612) / T(17842476412687)
    c6 = T(18) / T(25)
    c7 = T(191) / T(200)
    c8 = T(1)
    c_nonstiff = [0, c2, c3, c4, c5, c6, c7, c8]
    b1 = T(0)
    b2 = b1
    b3 = T(3517720773327) / T(20256071687669)
    b4 = T(4569610470461) / T(17934693873752)
    b5 = T(2819471173109) / T(11655438449929)
    b6 = T(3296210113763) / T(10722700128969)
    b7 = T(-1142099968913) / T(5710983926999)
    b8 = γ
    b_nonstiff = [b1, b2, b3, b4, b5, b6, b7, b8]
    a21 = c2
    a31 = T(1) / T(9)
    a32 = T(1183333538310) / T(1827251437969)
    a41 = T(895379019517) / T(9750411845327)
    a42 = T(477606656805) / T(13473228687314)
    a43 = T(-112564739183) / T(9373365219272)
    a51 = T(-4458043123994) / T(13015289567637)
    a52 = T(-2500665203865) / T(9342069639922)
    a53 = T(983347055801) / T(8893519644487)
    a54 = T(2185051477207) / T(2551468980502)
    a61 = T(-167316361917) / T(17121522574472)
    a62 = T(1605541814917) / T(7619724128744)
    a63 = T(991021770328) / T(13052792161721)
    a64 = T(2342280609577) / T(11279663441611)
    a65 = T(3012424348531) / T(12792462456678)
    a71 = T(6680998715867) / T(14310383562358)
    a72 = T(5029118570809) / T(3897454228471)
    a73 = T(2415062538259) / T(6382199904604)
    a74 = T(-3924368632305) / T(6964820224454)
    a75 = T(-4331110370267) / T(15021686902756)
    a76 = T(-3944303808049) / T(11994238218192)
    a81 = T(2193717860234) / T(3570523412979)
    a82 = a81
    a83 = T(5952760925747) / T(18750164281544)
    a84 = T(-4412967128996) / T(6196664114337)
    a85 = T(4151782504231) / T(36106512998704)
    a86 = T(572599549169) / T(6265429158920)
    a87 = T(-457874356192) / T(11306498036315)
    A_nonstiff = [0 0 0 0 0 0 0 0;
                  a21 0 0 0 0 0 0 0;
                  a31 a32 0 0 0 0 0 0;
                  a41 a42 a43 0 0 0 0 0;
                  a51 a52 a53 a54 0 0 0 0;
                  a61 a62 a63 a64 a65 0 0 0;
                  a71 a72 a73 a74 a75 a76 0 0;
                  a81 a82 a83 a84 a85 a86 a87 0]
    @assert c_nonstiff ≈ sum(A_nonstiff, dims = 2)

    a21 = γ
    a31 = T(2366667076620) / T(8822750406821)
    a32 = a31
    a41 = T(-257962897183) / T(4451812247028)
    a42 = a41
    a43 = T(128530224461) / T(14379561246022)
    a51 = T(-486229321650) / T(11227943450093)
    a52 = a51
    a53 = T(-225633144460) / T(6633558740617)
    a54 = T(1741320951451) / T(6824444397158)
    a61 = T(621307788657) / T(4714163060173)
    a62 = a61
    a63 = T(-125196015625) / T(3866852212004)
    a64 = T(940440206406) / T(7593089888465)
    a65 = T(961109811699) / T(6734810228204)
    a71 = T(2036305566805) / T(6583108094622)
    a72 = a71
    a73 = T(-3039402635899) / T(4450598839912)
    a74 = T(-1829510709469) / T(31102090912115)
    a75 = T(-286320471013) / T(6931253422520)
    a76 = T(8651533662697) / T(9642993110008)
    a81 = b1
    a82 = b2
    a83 = b3
    a84 = b4
    a85 = b5
    a86 = b6
    a87 = b7
    A_stiff = [0 0 0 0 0 0 0 0;
               a21 γ 0 0 0 0 0 0;
               a31 a32 γ 0 0 0 0 0;
               a41 a42 a43 γ 0 0 0 0;
               a51 a52 a53 a54 γ 0 0 0;
               a61 a62 a63 a64 a65 γ 0 0;
               a71 a72 a73 a74 a75 a76 γ 0;
               a81 a82 a83 a84 a85 a86 a87 γ]
    b_stiff = copy(b_nonstiff)
    c_stiff = copy(c_nonstiff)
    @assert c_stiff ≈ sum(A_stiff, dims = 2)

    return A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff
end

abstract type AbstractProjection end
"""
    NoProjection()

Do not use any relaxation/projection methods, just the baseline
time integration method.
"""
struct NoProjection <: AbstractProjection end
"""
    MassProjection()

Use a projection method that enforces mass conservation
(for equations where the total mass is a nonlinear invariant).
"""
struct MassProjection <: AbstractProjection end
"""
    MassRelaxation()

Use a relaxation method that enforces mass conservation
(for equations where the total mass is a nonlinear invariant).
"""
struct MassRelaxation <: AbstractProjection end
"""
    MassMomentumProjection()

Use a projection method that enforces mass and momentum conservation.
"""
struct MassMomentumProjection <: AbstractProjection end
"""
    MassMomentumRelaxation()

Use a relaxation method that enforces mass and momentum conservation
(for equations where the total mass is a linear invariant).
"""
struct MassMomentumRelaxation <: AbstractProjection end
"""
    MassMomentumRelaxation()

Use a relaxation method that enforces mass and energy conservation
for equations where the total mass is a linear invariant.
If the mass is a nonlinear invariant, this method first projects
to conserve mass and then relaxes to conserve energy.
"""
struct MassEnergyRelaxation <: AbstractProjection end
"""
    ProjectionEnergyRelaxation()

Use a relaxation method that enforces mass, momentum, and energy
conservation. Before the relaxation search, a projection is performed
to conserve the mass and momentum (like in the `MassMomentumProjection`).
"""
struct ProjectionEnergyRelaxation <: AbstractProjection end
"""
    RelaxationEnergyRelaxation()

Use a relaxation method that enforces mass, momentum, and energy
conservation. Before the relaxation search, a relaxation is performed
to conserve the mass and momentum (like in the `MassMomentumRelaxation`).
"""
struct RelaxationEnergyRelaxation <: AbstractProjection end

# IMEX ARK solver
# This assumes that the stiff part is linear and that the stiff solver is
# diagonally implicit.
function solve_imex(rhs_stiff!, rhs_stiff_operator, rhs_nonstiff!,
                    q0, tspan, parameters, alg;
                    dt,
                    relaxation::AbstractProjection = NoProjection(),
                    relaxation_alg = SimpleKlement(),
                    relaxation_tol = 1.0e-12,
                    callback = Returns(nothing),
                    save_everystep = false)
    A_stiff, b_stiff, c_stiff, A_nonstiff, b_nonstiff, c_nonstiff = coefficients(alg)

    s = length(b_stiff)
    @assert size(A_stiff, 1) == s && size(A_stiff, 2) == s &&
            length(b_stiff) == s && length(c_stiff) == s &&
            size(A_nonstiff, 1) == s && size(A_nonstiff, 2) == s &&
            length(b_nonstiff) == s && length(c_nonstiff) == s
    Base.require_one_based_indexing(A_stiff, b_stiff, c_stiff,
                                    A_nonstiff, b_nonstiff, c_nonstiff)

    q = copy(q0) # solution
    if save_everystep
        sol_q = [copy(q0)]
        sol_t = [first(tspan)]
    end
    y = similar(q) # stage value
    z = similar(q) # stage update value
    t = first(tspan)
    tmp = similar(q)
    k_stiff_q = similar(q) # derivative of the previous state
    k_stiff = Vector{typeof(q)}(undef, s) # stage derivatives
    k_nonstiff = Vector{typeof(q)}(undef, s) # stage derivatives
    for i in 1:s
        k_stiff[i] = similar(q)
        k_nonstiff[i] = similar(q)
    end

    # Setup system matrix template and factorizations
    W, factorization, factorizations = let
        a = findfirst(!iszero, diag(A_stiff))
        if isnothing(a)
            factor = zero(dt)
        else
            factor = a * dt
        end
        W = I - factor * rhs_stiff_operator

        if W isa UniformScaling
            # This happens if the stiff part is zero
            factorization = W
        elseif W isa SummationByPartsOperators.AbstractPeriodicDerivativeOperator
            factorization = W
        elseif W isa AbstractStiffOperator
            factorization = W
        else
            factorization = lu(W)
        end

        # We cache the factorizations for different factors for efficiency.
        # Since we do not use adaptive time stepping, we will only have a few
        # different factors.
        if factorization isa SummationByPartsOperators.AbstractPeriodicDerivativeOperator
            factorizations = nothing
        elseif factorization isa AbstractStiffOperator
            factorizations = nothing
        else
            factorizations = Dict(factor => copy(factorization))
        end

        W, factorization, factorizations
    end

    # used for relaxation
    mass_old = mass(q, parameters)
    momentum_old = momentum(q, parameters)
    energy_old = energy(q, parameters)

    while t < last(tspan)
        dt = min(dt, last(tspan) - t)

        # There are two possible formulations of a diagonally implicit RK method.
        # The "simple" one is
        #   y_i = q + h \sum_{j=1}^{i} a_{ij} f(y_j)
        # However, it can be better to use the smaller values
        #   z_i = (y_i - q) / h
        # so that the stage equations become
        #   q + h z_i = q + h \sum_{j=1}^{i} a_{ij} f(q + h z_j)
        # ⟺
        #   z_i - h a_{ii} f(q + h z_i) = \sum_{j=1}^{i-1} a_{ij} f(q + h z_j)
        # For a linear problem f(q) = T q, this becomes
        #   (I - h a_{ii} T z_i = a_{ii} T q + \sum_{j=1}^{i-1} a_{ij} T(q + h z_j)
        # We use this formulation and also avoid evaluating the stiff operator at
        # the numerical solutions (due to the large Lipschitz constant), but instead
        # rearrange the equation to obtain the required stiff RHS values as
        #   T(q + h z_i) = a_{ii}^{-1} (z_i - \sum_{j=1}^{i-1} a_{ij} f(q + h z_j))
        rhs_stiff!(k_stiff_q, q, parameters, t)

        # Compute stages
        for i in 1:s
            # RHS of linear system
            fill!(tmp, 0)
            for j in 1:(i - 1)
                @. tmp += A_stiff[i, j] * k_stiff[j] + A_nonstiff[i, j] * k_nonstiff[j]
            end
            # The right-hand side of the linear system formulated using the stages y_i
            # instead of the stage updates z_i would be
            #   @. tmp = q + dt * tmp
            # By using the stage updates z_i, we avoid the possibly different scales
            # for small dt.
            @. tmp = A_stiff[i, i] * k_stiff_q + tmp

            # Setup and solve linear system
            if iszero(rhs_stiff_operator) || iszero(A_stiff[i, i])
                copyto!(z, tmp)
            else
                factor = A_stiff[i, i] * dt

                if factorization isa SummationByPartsOperators.AbstractPeriodicDerivativeOperator
                    W = I - factor * rhs_stiff_operator
                    F = W
                    ldiv!(z, F, tmp)
                elseif factorization isa StiffOperatorKdV
                    @assert parameters.equation isa KdV
                    (; equation, D_small) = parameters
                    # We compute the discrete derivative operator on the smaller grid.
                    # We do not use the special handling of the highest frequency mode
                    # for an even number of grid points since we interpret the spectral
                    # coefficients as truncated from a larger grid.
                    for i in eachindex(tmp)
                        f = 1 - factor * (i - 1)^3 * im * D_small.jac^3 * size(D_small, 2)^3
                        z[i] = tmp[i] / f
                    end
                elseif factorization isa StiffOperatorCubicNLS
                    @assert parameters.equation isa CubicNLS
                    if parameters.semidiscretization isa FourierGalerkin
                        (; equation, D_small) = parameters
                        a = real(tmp, equation)
                        b = imag(tmp, equation)
                        z1 = real(z, equation)
                        z2 = imag(z, equation)
                        # We compute the discrete derivative operator on the smaller grid.
                        for i in eachindex(a)
                            f = -(i - 1)^2 * D_small.jac^2 * size(D_small, 2)^2
                            # Solve for z2
                            num = b[i] + factor * f * a[i]
                            den = 1 + factor^2 * f^2
                            z2[i] = num / den
                            # Solve for z1
                            z1[i] = a[i] - factor * f * z2[i]
                        end
                    elseif parameters.semidiscretization isa FourierCollocation
                        (; equation, D2, tmp1) = parameters
                        a = real(tmp, equation)
                        b = imag(tmp, equation)
                        z1 = real(z, equation)
                        z2 = imag(z, equation)
                        mul!(tmp1, D2, a)
                        @. tmp1 = b + factor * tmp1
                        ldiv!(z2, I + factor^2 * D2 * D2, tmp1)
                        mul!(tmp1, D2, z2)
                        @. z1 = a - factor * tmp1
                    else
                        error("Unknown semidiscretization $(parameters.semidiscretization) for CubicNLS")
                    end
                elseif factorization isa StiffOperatorHyperbolizedCubicNLS
                    @assert parameters.equation isa HyperbolizedCubicNLS
                    @assert parameters.semidiscretization isa FourierGalerkin
                    (; equation, D_small) = parameters
                    (; τ ) = equation

                    tmp0 = get_qi(tmp, equation, 0)
                    tmp1 = get_qi(tmp, equation, 1)
                    tmp2 = get_qi(tmp, equation, 2)
                    tmp3 = get_qi(tmp, equation, 3)

                    z0 = get_qi(z, equation, 0)
                    z1 = get_qi(z, equation, 1)
                    z2 = get_qi(z, equation, 2)
                    z3 = get_qi(z, equation, 3)

                    # We have to solve the system
                    #
                    # [z0, z1, z2, z3]
                    # = [I 0 0 factor*D
                    #    0 I -factor*D 0
                    #    0 -factor/τ*D I factor/τ*I
                    #    factor/τ*D 0 -factor/τ*I I] *
                    #    [tmp0, tmp1, tmp2, tmp3]
                    #
                    # We do this by Gaussian elimination. First, we obtain
                    #
                    #   [I 0 0 factor*D
                    #    0 I -factor*D 0
                    #    0 0 I-factor^2/τ^2*D factor/τ*I
                    #    0 0 -factor/τ*I I-factor^2/τ^2*D]
                    #
                    # Then, we solve the 2x2 block system for z2 and z3, and
                    # finally compute z0 and z1.

                    for i in eachindex(z3)
                        factor_D = factor * (i - 1) * im * D_small.jac * size(D_small, 2)
                        factor_D_over_τ = factor_D / τ
                        factor_over_τ = factor / τ
                        id_minus_factor2_D2_over_τ = 1 + factor^2 * (i - 1)^2 * D_small.jac^2 * size(D_small, 2)^2 / τ

                        # Compute z3
                        z3[i] = (factor_over_τ * (tmp2[i] + factor_D_over_τ * tmp1[i]) + id_minus_factor2_D2_over_τ * (tmp3[i] - factor_D_over_τ * tmp0[i])) / (factor_over_τ^2 + id_minus_factor2_D2_over_τ^2)

                        # Compute z2
                        z2[i] = (id_minus_factor2_D2_over_τ * (tmp2[i] + factor_D_over_τ * tmp1[i]) - factor_over_τ * (tmp3[i] - factor_D_over_τ * tmp0[i])) / (factor_over_τ^2 + id_minus_factor2_D2_over_τ^2)

                        # Compute z1
                        z1[i] = tmp1[i] + factor_D * z2[i]

                        # Compute z0
                        z0[i] = tmp0[i] - factor_D * z3[i]
                    end
                else
                    F = let W = W, factor = factor,
                            factorization = factorization,
                            rhs_stiff_operator = rhs_stiff_operator
                        get!(factorizations, factor) do
                            fill!(W, 0)
                            W[diagind(W)] .= 1
                            @. W -= factor * rhs_stiff_operator
                            if issparse(W)
                                lu!(factorization, W)
                            else
                                factorization = lu!(W)
                            end
                            copy(factorization)
                        end
                    end
                    ldiv!(z, F, tmp)
                end
            end

            # Compute new stage derivatives
            @. y = q + dt * z
            rhs_nonstiff!(k_nonstiff[i], y, parameters, t + c_nonstiff[i] * dt)
            if iszero(rhs_stiff_operator) || iszero(A_stiff[i, i])
                rhs_stiff!(k_stiff[i], y, parameters, t + c_stiff[i] * dt)
            else
                # The code below is equivalent to
                #   rhs_stiff!(k_stiff[i], y, parameters, t + c_stiff[i] * dt)
                # but avoids evaluating the stiff operator at the numerical solution.
                @. tmp = z
                for j in 1:(i-1)
                    @. tmp = tmp - A_stiff[i, j] * k_stiff[j] - A_nonstiff[i, j] * k_nonstiff[j]
                end
                @. k_stiff[i] = tmp / A_stiff[i, i]
            end
        end

        # Update solution
        fill!(tmp, 0)
        for j in 1:s
            @. tmp += b_stiff[j] * k_stiff[j] + b_nonstiff[j] * k_nonstiff[j]
        end

        t = relaxation!(q, tmp, y, z, t, dt,
                        parameters.equation, parameters,
                        relaxation, relaxation_alg, relaxation_tol,
                        mass_old, momentum_old, energy_old)

        if save_everystep
            push!(sol_q, copy(q))
            append!(sol_t, t)
        end
        callback(q, parameters, t)

        if any(isnan, q)
            @error "NaNs in solution at time $t" q @__LINE__
            error()
        end
    end

    if save_everystep
        return (; u = sol_q,
                  t = sol_t)
    else
        return (; u = (q0, q),
                  t = (first(tspan), t))
    end
end



#####################################################################
# General interface

abstract type AbstractEquation end
Base.Broadcast.broadcastable(equation::AbstractEquation) = (equation,)

abstract type AbstractSemidiscretization end
struct FourierGalerkin <: AbstractSemidiscretization end
abstract type FourierCollocation <: AbstractSemidiscretization end
struct FourierCollocationConserveMassEnergy <: FourierCollocation end


#####################################################################
# BBM discretization

struct BBM <: AbstractEquation end

# Fourier Galerkin discretization
function get_u(q::AbstractVector{<:Complex}, equations::BBM, parameters)
    (; D_small) = parameters
    D_small.tmp .= q ./ size(D_small, 2)
    return D_small.brfft_plan * D_small.tmp
end

function rhs_stiff!(dq, q, equation::BBM, ::FourierGalerkin, parameters, t)
    fill!(dq, zero(eltype(dq)))
    return nothing
end

operator(::typeof(rhs_stiff!), equation::BBM, ::FourierGalerkin, parameters) = 0 * I

function rhs_nonstiff!(dq, q, equation::BBM, ::FourierGalerkin,
                       parameters, t)
    (; D_small, D_large2, tmp1_large2) = parameters
    one_half = one(real(eltype(q))) / 2

    # First, we copy the spectral coefficients
    # to the larger grid, pad them by zeros, and transform back to
    # physical space on the larger grid. Since we use brfft instead
    # of irfft, we need to normalize by the number of grid points -
    # of the grid where we computed the spectral coefficients originally
    # (i.e., the smaller grid).
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(q)] .= q ./ size(D_small, 2)
    mul!(tmp1_large2, D_large2.brfft_plan, D_large2.tmp)
    # Next, we compute the nonlinear term on the larger grid.
    @. tmp1_large2 = -one_half * tmp1_large2^2
    # Now, we transform back to spectral space on the larger grid.
    mul!(D_large2.tmp, D_large2.rfft_plan, tmp1_large2)
    # We copy the relevant coefficients to the smaller grid.
    copyto!(D_small.tmp, 1, D_large2.tmp, 1, length(D_small.tmp))
    # We compute the discrete derivative operator on the smaller grid.
    # We do not use the special handling of the highest frequency mode
    # for an even number of grid points since we interpret the spectral
    # coefficients as truncated from a larger grid.
    ratio = size(D_small, 2) / size(D_large2, 2)
    for i in eachindex(D_small.tmp)
        f1 = (i - 1) * im * D_large2.jac * size(D_large2, 2)
        f2 = 1 + (i - 1)^2 * D_large2.jac^2 * size(D_large2, 2)^2
        dq[i] = D_small.tmp[i] * f1 / f2 * ratio
    end

    return nothing
end

function mass(q, equation::BBM, ::FourierGalerkin, parameters)
    (; D_small) = parameters
    return real(q[1]) * D_small.Δx
end

function momentum(q, equation::BBM, ::FourierGalerkin, parameters)
    (; D_small, D_large2, tmp1_large2, tmp2_large2) = parameters

    # First, we copy the spectral coefficients
    # to the larger grid, pad them by zeros, and transform back to
    # physical space on the larger grid. Since we use brfft instead
    # of irfft, we need to normalize by the number of grid points -
    # of the grid where we computed the spectral coefficients originally
    # (i.e., the smaller grid).
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(q)] .= q ./ size(D_small, 2)
    mul!(tmp1_large2, D_large2.brfft_plan, D_large2.tmp)
    # Next, we compute the nonlinear term on the larger grid.
    mul!(tmp2_large2, D_large2, tmp1_large2)
    @. tmp1_large2 = (tmp1_large2^2 + tmp2_large2^2) / 2
    return integrate(tmp1_large2, D_large2)
end

function energy(q, equation::BBM, ::FourierGalerkin, parameters)
    (; D_small, D_large3, tmp1_large3) = parameters

    # First, we copy the spectral coefficients
    # to the larger grid, pad them by zeros, and transform back to
    # physical space on the larger grid. Since we use brfft instead
    # of irfft, we need to normalize by the number of grid points -
    # of the grid where we computed the spectral coefficients originally
    # (i.e., the smaller grid).
    fill!(D_large3.tmp, 0)
    D_large3.tmp[1:length(q)] .= q ./ size(D_small, 2)
    mul!(tmp1_large3, D_large3.brfft_plan, D_large3.tmp)
    # Next, we compute the nonlinear term on the larger grid.
    @. tmp1_large3 = tmp1_large3^3 / 6
    return integrate(tmp1_large3, D_large3)
end

function setup(u_func, equation::BBM, semidiscretization::FourierGalerkin, tspan, D)
    @assert D isa FourierDerivativeOperator

    # De-aliasing rule for degree p nonlinearities:
    # Use a grid with at least factor (p + 1) / 2 more modes.
    # We also need to remember that the sizes below are the numbers
    # of nodes in physical space. When using the rfft, the number of
    # modes is $(N + 1) / 2$ for odd N and $N / 2 + 1$ for even N.
    #
    # The BBM equation has at most a cubic nonlinearity in the
    # energy, for which we use the factor 4 / 2. However, several
    # other terms require only lower-order nonlinearities, e.g.,
    # quadratic nonlinearities in the nonlinear flux and momentum.
    # To improve efficiency, we introduce specialized larger grids
    # for them.
    D_small = D
    D_large2 = fourier_derivative_operator(xmin(D), xmax(D),
                    dealiased_number_of_nodes(2, size(D_small, 2)))
    D_large3 = fourier_derivative_operator(xmin(D), xmax(D),
                    dealiased_number_of_nodes(3, size(D_small, 2)))

    x = grid(D_small)
    u0 = u_func.(tspan[1], x, equation)
    tmp1_small = similar(u0)
    tmp1_large2 = similar(u0, size(D_large2, 2))
    tmp2_large2 = similar(u0, size(D_large2, 2))
    tmp1_large3 = similar(u0, size(D_large3, 2))
    tmp2_large3 = similar(u0, size(D_large3, 2))

    q0 = D_small.rfft_plan * u0
    parameters = (; equation, semidiscretization,
                    D_small, tmp1_small,
                    D_large2, tmp1_large2, tmp2_large2,
                    D_large3, tmp1_large3, tmp2_large3)
    return (; q0, parameters)
end


function relaxation!(q, tmp, y, z, t, dt,
                     equation::BBM, parameters,
                     ::NoProjection,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    @. q = q + dt * tmp
    return t + dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::BBM, parameters,
                     ::MassMomentumProjection,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    y[1] = 1
    volume = mass(y, parameters)
    mean_value = mass(q, parameters) / volume
    mean_value_old = mass_old / volume
    fill!(y, 0); y[1] = mean_value
    momentum_mean = momentum(y, parameters)
    @. y = q + dt * tmp; y[1] = 0
    momentum_rel = momentum(y, parameters)
    factor = sqrt((momentum_old - momentum_mean) / momentum_rel)
    @. q = factor * y; q[1] = mean_value_old
    return t + dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::BBM, parameters,
                     ::MassMomentumRelaxation,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    @. y = q - tmp
    momentum_minus = momentum(y, parameters)
    @. y = q + tmp
    momentum_plus = momentum(y, parameters)
    momentum_tmp = momentum(tmp, parameters)
    gamma = (momentum_minus - momentum_plus) / (2 * dt * momentum_tmp)
    if !(0.1 < gamma < 10)
        error("Relaxation failed: gamma = $gamma at time $t. Try using a smaller time step size.")
    end
    gamma_dt = gamma * dt
    @. q = q + gamma_dt * tmp
    return t + gamma_dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::BBM, parameters,
                     ::MassEnergyRelaxation,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    # We search for a solution conserving the energy
    # along the secant between q (old value) and y (new value)
    prob = NonlinearProblem{false}(1.0) do gamma, _
        gamma_dt = gamma * dt
        @. z = q + gamma_dt * tmp
        return (energy(z, parameters) - energy_old) / abs(energy_old)
    end
    sol = solve(prob, relaxation_alg;
                abstol = relaxation_tol, reltol = relaxation_tol)
    gamma = sol.u
    if !(0.1 < gamma < 10)
        error("Relaxation failed: gamma = $gamma at time $t. Try using a smaller time step size.")
    end
    @. q = q + gamma * dt * tmp
    return t + gamma * dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::BBM, parameters,
                     ::ProjectionEnergyRelaxation,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    # First, we project the momentum
    y[1] = 1 # nodal: @. y = 1
    volume = mass(y, parameters)
    mean_value = mass(q, parameters) / volume
    mean_value_old = mass_old / volume
    fill!(y, 0); y[1] = mean_value # nodal: fill!(y, mean_value)
    momentum_mean = momentum(y, parameters)
    @. y = q + dt * tmp; y[1] = 0 # nodal: @. y = q - mean_value + dt * tmp
    momentum_rel = momentum(y, parameters)
    factor = sqrt((momentum_old - momentum_mean) / momentum_rel)
    @. y = factor * y; y[1] = mean_value_old # nodal: @. y = mean_value_old + factor * y

    # Next, we search for a solution conserving the energy
    # along the geodesic between q (old value) and y (new value)
    prob = NonlinearProblem{false}(1.0) do gamma, _
        @. z = q + gamma * (y - q); z[1] = 0
        momentum_rel = momentum(z, parameters)
        factor = sqrt((momentum_old - momentum_mean) / momentum_rel)
        @. z = factor * z; z[1] = mean_value_old
        return (energy(z, parameters) - energy_old) / abs(energy_old)
    end
    sol = solve(prob, relaxation_alg;
                abstol = relaxation_tol, reltol = relaxation_tol)
    gamma = sol.u
    if !(0.1 < gamma < 10)
        error("Relaxation failed: gamma = $gamma at time $t. Try using a smaller time step size.")
    end
    @. z = q + gamma * (y - q); z[1] = 0
    momentum_rel = momentum(z, parameters)
    factor = sqrt((momentum_old - momentum_mean) / momentum_rel)
    @. q = factor * z; q[1] = mean_value_old
    return t + gamma * dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::BBM, parameters,
                     ::RelaxationEnergyRelaxation,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    y[1] = 1 # nodal: @. y = 1
    volume = mass(y, parameters)
    mean_value = mass(q, parameters) / volume
    mean_value_old = mass_old / volume
    fill!(y, 0); y[1] = mean_value # nodal: fill!(y, mean_value)
    momentum_mean = momentum(y, parameters)
    @. y = q + dt * tmp; y[1] = 0 # nodal: @. y = q - mean_value + dt * tmp
    momentum_rel = momentum(y, parameters)
    # First, we relax to conserve the momentum
    @. y = q - tmp
    momentum_minus = momentum(y, parameters)
    @. y = q + tmp
    momentum_plus = momentum(y, parameters)
    momentum_tmp = momentum(tmp, parameters)
    gamma_momentum = (momentum_minus - momentum_plus) / (2 * dt * momentum_tmp)
    @. y = q + gamma_momentum * dt * tmp

    # Next, we search for a solution conserving the energy
    # along the geodesic between q (old value) and y (new value)
    prob = NonlinearProblem{false}(1.0) do gamma, _
        @. z = q + gamma * (y - q); z[1] = 0
        momentum_rel = momentum(z, parameters)
        factor = sqrt((momentum_old - momentum_mean) / momentum_rel)
        @. z = factor * z; z[1] = mean_value_old
        return (energy(z, parameters) - energy_old) / abs(energy_old)
    end
    sol = solve(prob, relaxation_alg;
                abstol = relaxation_tol, reltol = relaxation_tol)
    gamma = sol.u
    if !(0.1 < gamma < 10)
        error("Relaxation failed: gamma = $gamma at time $t. Try using a smaller time step size.")
    end
    @. z = q + gamma * (y - q); z[1] = 0
    momentum_rel = momentum(z, parameters)
    factor = sqrt((momentum_old - momentum_mean) / momentum_rel)
    @. q = factor * z; q[1] = mean_value_old
    return t + gamma_momentum * gamma * dt
end

function solitary_wave(t, x::Number, equation::BBM)
    (; xmin, xmax) = domain(solitary_wave, equation)

    c = 1.2
    A = 3 * (c - 1)
    K = 0.5 * sqrt(1 - 1 / c)
    x_t = mod(x - 20 - c * t - xmin, xmax - xmin) + xmin

    # There are two normalizations of the BBM equation:
    # 1. u_t - u_{txx} + u_x + u u_x = 0
    # return A / cosh(K * x_t)^2
    # 2. u_t - u_{txx} + u u_x = 0
    return 1 + A / cosh(K * x_t)^2
end
domain(::typeof(solitary_wave), ::BBM) = (xmin = -90.0, xmax = 90.0)

"""
    periodic_wave(t, x::Number, equation::BBM)

Periodic traveling-wave solution on mean level 1 for the BBM equation

    u_t + (u^2 / 2)_x - u_{txx} = 0

of the form

    u(t, x) = r2 + H * cn(κ * (x - c t), m)^2,

where `cn` is the Jacobi elliptic cosine with modulus `m = 0.8`
and wave speed `c = 0.9`.
"""
function periodic_wave(t, x::Number, equation::BBM)
    m = 0.8 # elliptic modulus
    c = 0.9 # wave speed
    (; xmin, xmax) = domain(periodic_wave, equation)

    # Complete elliptic integrals (parameter m)
    K = JacobiElliptic.K(m)
    E = JacobiElliptic.E(m)

    # Wave height H for unit mean:  ⟨u⟩ = 1
    denom = (E / K) - (2m - 1) / (3m)
    H = (1 - c) / denom

    # Cubic roots r1 < r2 < r3 encoded via (c, m, H)
    r2 = (3c - (H / m) * (2m - 1)) / 3
    # r3 = r2 + H
    # r1 is not needed for evaluation but shown for completeness:
    # r1 = r2 - H * (1 - m) / m

    # Spatial frequency κ and period Λ
    κ = sqrt(H / (12c * m))
    # Λ = 2K / κ   # wavelength if needed

    # Phase (wrap for torus domains)
    L = xmax - xmin
    x_t = mod(x - c * t - xmin, L) + xmin

    # Jacobi elliptic functions at argument κ ξ with modulus m
    cn = JacobiElliptic.cn(κ * (x_t - xmin), m)
    return r2 + H * cn^2
end
domain(::typeof(periodic_wave), ::BBM) = (xmin = 0.0, xmax = 10 * 21.88886653237668)

function two_solitary_waves(t, x::Number, equation::BBM)
    (; xmin, xmax) = domain(two_solitary_waves, equation)

    c = 1.2
    A = 3 * (c - 1)
    K = 0.5 * sqrt(1 - 1 / c)
    x_t = mod(x - 20 - c * t - xmin, xmax - xmin) + xmin
    u1 = A / cosh(K * x_t)^2

    c = 1.3
    A = 3 * (c - 1)
    K = 0.5 * sqrt(1 - 1 / c)
    x_t = mod(x + 20 - c * t - xmin, xmax - xmin) + xmin
    u2 = A / cosh(K * x_t)^2

    return 1 + u1 + u2
end
domain(::typeof(two_solitary_waves), ::BBM) = (xmin = -100.0, xmax = 100.0)



#####################################################################
# KdV discretization

struct KdV <: AbstractEquation end

# Fourier Galerkin discretization
function get_u(q::AbstractVector{<:Complex}, equations::KdV, parameters)
    (; D_small) = parameters
    D_small.tmp .= q ./ size(D_small, 2)
    return D_small.brfft_plan * D_small.tmp
end

function rhs_stiff!(dq, q, equation::KdV, ::FourierGalerkin, parameters, t)
    (; D_small) = parameters
    # We compute the discrete derivative operator on the smaller grid.
    # We do not use the special handling of the highest frequency mode
    # for an even number of grid points since we interpret the spectral
    # coefficients as truncated from a larger grid.
    for i in eachindex(q)
        f = (i - 1)^3 * im * D_small.jac^3 * size(D_small, 2)^3
        dq[i] = q[i] * f
    end
    return nothing
end

abstract type AbstractStiffOperator end
struct StiffOperatorKdV <: AbstractStiffOperator end
Base.:*(::Number, op::StiffOperatorKdV) = op
Base.:-(::UniformScaling, op::StiffOperatorKdV) = op
Base.iszero(op::StiffOperatorKdV) = false

operator(::typeof(rhs_stiff!), equation::KdV, ::FourierGalerkin, parameters) = StiffOperatorKdV()

function rhs_nonstiff!(dq, q, equation::KdV, ::FourierGalerkin, parameters, t)
    (; D_small, D_large2, tmp1_large2) = parameters
    one_half = one(real(eltype(q))) / 2

    # First, we copy the spectral coefficients
    # to the larger grid, pad them by zeros, and transform back to
    # physical space on the larger grid. Since we use brfft instead
    # of irfft, we need to normalize by the number of grid points -
    # of the grid where we computed the spectral coefficients originally
    # (i.e., the smaller grid).
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(q)] .= q ./ size(D_small, 2)
    mul!(tmp1_large2, D_large2.brfft_plan, D_large2.tmp)
    # Next, we compute the nonlinear term on the larger grid.
    @. tmp1_large2 = -one_half * tmp1_large2^2
    # Now, we transform back to spectral space on the larger grid.
    mul!(D_large2.tmp, D_large2.rfft_plan, tmp1_large2)
    # We copy the relevant coefficients to the smaller grid.
    copyto!(D_small.tmp, 1, D_large2.tmp, 1, length(D_small.tmp))
    # We compute the discrete derivative operator on the smaller grid.
    # We do not use the special handling of the highest frequency mode
    # for an even number of grid points since we interpret the spectral
    # coefficients as truncated from a larger grid.
    ratio = size(D_small, 2) / size(D_large2, 2)
    for i in eachindex(D_small.tmp)
        f = (i - 1) * im * D_large2.jac * size(D_large2, 2)
        dq[i] = D_small.tmp[i] * f * ratio
    end

    return nothing
end

function mass(q, equation::KdV, ::FourierGalerkin, parameters)
    (; D_small) = parameters
    return real(q[1]) * D_small.Δx
end

function momentum(q, equation::KdV, ::FourierGalerkin, parameters)
    (; D_small, D_large2, tmp1_large2) = parameters

    # First, we copy the spectral coefficients
    # to the larger grid, pad them by zeros, and transform back to
    # physical space on the larger grid. Since we use brfft instead
    # of irfft, we need to normalize by the number of grid points -
    # of the grid where we computed the spectral coefficients originally
    # (i.e., the smaller grid).
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(q)] .= q ./ size(D_small, 2)
    mul!(tmp1_large2, D_large2.brfft_plan, D_large2.tmp)
    # Next, we compute the nonlinear term on the larger grid.
    @. tmp1_large2 = tmp1_large2^2 / 2
    return integrate(tmp1_large2, D_large2)
end

function energy(q, equation::KdV, ::FourierGalerkin, parameters)
    (; D_small, D_large3, tmp1_large3, tmp2_large3) = parameters

    # First, we copy the spectral coefficients
    # to the larger grid, pad them by zeros, and transform back to
    # physical space on the larger grid. Since we use brfft instead
    # of irfft, we need to normalize by the number of grid points -
    # of the grid where we computed the spectral coefficients originally
    # (i.e., the smaller grid).
    fill!(D_large3.tmp, 0)
    D_large3.tmp[1:length(q)] .= q ./ size(D_small, 2)
    mul!(tmp1_large3, D_large3.brfft_plan, D_large3.tmp)
    # Next, we compute the derivative on the larger grid.
    # We do not use the special handling of the highest frequency mode
    # for an even number of grid points since we interpret the spectral
    # coefficients as truncated from a larger grid.
    # To compute the derivative, we need to multiply by size(D_small, 2).
    # This cancels with the division by size(D_small, 2) that we need
    # for the inverse transform using brfft.
    for i in eachindex(q)
        f = (i - 1) * im * D_small.jac
        D_large3.tmp[i] = q[i] * f
    end
    mul!(tmp2_large3, D_large3.brfft_plan, D_large3.tmp)
    # Next, we compute the nonlinear term on the larger grid.
    @. tmp1_large3 = tmp2_large3^2 / 2 - tmp1_large3^3 / 6
    return integrate(tmp1_large3, D_large3)
end

function setup(u_func, equation::KdV, semidiscretization::FourierGalerkin, tspan, D)
    @assert D isa FourierDerivativeOperator

    # De-aliasing rule for degree p nonlinearities:
    # Use a grid with at least factor (p + 1) / 2 more points.
    # The KdV equation has at most a cubic nonlinearity in the
    # energy, for which we use the factor 4 / 2. However, several
    # other terms require only lower-order nonlinearities, e.g.,
    # quadratic nonlinearities in the nonlinear flux and momentum.
    # To improve efficiency, we introduce specialized larger grids
    # for them.
    D_small = D
    D_large2 = fourier_derivative_operator(xmin(D), xmax(D),
                    dealiased_number_of_nodes(2, size(D_small, 2)))
    D_large3 = fourier_derivative_operator(xmin(D), xmax(D),
                    dealiased_number_of_nodes(3, size(D_small, 2)))

    x = grid(D_small)
    u0 = u_func.(tspan[1], x, equation)
    tmp1_small = similar(u0)
    tmp1_large2 = similar(u0, size(D_large2, 2))
    tmp2_large2 = similar(u0, size(D_large2, 2))
    tmp1_large3 = similar(u0, size(D_large3, 2))
    tmp2_large3 = similar(u0, size(D_large3, 2))

    q0 = D_small.rfft_plan * u0
    parameters = (; equation, semidiscretization,
                    D_small, tmp1_small,
                    D_large2, tmp1_large2, tmp2_large2,
                    D_large3, tmp1_large3, tmp2_large3)
    return (; q0, parameters)
end

function relaxation!(q, tmp, y, z, t, dt,
                     equation::KdV, parameters,
                     ::NoProjection,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    @. q = q + dt * tmp
    return t + dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::KdV, parameters,
                     ::MassMomentumProjection,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    y[1] = 1
    volume = mass(y, parameters)
    mean_value = mass(q, parameters) / volume
    mean_value_old = mass_old / volume
    fill!(y, 0); y[1] = mean_value
    momentum_mean = momentum(y, parameters)
    @. y = q + dt * tmp; y[1] = 0
    momentum_rel = momentum(y, parameters)
    factor = sqrt((momentum_old - momentum_mean) / momentum_rel)
    @. q = factor * y; q[1] = mean_value_old
    return t + dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::KdV, parameters,
                     ::MassMomentumRelaxation,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    @. y = q - tmp
    momentum_minus = momentum(y, parameters)
    @. y = q + tmp
    momentum_plus = momentum(y, parameters)
    momentum_tmp = momentum(tmp, parameters)
    gamma = (momentum_minus - momentum_plus) / (2 * dt * momentum_tmp)
    if !(0.1 < gamma < 10)
        error("Relaxation failed: gamma = $gamma at time $t. Try using a smaller time step size.")
    end
    gamma_dt = gamma * dt
    @. q = q + gamma_dt * tmp
    return t + gamma_dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::KdV, parameters,
                     ::MassEnergyRelaxation,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    # We search for a solution conserving the energy
    # along the secant between q (old value) and y (new value)
    prob = NonlinearProblem{false}(1.0) do gamma, _
        gamma_dt = gamma * dt
        @. z = q + gamma_dt * tmp
        return (energy(z, parameters) - energy_old) / abs(energy_old)
    end
    sol = solve(prob, relaxation_alg;
                abstol = relaxation_tol, reltol = relaxation_tol)
    gamma = sol.u
    if !(0.1 < gamma < 10)
        error("Relaxation failed: gamma = $gamma at time $t. Try using a smaller time step size.")
    end
    @. q = q + gamma * dt * tmp
    return t + gamma * dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::KdV, parameters,
                     ::ProjectionEnergyRelaxation,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    # First, we project the momentum
    y[1] = 1 # nodal: @. y = 1
    volume = mass(y, parameters)
    mean_value = mass(q, parameters) / volume
    mean_value_old = mass_old / volume
    fill!(y, 0); y[1] = mean_value # nodal: fill!(y, mean_value)
    momentum_mean = momentum(y, parameters)
    @. y = q + dt * tmp; y[1] = 0 # nodal: @. y = q - mean_value + dt * tmp
    momentum_rel = momentum(y, parameters)
    factor = sqrt((momentum_old - momentum_mean) / momentum_rel)
    @. y = factor * y; y[1] = mean_value_old # nodal: @. y = mean_value_old + factor * y

    # Next, we search for a solution conserving the energy
    # along the geodesic between q (old value) and y (new value)
    prob = NonlinearProblem{false}(1.0) do gamma, _
        @. z = q + gamma * (y - q); z[1] = 0
        momentum_rel = momentum(z, parameters)
        factor = sqrt((momentum_old - momentum_mean) / momentum_rel)
        @. z = factor * z; z[1] = mean_value_old
        return (energy(z, parameters) - energy_old) / abs(energy_old)
    end
    sol = solve(prob, relaxation_alg;
                abstol = relaxation_tol, reltol = relaxation_tol)
    gamma = sol.u
    if !(0.1 < gamma < 10)
        error("Relaxation failed: gamma = $gamma at time $t. Try using a smaller time step size.")
    end
    @. z = q + gamma * (y - q); z[1] = 0
    momentum_rel = momentum(z, parameters)
    factor = sqrt((momentum_old - momentum_mean) / momentum_rel)
    @. q = factor * z; q[1] = mean_value_old
    return t + gamma * dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::KdV, parameters,
                     ::RelaxationEnergyRelaxation,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    y[1] = 1 # nodal: @. y = 1
    volume = mass(y, parameters)
    mean_value = mass(q, parameters) / volume
    mean_value_old = mass_old / volume
    fill!(y, 0); y[1] = mean_value # nodal: fill!(y, mean_value)
    momentum_mean = momentum(y, parameters)
    @. y = q + dt * tmp; y[1] = 0 # nodal: @. y = q - mean_value + dt * tmp
    momentum_rel = momentum(y, parameters)
    # First, we relax to conserve the momentum
    @. y = q - tmp
    momentum_minus = momentum(y, parameters)
    @. y = q + tmp
    momentum_plus = momentum(y, parameters)
    momentum_tmp = momentum(tmp, parameters)
    gamma_momentum = (momentum_minus - momentum_plus) / (2 * dt * momentum_tmp)
    @. y = q + gamma_momentum * dt * tmp

    # Next, we search for a solution conserving the energy
    # along the geodesic between q (old value) and y (new value)
    prob = NonlinearProblem{false}(1.0) do gamma, _
        @. z = q + gamma * (y - q); z[1] = 0
        momentum_rel = momentum(z, parameters)
        factor = sqrt((momentum_old - momentum_mean) / momentum_rel)
        @. z = factor * z; z[1] = mean_value_old
        return (energy(z, parameters) - energy_old) / abs(energy_old)
    end
    sol = solve(prob, relaxation_alg;
                abstol = relaxation_tol, reltol = relaxation_tol)
    gamma = sol.u
    if !(0.1 < gamma < 10)
        error("Relaxation failed: gamma = $gamma at time $t. Try using a smaller time step size.")
    end
    @. z = q + gamma * (y - q); z[1] = 0
    momentum_rel = momentum(z, parameters)
    factor = sqrt((momentum_old - momentum_mean) / momentum_rel)
    @. q = factor * z; q[1] = mean_value_old
    return t + gamma_momentum * gamma * dt
end

function solitary_wave(t, x::Number, equation::KdV)
    c = 1.2
    (; xmin, xmax) = domain(solitary_wave, equation)

    A = 3 * c
    K = sqrt(3 * A) / 6
    x_t = mod(x - c * t - xmin, xmax - xmin) + xmin

    return A / cosh(K * x_t)^2
end
domain(::typeof(solitary_wave), ::KdV) = (xmin = -40.0, xmax = 40.0)

"""
    periodic_wave(t, x::Number, equation::KdV)

Unit-mean periodic (cnoidal) traveling-wave solution for the KdV equation

    u_t + (u^2 / 2)_x + u_{xxx} = 0,

of the form

    u(x,t) = r2 + H * cn(κ * (x - c t), m)^2,

where `cn` is the Jacobi elliptic cosine with modulus `m = 0.8`
and wave speed `c = 0.9`.
"""
function periodic_wave(t, x::Number, equation::KdV)
    m = 0.8 # elliptic modulus
    c = 0.9 # wave speed
    (; xmin, xmax) = domain(periodic_wave, equation)

    Kc = JacobiElliptic.K(m)
    Ec = JacobiElliptic.E(m)

    # Average of cn^2 over one period
    cn2_avg = (Ec / Kc - (1 - m)) / m

    # Wave height from mean-1 constraint and KdV relations
    denom = (2 - 1/m) - 3 * cn2_avg
    H = 3 * (c - 1) / denom

    # r2 is the trough level; crest is r3 = r2 + H
    r2 = 1 - H * cn2_avg

    # Spatial frequency and period
    κ = sqrt(H / (12 * m))
    # Λ = 2Kc / κ   # wavelength if needed

    # Phase (wrap for torus domains)
    L = xmax - xmin
    x_t = mod(x - c * t - xmin, L) + xmin

    # Jacobi elliptic functions at argument κ ξ with modulus m
    cn = JacobiElliptic.cn(κ * (x_t - xmin), m)
    return r2 + H * cn^2
end
domain(::typeof(periodic_wave), ::KdV) = (xmin = 00.0, xmax = 10 * 17.280324716756375)

function one_soliton(t, x::Number, equation::KdV)
    (; xmin, xmax) = domain(one_soliton, equation)

    k1 = 0.75
    x1 = 0.0

    xc1 = mod(x1 + k1^2 * t - xmin, xmax - xmin) + xmin

    u = nth_derivative(x, Val{2}()) do x
        eta1 = k1 * ((x - xc1) - (xmax - xmin) * round((x - xc1) / (xmax - xmin)))
        # Compute log(F) where
        #   F = 1 + exp(eta1)
        # using the log sum exp stabilization (where 1 = exp(0)).
        α = max(0, eta1)
        log_F = α + log(exp(-α) + exp(eta1 - α))
        return 12 * log_F
    end

    return u
end
domain(::typeof(one_soliton), ::KdV) = (xmin = -50.0, xmax = 50.0)

function two_solitons(t, x::Number, equation::KdV)
    (; xmin, xmax) = domain(two_solitons, equation)

    u = nth_derivative(x, Val{2}()) do x
        k1 = 0.75
        k2 = 0.5
        x1 = -50.0
        x2 = +50.0

        a = ((k1 - k2) / (k1 + k2))^2

        log_F = zero(x)

        for n in -2:0
            # This works fine until the solitons interact multiple times.
            # With the current choice of parameters, this is around time
            # t = 1270. To extend this, increase the `domain` size.
            eta1 = k1 * (x - x1 - n * (xmax - xmin) - k1^2 * t)
            eta2 = k2 * (x - x2 - n * (xmax - xmin) - k2^2 * t)

            # Compute log(F) where
            #   F = 1 + exp(eta1) + exp(eta2) + a * exp(eta1 + eta2)
            # using the log sum exp stabilization (where 1 = exp(0)).
            α = maximum((0, eta1, eta2, eta2, eta1 + eta2))
            log_F += α + log(exp(-α) + exp(eta1 - α) + exp(eta2 - α) +
                             a * exp(eta1 + eta2 - α))
        end

        return 12 * log_F
    end

    return u
end
domain(::typeof(two_solitons), ::KdV) = (xmin = -200.0, xmax = 200.0)

function three_solitons(t, x::Number, equation::KdV)
    (; xmin, xmax) = domain(three_solitons, equation)

    u = nth_derivative(x, Val{2}()) do x
        k1 = 0.75
        k2 = 0.5
        k3 = 0.25
        x1 = -100.0
        x2 = 0.0
        x3 = +100.0

        a12 = ((k1 - k2) / (k1 + k2))^2
        a13 = ((k1 - k3) / (k1 + k3))^2
        a23 = ((k2 - k3) / (k2 + k3))^2

        log_F = zero(x)

        for n in -2:0
            # This works fine until the solitons interact multiple times.
            # With the current choice of parameters, this is around time
            # t = 1600. To extend this, increase the `domain` size.
            eta1 = k1 * (x - x1 - n * (xmax - xmin) - k1^2 * t)
            eta2 = k2 * (x - x2 - n * (xmax - xmin) - k2^2 * t)
            eta3 = k3 * (x - x3 - n * (xmax - xmin) - k3^2 * t)

            # Compute log(F) where
            #   F = 1 + exp(eta1) + exp(eta2) + exp(eta3) + a12 * exp(eta1 + eta2) + a13 * exp(eta1 + eta3) + a23 * exp(eta2 + eta3) + a12 * a13 * a23 * exp(eta1 + eta2 + eta3)
            # using the log sum exp stabilization (where 1 = exp(0)).
            α = maximum((0, eta1, eta2, eta3, eta1 + eta2, eta1 + eta3, eta2 + eta3, eta1 + eta2 + eta3))
            log_F += α + log(exp(-α) + exp(eta1 - α) + exp(eta2 - α) + exp(eta3 - α) +
                             a12 * exp(eta1 + eta2 - α) + a13 * exp(eta1 + eta3 - α) + a23 * exp(eta2 + eta3 - α) +
                             a12 * a13 * a23 * exp(eta1 + eta2 + eta3 - α))
        end

        return 12 * log_F
    end

    return u
end
domain(::typeof(three_solitons), ::KdV) = (xmin = -400.0, xmax = 400.0)



#####################################################################
# CubicNLS discretization

struct CubicNLS{T} <: AbstractEquation
    β::T
end

# Fourier Galerkin discretization
Base.real(q, equation::CubicNLS) = get_qi(q, equation, 0)
Base.imag(q, equation::CubicNLS) = get_qi(q, equation, 1)
function get_qi(q, equation::CubicNLS, i)
    N = length(q) ÷ 2
    return view(q, (i * N + 1):((i + 1) * N))
end

function density(q, equation::CubicNLS, parameters)
    return density(q, equation, parameters.semidiscretization, parameters)
end
function density(q, equation::CubicNLS, ::FourierGalerkin, parameters)
    (; D_small) = parameters
    v = D_small.brfft_plan * (real(q, equation) ./ size(D_small, 2))
    w = D_small.brfft_plan * (imag(q, equation) ./ size(D_small, 2))
    return @. v^2 + w^2
end
function density(q, equation::CubicNLS, ::FourierCollocation, parameters)
    v = real(q, equation)
    w = imag(q, equation)
    return @. v^2 + w^2
end

function rhs_stiff!(dq, q, equation::CubicNLS, ::FourierGalerkin, parameters, t)
    (; D_small) = parameters

    dv = real(dq, equation)
    dw = imag(dq, equation)

    v = real(q, equation)
    w = imag(q, equation)

    # We compute the discrete derivative operator on the smaller grid.
    for i in eachindex(dv)
        f = -(i - 1)^2 * D_small.jac^2 * size(D_small, 2)^2
        # dv = -D_2 w
        dv[i] = -f * w[i]
        # dw = D_2 v
        dw[i] = f * v[i]
    end

    return nothing
end
function rhs_stiff!(dq, q, equation::CubicNLS, ::FourierCollocationConserveMassEnergy, parameters, t)
    (; D2) = parameters

    dv = real(dq, equation)
    dw = imag(dq, equation)

    v = real(q, equation)
    w = imag(q, equation)

    # dv = M^{-1} A_2 w = -D_2 w
    mul!(dv, D2, w)
    @. dv = -dv

    # dw = -M^{-1} A_2 v = D_2 v
    mul!(dw, D2, v)

    return nothing
end

struct StiffOperatorCubicNLS <: AbstractStiffOperator end
Base.:*(::Number, op::StiffOperatorCubicNLS) = op
Base.:-(::UniformScaling, op::StiffOperatorCubicNLS) = op
Base.iszero(op::StiffOperatorCubicNLS) = false

operator(::typeof(rhs_stiff!), equation::CubicNLS, ::FourierGalerkin, parameters) = StiffOperatorCubicNLS()
operator(::typeof(rhs_stiff!), equation::CubicNLS, ::FourierCollocation, parameters) = StiffOperatorCubicNLS()

function rhs_nonstiff!(dq, q, equation::CubicNLS, ::FourierGalerkin, parameters, t)
    (; D_small, D_large3, tmp1_large3, tmp2_large3, tmp3_large3) = parameters
    (; β) = equation

    dv = real(dq, equation)
    dw = imag(dq, equation)

    v = real(q, equation)
    w = imag(q, equation)

    # First, we copy the spectral coefficients of the real part
    # to the larger grid, pad them by zeros, and transform back to
    # physical space on the larger grid. Since we use brfft instead
    # of irfft, we need to normalize by the number of grid points -
    # of the grid where we computed the spectral coefficients originally
    # (i.e., the smaller grid).
    fill!(D_large3.tmp, 0)
    D_large3.tmp[1:length(v)] .= v ./ size(D_small, 2)
    mul!(tmp1_large3, D_large3.brfft_plan, D_large3.tmp)
    # Now, we do the same for the imaginary part.
    fill!(D_large3.tmp, 0)
    D_large3.tmp[1:length(w)] .= w ./ size(D_small, 2)
    mul!(tmp2_large3, D_large3.brfft_plan, D_large3.tmp)
    # Next, we compute the nonlinear term on the larger grid.
    @. tmp3_large3 = β * (tmp1_large3^2 + tmp2_large3^2)
    @. tmp1_large3 = tmp3_large3 * tmp1_large3
    @. tmp2_large3 = tmp3_large3 * tmp2_large3
    # Now, we transform the rate of change of the imaginary part
    # back to spectral space on the larger grid.
    mul!(D_large3.tmp, D_large3.rfft_plan, tmp1_large3)
    # We copy the relevant coefficients to the smaller grid.
    ratio = size(D_small, 2) / size(D_large3, 2)
    for i in eachindex(dw)
        dw[i] = D_large3.tmp[i] * ratio
    end
    # Now, we transform the rate of change of the real part
    # back to spectral space on the larger grid.
    mul!(D_large3.tmp, D_large3.rfft_plan, tmp2_large3)
    for i in eachindex(dv)
        dv[i] = -D_large3.tmp[i] * ratio
    end

    return nothing
end
function rhs_nonstiff!(dq, q, equation::CubicNLS, ::FourierCollocationConserveMassEnergy, parameters, t)
    (; tmp1) = parameters
    (; β) = equation

    dv = real(dq, equation)
    dw = imag(dq, equation)

    v = real(q, equation)
    w = imag(q, equation)

    @. tmp1 = v^2 + w^2

    # dv = -β (v^2 + w^2) w
    @. dv = -β * tmp1 * w

    # dw = +β (v^2 + w^2) v
    @. dw = β * tmp1 * v

    return nothing
end

function mass(q, equation::CubicNLS, ::FourierGalerkin, parameters)
    (; D_small, D_large2, tmp1_large2, tmp2_large2) = parameters

    v = real(q, equation)
    w = imag(q, equation)

    # First, we copy the spectral coefficients of the real part
    # to the larger grid, pad them by zeros, and transform back to
    # physical space on the larger grid. Since we use brfft instead
    # of irfft, we need to normalize by the number of grid points -
    # of the grid where we computed the spectral coefficients originally
    # (i.e., the smaller grid).
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(v)] .= v ./ size(D_small, 2)
    mul!(tmp1_large2, D_large2.brfft_plan, D_large2.tmp)
    # Now, we do the same for the imaginary part.
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(w)] .= w ./ size(D_small, 2)
    mul!(tmp2_large2, D_large2.brfft_plan, D_large2.tmp)
    # Next, we compute the nonlinear term on the larger grid.
    @. tmp1_large2 = tmp1_large2^2 + tmp2_large2^2
    return integrate(tmp1_large2, D_large2)
end
function mass(q, equation::CubicNLS, ::FourierCollocation, parameters)
    (; D2, tmp1) = parameters

    v = real(q, equation)
    w = imag(q, equation)

    @. tmp1 = v^2 + w^2

    return integrate(tmp1, D2)
end

function momentum(q, equation::CubicNLS, ::FourierGalerkin, parameters)
    (; D_small, D_large2, tmp1_large2, tmp3_large2) = parameters

    v = real(q, equation)
    w = imag(q, equation)

    # First, we copy the spectral coefficients of the real part
    # to the larger grid, pad them by zeros, and transform back to
    # physical space on the larger grid. Since we use brfft instead
    # of irfft, we need to normalize by the number of grid points -
    # of the grid where we computed the spectral coefficients originally
    # (i.e., the smaller grid).
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(v)] .= v ./ size(D_small, 2)
    mul!(tmp1_large2, D_large2.brfft_plan, D_large2.tmp)
    # Now, we do the same for the imaginary part while computing
    # the derivative.
    # We do not use the special handling of the highest frequency mode
    # for an even number of grid points since we interpret the spectral
    # coefficients as truncated from a larger grid.
    # To compute the derivative, we need to multiply by size(D_small, 2).
    # This cancels with the division by size(D_small, 2) that we need
    # for the inverse transform using brfft.
    fill!(D_large2.tmp, 0)
    for i in eachindex(w)
        f = (i - 1) * im * D_small.jac
        D_large2.tmp[i] = w[i] * f
    end
    mul!(tmp3_large2, D_large2.brfft_plan, D_large2.tmp)
    # Next, we compute the nonlinear term on the larger grid.
    @. tmp1_large2 = 2 * tmp1_large2 * tmp3_large2
    return integrate(tmp1_large2, D_large2)
end
function momentum(q, equation::CubicNLS, ::FourierCollocation, parameters)
    (; D2, tmp1) = parameters
    D = D2.D1

    v = real(q, equation)
    w = imag(q, equation)

    mul!(tmp1, D, w)
    @. tmp1 = v * tmp1

    # 2 ∫ v w_x dx
    return 2 * integrate(tmp1, D2)
end

function mass_momentum(q, equation::CubicNLS, ::FourierGalerkin, parameters)
    (; D_small, D_large2, tmp1_large2, tmp2_large2, tmp3_large2) = parameters

    v = real(q, equation)
    w = imag(q, equation)

    # First, we copy the spectral coefficients of the real part
    # to the larger grid, pad them by zeros, and transform back to
    # physical space on the larger grid. Since we use brfft instead
    # of irfft, we need to normalize by the number of grid points -
    # of the grid where we computed the spectral coefficients originally
    # (i.e., the smaller grid).
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(v)] .= v ./ size(D_small, 2)
    mul!(tmp1_large2, D_large2.brfft_plan, D_large2.tmp)
    # Now, we do the same for the imaginary part.
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(w)] .= w ./ size(D_small, 2)
    mul!(tmp2_large2, D_large2.brfft_plan, D_large2.tmp)
    # Next, we compute the derivative of the imaginary part.
    # We do not use the special handling of the highest frequency mode
    # for an even number of grid points since we interpret the spectral
    # coefficients as truncated from a larger grid.
    # To compute the derivative, we need to multiply by size(D_small, 2).
    # This cancels with the division by size(D_small, 2) that we need
    # for the inverse transform using brfft.
    for i in eachindex(w)
        f = (i - 1) * im * D_small.jac
        D_large2.tmp[i] = w[i] * f
    end
    mul!(tmp3_large2, D_large2.brfft_plan, D_large2.tmp)

    # Next, we compute the nonlinear terms on the larger grid.
    @. tmp2_large2 = tmp1_large2^2 + tmp2_large2^2
    @. tmp1_large2 = 2 * tmp1_large2 * tmp3_large2
    # Return mass and momentum
    return integrate(tmp2_large2, D_large2), integrate(tmp1_large2, D_large2)
end
function mass_momentum(q, equation::CubicNLS, semidiscretization::FourierCollocation, parameters)
    return mass(q, equation, semidiscretization, parameters),
           momentum(q, equation, semidiscretization, parameters)
end

function energy(q, equation::CubicNLS, ::FourierGalerkin, parameters)
    (; D_small, D_large4, tmp1_large4, tmp2_large4, tmp3_large4) = parameters
    (; β) = equation

    v = real(q, equation)
    w = imag(q, equation)

    # First, we copy the spectral coefficients of the real part
    # to the larger grid, pad them by zeros, and transform back to
    # physical space on the larger grid. Since we use brfft instead
    # of irfft, we need to normalize by the number of grid points -
    # of the grid where we computed the spectral coefficients originally
    # (i.e., the smaller grid).
    fill!(D_large4.tmp, 0)
    D_large4.tmp[1:length(v)] .= v ./ size(D_small, 2)
    mul!(tmp1_large4, D_large4.brfft_plan, D_large4.tmp)
    # Now, we do the same for the imaginary part.
    fill!(D_large4.tmp, 0)
    D_large4.tmp[1:length(w)] .= w ./ size(D_small, 2)
    mul!(tmp2_large4, D_large4.brfft_plan, D_large4.tmp)
    # Next, we compute the nonlinear term on the larger grid.
    mul!(tmp3_large4, D_large4^2, tmp1_large4)
    @. tmp3_large4 = -tmp1_large4 * tmp3_large4
    kinetic = integrate(tmp3_large4, D_large4)

    mul!(tmp3_large4, D_large4^2, tmp2_large4)
    @. tmp3_large4 = -tmp2_large4 * tmp3_large4
    kinetic += integrate(tmp3_large4, D_large4)

    @. tmp3_large4 = (tmp1_large4^2 + tmp2_large4^2)^2
    potential = -0.5 * β * integrate(tmp3_large4, D_large4)

    return kinetic + potential
end
function energy(q, equation::CubicNLS, ::FourierCollocation, parameters)
    (; D2, tmp1) = parameters
    (; β) = equation

    v = real(q, equation)
    w = imag(q, equation)

    # Kinetic energy
    mul!(tmp1, D2, v)
    @. tmp1 = -v * tmp1
    kinetic = integrate(tmp1, D2)

    mul!(tmp1, D2, w)
    @. tmp1 = -w * tmp1
    kinetic += integrate(tmp1, D2)

    # Potential energy
    @. tmp1 = (v^2 + w^2)^2
    potential = -0.5 * β * integrate(tmp1, D2)

    return kinetic + potential
end

function setup(u_func, equation::CubicNLS, semidiscretization::FourierGalerkin, tspan, D)
    @assert D isa FourierDerivativeOperator

    # De-aliasing rule for degree p nonlinearities:
    # Use a grid with at least factor (p + 1) / 2 more points.
    # The NLS equation has at most a quartic nonlinearity in the
    # energy, for which we use the factor 5 / 2. However, several
    # other terms require only lower-order nonlinearities, e.g.,
    # quadratic nonlinearities in the mass and momentum. To improve
    # efficiency, we introduce specialized larger grids for them.
    D_small = D
    D_large2 = fourier_derivative_operator(xmin(D), xmax(D),
                    dealiased_number_of_nodes(2, size(D_small, 2)))
    D_large3 = fourier_derivative_operator(xmin(D), xmax(D),
                    dealiased_number_of_nodes(3, size(D_small, 2)))
    D_large4 = fourier_derivative_operator(xmin(D), xmax(D),
                    dealiased_number_of_nodes(4, size(D_small, 2)))

    x = grid(D_small)
    u0 = u_func.(tspan[begin], x, equation)
    v0 = real.(u0)
    w0 = imag.(u0)
    tmp1_small = similar(v0)
    tmp2_small = similar(v0)
    tmp1_large2 = similar(v0, size(D_large2, 2))
    tmp2_large2 = similar(v0, size(D_large2, 2))
    tmp3_large2 = similar(v0, size(D_large2, 2))
    tmp1_large3 = similar(v0, size(D_large3, 2))
    tmp2_large3 = similar(v0, size(D_large3, 2))
    tmp3_large3 = similar(v0, size(D_large3, 2))
    tmp1_large4 = similar(v0, size(D_large4, 2))
    tmp2_large4 = similar(v0, size(D_large4, 2))
    tmp3_large4 = similar(v0, size(D_large4, 2))

    q0 = vcat(D_small.rfft_plan * v0, D_small.rfft_plan * w0)
    q_tmp = similar(q0)

    parameters = (; equation, semidiscretization,
                    D_small, tmp1_small, tmp2_small,
                    D_large2, tmp1_large2, tmp2_large2, tmp3_large2,
                    D_large3, tmp1_large3, tmp2_large3, tmp3_large3,
                    D_large4, tmp1_large4, tmp2_large4, tmp3_large4,
                    q_tmp)
    return (; q0, parameters)
end
function setup(u_func, equation::CubicNLS, semidiscretization::FourierCollocation, tspan, D)
    @assert D isa FourierDerivativeOperator
    D2 = D^2

    x = grid(D2)
    u0 = u_func.(tspan[begin], x, equation)
    v0 = real.(u0)
    w0 = imag.(u0)
    q0 = vcat(v0, w0)

    tmp1 = similar(v0)

    parameters = (; equation, semidiscretization,
                    D2, tmp1)
    return (; q0, parameters)
end

function relaxation!(q, tmp, y, z, t, dt,
                     equation::CubicNLS, parameters,
                     ::NoProjection,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    @. q = q + dt * tmp
    return t + dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::CubicNLS, parameters,
                     ::MassProjection,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    @. q = q + dt * tmp
    mass_new = mass(q, parameters)
    q .= q .* sqrt(mass_old / mass_new)
    return t + dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::CubicNLS, parameters,
                     ::MassRelaxation,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    @. y = q - tmp
    mass_minus = mass(y, parameters)
    @. y = q + tmp
    mass_plus = mass(y, parameters)
    mass_tmp = mass(tmp, parameters)
    gamma = (mass_minus - mass_plus) / (2 * dt * mass_tmp)
    if !(0.1 < gamma < 10)
        error("Relaxation failed: gamma = $gamma at time $t. Try using a smaller time step size.")
    end
    gamma_dt = gamma * dt
    @. q = q + gamma_dt * tmp
    return t + gamma_dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::CubicNLS, parameters,
                     ::MassEnergyRelaxation,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    # First, we project the mass
    @. y = q + dt * tmp
    mass_new = mass(y, parameters)
    y .= y .* sqrt(mass_old / mass_new)

    # Next, we search for a solution conserving the energy
    # along the geodesic between q (old value) and y (new value)
    prob = NonlinearProblem{false}(1.0) do gamma, _
        @. z = (1 - gamma) * q + gamma * y
        mass_tmp = mass(z, parameters)
        z .= z .* sqrt(mass_old / mass_tmp)
        energy(z, parameters) - energy_old
    end
    sol = solve(prob, relaxation_alg)
    gamma = sol.u
    if !(0.1 < gamma < 10)
        error("Relaxation failed: gamma = $gamma at time $t. Try using a smaller time step size.")
    end
    @. z = (1 - gamma) * q + gamma * y
    q .= z .* sqrt(mass_old / mass(z, parameters))
    return t + gamma * dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::CubicNLS, parameters,
                     ::ProjectionEnergyRelaxation,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    # First, we project the mass
    @. y = q + dt * tmp
    project_mass_momentum!(y, equation, parameters, mass_old, momentum_old)

    # Next, we search for a solution conserving the energy
    # along the geodesic between q (old value) and y (new value)
    prob = NonlinearProblem{false}(1.0) do gamma, _
        @. z = (1 - gamma) * q + gamma * y
        project_mass_momentum!(z, equation, parameters, mass_old, momentum_old)
        energy(z, parameters) - energy_old
    end
    sol = solve(prob, relaxation_alg)
    gamma = sol.u
    if !(0.1 < gamma < 10)
        error("Relaxation failed: gamma = $gamma at time $t. Try using a smaller time step size.")
    end
    @. q = (1 - gamma) * q + gamma * y
    project_mass_momentum!(q, equation, parameters, mass_old, momentum_old)
    return t + gamma * dt
end
function project_mass_momentum!(q, equation::CubicNLS, parameters, mass_old, momentum_old)
    # Simplified projection method using the gradient/direction of
    # the current value
    (; D_small, q_tmp) = parameters

    mass_new, momentum_new = mass_momentum(q, parameters)

    v = real(q, equation)
    w = imag(q, equation)
    dv = real(q_tmp, equation)
    dw = imag(q_tmp, equation)
    for i in eachindex(v, w, dv, dw)
        f = (i - 1) * im * D_small.jac * size(D_small, 2)
        dv[i] = f * w[i]
        dw[i] = -f * v[i]
    end
    mass_grad, momentum_grad = mass_momentum(q_tmp, parameters)

    prob = NonlinearProblem{false}(SVector(1.0, 0.0)) do λμ, _
        λ, μ = λμ
        r1 = λ^2 * mass_new + μ^2 * mass_grad + 2 * λ * μ * momentum_new - mass_old
        r2 = λ^2 * momentum_new + μ^2 * momentum_grad + 2 * λ * μ * mass_grad - momentum_old
        return SVector(r1, r2)
    end
    sol = solve(prob, SimpleNewtonRaphson())
    λμ = sol.u
    @. q = λμ[1] * q + λμ[2] * q_tmp

    return nothing
end

one_soliton(t, x::Number, equation::CubicNLS) = cis(t) * sech(x)
get_β(::typeof(one_soliton)) = 2 * 1^2
domain(::typeof(one_soliton), ::CubicNLS) = (xmin = -35.0, xmax = 35.0)

function two_solitons(t, x::Number, equation::CubicNLS)
    num = 2 * exp(x + 9im * t) * (3 * exp(2x + 8im * t) + 3 * exp(4x + 8im * t) + exp(6x) + 1)
    den = 3 * exp(4x + 16im * t) + 4 * exp(2x + 8im * t) + 4 * exp(6x + 8im * t) + exp(8x + 8im * t) + exp(8im * t) + 3 * exp(4x)
    return num / den
end
get_β(::typeof(two_solitons)) = 2 * 2^2
domain(::typeof(two_solitons), ::CubicNLS) = (xmin = -35.0, xmax = 35.0)

function three_solitons(t, x::Number, equation::CubicNLS)
    # The appendix of Biswas and Ketcheson (2024) contains the following expression:
    # num = (80 * exp(7x + 49im * t)
    #        + 2 * exp(x + 25im * t)
    #        + 16 * exp(3x + 33im * t)
    #        + 36 * exp(5x + 33im * t)
    #        + 20 * exp(5x + 49im * t)
    #        + 32 * exp(7x + 25im * t)
    #        + 10 * exp(9x + 9im * t)
    #        + 90 * exp(9x + 41im * t)
    #        + 40 * exp(9x + 57im * t)
    #        + 32 * exp(11x + 25im * t)
    #        + 80 * exp(11x + 49im * t)
    #        + 32 * exp(13x + 33im * t)
    #        + 20 * exp(13x + 49im * t)
    #        + 16 * exp(15x + 33im * t)
    #        + 2 * exp(17x + 25im * t))
    # den = (64 * exp(12x + 24im * t)
    #        + 36 * exp(8x + 24im * t)
    #        + 18 * exp(4x + 16im * t)
    #        + 64 * exp(6x + 24im * t)
    #        + 45 * exp(10x + 40im * t)
    #        + 10 * exp(12x + 48im * t)
    #        + 45 * exp(8x + 40im * t)
    #        + 18 * exp(4x + 32im * t)
    #        + 10 * exp(6x + 48im * t)
    #        + 9 * exp(2x + 24im * t)
    #        + 45 * exp(8x + 8im * t)
    #        + 45 * exp(10x + 8im * t)
    #        + 36 * exp(10x + 24im * t)
    #        + 18 * exp(14x + 16im * t)
    #        + 18 * exp(14x + 32im * t)
    #        + 9 * exp(16x + 24im * t)
    #        + exp(18x + 24im * t)
    #        + exp(24im * t)
    #        + 10 * exp(6x)
    #        + 10 * exp(12x))
    # However, this is wrong since it is not even symmetric in x for t = 0.
    # Their reproducibility repository led me to the following version.
    # num = (2*(3*exp(t*25*im)*exp(x) + 15*exp(t*9*im)*exp(9*x) + 48*exp(t*25*im)*exp(7*x) + 48*exp(t*25*im)*exp(11*x) + 24*exp(t*33*im)*exp(3*x) + 54*exp(t*33*im)*exp(5*x) + 3*exp(t*25*im)*exp(17*x) + 54*exp(t*33*im)*exp(13*x) + 24*exp(t*33*im)*exp(15*x) + 135*exp(t*41*im)*exp(9*x) + 30*exp(t*49*im)*exp(5*x) + 120*exp(t*49*im)*exp(7*x) + 120*exp(t*49*im)*exp(11*x) + 30*exp(t*49*im)*exp(13*x) + 60*exp(t*57*im)*exp(9*x)))
    # den = (3*(exp(t*24*im) + 10*exp(6*x) + 10*exp(12*x) + 45*exp(t*8*im)*exp(8*x) + 45*exp(t*8*im)*exp(10*x) + 18*exp(t*16*im)*exp(4*x) + 9*exp(t*24*im)*exp(2*x) + 18*exp(t*16*im)*exp(14*x) + 64*exp(t*24*im)*exp(6*x) + 36*exp(t*24*im)*exp(8*x) + 36*exp(t*24*im)*exp(10*x) + 64*exp(t*24*im)*exp(12*x) + 18*exp(t*32*im)*exp(4*x) + 9*exp(t*24*im)*exp(16*x) + exp(t*24*im)*exp(18*x) + 18*exp(t*32*im)*exp(14*x) + 45*exp(t*40*im)*exp(8*x) + 45*exp(t*40*im)*exp(10*x) + 10*exp(t*48*im)*exp(6*x) + 10*exp(t*48*im)*exp(12*x)))
    # Performing common subexpression elimination manually gives:
    exp_t_8_im = cis(8 * t)
    exp_t_16_im = cis(16 * t)
    exp_t_24_im = cis(24 * t)
    exp_t_25_im = cis(25 * t)
    exp_t_32_im = cis(32 * t)
    exp_t_33_im = cis(33 * t)
    exp_t_40_im = cis(40 * t)
    exp_t_48_im = cis(48 * t)
    exp_t_49_im = cis(49 * t)
    exp_4_x = exp(4 * x)
    exp_5_x = exp(5 * x)
    exp_6_x = exp(6 * x)
    exp_7_x = exp(7 * x)
    exp_8_x = exp(8 * x)
    exp_9_x = exp(9 * x)
    exp_10_x = exp(10 * x)
    exp_11_x = exp(11 * x)
    exp_12_x = exp(12 * x)
    exp_13_x = exp(13 * x)
    exp_14_x = exp(14 * x)
    num = (2*(3*exp_t_25_im*exp(x) + 15*cis(t*9)*exp_9_x + 48*exp_t_25_im*exp_7_x + 48*exp_t_25_im*exp_11_x + 24*exp_t_33_im*exp(3*x) + 54*exp_t_33_im*exp_5_x + 3*exp_t_25_im*exp(17*x) + 54*exp_t_33_im*exp_13_x + 24*exp_t_33_im*exp(15*x) + 135*cis(t*41)*exp_9_x + 30*exp_t_49_im*exp_5_x + 120*exp_t_49_im*exp_7_x + 120*exp_t_49_im*exp_11_x + 30*exp_t_49_im*exp_13_x + 60*cis(t*57)*exp_9_x))
    den = (3*(exp_t_24_im + 10*exp_6_x + 10*exp_12_x + 45*exp_t_8_im*exp_8_x + 45*exp_t_8_im*exp_10_x + 18*exp_t_16_im*exp_4_x + 9*exp_t_24_im*exp(2*x) + 18*exp_t_16_im*exp_14_x + 64*exp_t_24_im*exp_6_x + 36*exp_t_24_im*exp_8_x + 36*exp_t_24_im*exp_10_x + 64*exp_t_24_im*exp_12_x + 18*exp_t_32_im*exp_4_x + 9*exp_t_24_im*exp(16*x) + exp_t_24_im*exp(18*x) + 18*exp_t_32_im*exp_14_x + 45*exp_t_40_im*exp_8_x + 45*exp_t_40_im*exp_10_x + 10*exp_t_48_im*exp_6_x + 10*exp_t_48_im*exp_12_x))
    return num / den
end
get_β(::typeof(three_solitons)) = 2 * 3^2
domain(::typeof(three_solitons), ::CubicNLS) = (xmin = -35.0, xmax = 35.0)

function one_moving_gray_soliton(t, x::Number, equation::CubicNLS)
    # Parameters of the gray soliton
    b0 = 1.5 # background mass density
    b1 = 1.0 # minimum mass density
    c1 = 2 * sqrt(2) # speed of the soliton

    κ = (c1 - sqrt(2 * b1)) / 2
    ω = b0 - (c1^2 - 2 * b1) / 4

    x0 = 0.0
    (; xmin, xmax) = domain(one_moving_gray_soliton, equation)

    # Wrapping the solution in the periodic domain
    x_t = mod(x - x0 - c1 * t - xmin, xmax - xmin) + xmin
    u = sqrt(b0) * cis(κ * x_t - ω * t) * (im * sqrt(b1 / b0) + sqrt(1 - b1 / b0) * tanh(sqrt((b0 - b1) / 2) * x_t))
    return u * cis(π / 2)
end
get_β(::typeof(one_moving_gray_soliton)) = -1
function domain(::typeof(one_moving_gray_soliton), ::CubicNLS)
    # The following solution has been obtained by setting this value
    # temporarily to 90.0 and executing
    # julia> prob = NonlinearProblem(31) do x, _
    #         abs(one_moving_gray_soliton(0.0, x, CubicNLS(-1)) - one_moving_gray_soliton(0.0, -x, CubicNLS(-1)))
    #     end; sol = solve(prob, SimpleNewtonRaphson())
    xmin = -31.970600318475647
    xmax = +31.970600318475647
    return (; xmin, xmax)
end

function two_gray_solitons(t, x, equation::CubicNLS)
    # Parameters of the gray solitons
    a3 = 1.5 # background mass density
    a1 = 1.0 # minimum mass density

    mu = 4 * sqrt(a1 * (a3 - a1))
    p = sqrt(a3 - a1)
    num = (2 * a3 - 4 * a1) * cosh(mu * t / 2) - 2 * sqrt(a1 * a3) * cosh(2 * p * x / sqrt(2)) - im * mu * sinh(mu * t / 2)
    den = 2 * sqrt(a3) * cosh(mu * t / 2) + 2 * sqrt(a1) * cosh(2 * p * x / sqrt(2))
    u = num / den * cis(-a3 * t)
    return u
end
get_β(::typeof(two_gray_solitons)) = -1
function domain(::typeof(two_gray_solitons), ::CubicNLS)
    xmin = -200.0
    xmax = +200.0
    return (; xmin, xmax)
end

function two_moving_gray_solitons(t, x, equation::CubicNLS)
    k = 2.0
    u = two_gray_solitons(t, x - 2 * k * t, equation)
    return cis(k * x - k^2 * t) * u
end
get_β(::typeof(two_moving_gray_solitons)) = -1
function domain(::typeof(two_moving_gray_solitons), ::CubicNLS)
    xmin = -409.97784129346803 # k = 2.0, t in (-70, 70)
    xmax = +409.97784129346803
    return (; xmin, xmax)
end


#####################################################################
# HyperbolizedCubicNLS discretization

struct HyperbolizedCubicNLS{T} <: AbstractEquation
    β::T
    τ::T
end
function HyperbolizedCubicNLS(β, τ)
    β, τ = promote(β, τ)
    return HyperbolizedCubicNLS{typeof(β)}(β, τ)
end

Base.real(q, equation::HyperbolizedCubicNLS) = get_qi(q, equation, 0)
Base.imag(q, equation::HyperbolizedCubicNLS) = get_qi(q, equation, 1)
function get_qi(q, equation::HyperbolizedCubicNLS, i)
    N = length(q) ÷ 4
    return view(q, (i * N + 1):((i + 1) * N))
end

function density(q, equation::HyperbolizedCubicNLS, parameters)
    return density(q, equation, parameters.semidiscretization, parameters)
end
function density(q, equation::HyperbolizedCubicNLS, ::FourierGalerkin, parameters)
    (; D_small) = parameters
    v = D_small.brfft_plan * (real(q, equation) ./ size(D_small, 2))
    w = D_small.brfft_plan * (imag(q, equation) ./ size(D_small, 2))
    return @. v^2 + w^2
end
function density(q, equation::HyperbolizedCubicNLS, ::FourierCollocation, parameters)
    v = real(q, equation)
    w = imag(q, equation)
    return @. v^2 + w^2
end

function rhs_stiff!(dq, q, equation::HyperbolizedCubicNLS, ::FourierGalerkin, parameters, t)
    (; D_small) = parameters
    (; τ) = equation

    dv = get_qi(dq, equation, 0)
    dw = get_qi(dq, equation, 1)
    dvx = get_qi(dq, equation, 2)
    dwx = get_qi(dq, equation, 3)

    v = get_qi(q, equation, 0)
    w = get_qi(q, equation, 1)
    vx = get_qi(q, equation, 2)
    wx = get_qi(q, equation, 3)

    # We compute the discrete derivative operator on the smaller grid.
    # We do not use the special handling of the highest frequency mode
    # for an even number of grid points since we interpret the spectral
    # coefficients as truncated from a larger grid.
    for i in eachindex(dv)
        f = (i - 1) * im * D_small.jac * size(D_small, 2)

        dv[i] = -f * wx[i]
        dw[i] = +f * vx[i]
        dvx[i] = (f * w[i] - wx[i]) / τ
        dwx[i] = (vx[i] - f * v[i]) / τ
    end

    return nothing
end

struct StiffOperatorHyperbolizedCubicNLS <: AbstractStiffOperator end
Base.:*(::Number, op::StiffOperatorHyperbolizedCubicNLS) = op
Base.:-(::UniformScaling, op::StiffOperatorHyperbolizedCubicNLS) = op
Base.iszero(op::StiffOperatorHyperbolizedCubicNLS) = false

operator(::typeof(rhs_stiff!), equation::HyperbolizedCubicNLS, ::FourierGalerkin, parameters) = StiffOperatorHyperbolizedCubicNLS()

function rhs_nonstiff!(dq, q, equation::HyperbolizedCubicNLS, ::FourierGalerkin, parameters, t)
    (; D_small, D_large4, tmp1_large4, tmp2_large4, tmp3_large4) = parameters
    (; β) = equation

    dv = get_qi(dq, equation, 0)
    dw = get_qi(dq, equation, 1)
    dvx = get_qi(dq, equation, 2)
    dwx = get_qi(dq, equation, 3)

    v = get_qi(q, equation, 0)
    w = get_qi(q, equation, 1)

    # First, we copy the spectral coefficients of the real part
    # to the larger grid, pad them by zeros, and transform back to
    # physical space on the larger grid. Since we use brfft instead
    # of irfft, we need to normalize by the number of grid points -
    # of the grid where we computed the spectral coefficients originally
    # (i.e., the smaller grid).
    fill!(D_large4.tmp, 0)
    D_large4.tmp[1:length(v)] .= v ./ size(D_small, 2)
    mul!(tmp1_large4, D_large4.brfft_plan, D_large4.tmp)
    # Now, we do the same for the imaginary part.
    fill!(D_large4.tmp, 0)
    D_large4.tmp[1:length(w)] .= w ./ size(D_small, 2)
    mul!(tmp2_large4, D_large4.brfft_plan, D_large4.tmp)
    # Next, we compute the nonlinear term on the larger grid.
    @. tmp3_large4 = β * (tmp1_large4^2 + tmp2_large4^2)
    @. tmp1_large4 = tmp3_large4 * tmp1_large4
    @. tmp2_large4 = tmp3_large4 * tmp2_large4
    # Now, we transform the rate of change of the imaginary part
    # back to spectral space on the larger grid.
    mul!(D_large4.tmp, D_large4.rfft_plan, tmp1_large4)
    # We copy the relevant coefficients to the smaller grid.
    ratio = size(D_small, 2) / size(D_large4, 2)
    for i in eachindex(dw)
        dw[i] = D_large4.tmp[i] * ratio
    end
    # Now, we transform the rate of change of the real part
    # back to spectral space on the larger grid.
    mul!(D_large4.tmp, D_large4.rfft_plan, tmp2_large4)
    for i in eachindex(dv)
        dv[i] = -D_large4.tmp[i] * ratio
    end

    @. dvx = 0
    @. dwx = 0

    return nothing
end

function mass(q, equation::HyperbolizedCubicNLS, ::FourierGalerkin, parameters)
    (; D_small, D_large2, tmp1_large2, tmp2_large2, tmp3_large2) = parameters
    (; τ) = equation

    v = get_qi(q, equation, 0)
    w = get_qi(q, equation, 1)
    vx = get_qi(q, equation, 2)
    wx = get_qi(q, equation, 3)

    # First, we copy the spectral coefficients of the real part
    # to the larger grid, pad them by zeros, and transform back to
    # physical space on the larger grid. Since we use brfft instead
    # of irfft, we need to normalize by the number of grid points -
    # of the grid where we computed the spectral coefficients originally
    # (i.e., the smaller grid).
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(v)] .= v ./ size(D_small, 2)
    mul!(tmp1_large2, D_large2.brfft_plan, D_large2.tmp)
    # Now, we do the same for the imaginary part.
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(w)] .= w ./ size(D_small, 2)
    mul!(tmp2_large2, D_large2.brfft_plan, D_large2.tmp)
    # Next, we compute the original nonlinear term on the larger grid.
    @. tmp1_large2 = tmp1_large2^2 + tmp2_large2^2
    # Next, we do the same for the derivative approximations.
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(vx)] .= vx ./ size(D_small, 2)
    mul!(tmp2_large2, D_large2.brfft_plan, D_large2.tmp)
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(wx)] .= wx ./ size(D_small, 2)
    mul!(tmp3_large2, D_large2.brfft_plan, D_large2.tmp)
    # Next, we add the additional nonlinear term on the larger grid.
    @. tmp1_large2 = tmp1_large2 + τ * (tmp2_large2^2 + tmp3_large2^2)
    return integrate(tmp1_large2, D_large2)
end

function momentum(q, equation::HyperbolizedCubicNLS, ::FourierGalerkin, parameters)
    (; D_small, D_large2, tmp1_large2, tmp2_large2, tmp3_large2) = parameters
    (; τ) = equation

    v = get_qi(q, equation, 0)
    w = get_qi(q, equation, 1)
    vx = get_qi(q, equation, 2)
    wx = get_qi(q, equation, 3)

    # First, we copy the spectral coefficients of the real part
    # to the larger grid, pad them by zeros, and transform back to
    # physical space on the larger grid. Since we use brfft instead
    # of irfft, we need to normalize by the number of grid points -
    # of the grid where we computed the spectral coefficients originally
    # (i.e., the smaller grid).
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(v)] .= v ./ size(D_small, 2)
    mul!(tmp1_large2, D_large2.brfft_plan, D_large2.tmp)
    # Now, we do the same for the imaginary part while computing
    # the derivative.
    # We do not use the special handling of the highest frequency mode
    # for an even number of grid points since we interpret the spectral
    # coefficients as truncated from a larger grid.
    # To compute the derivative, we need to multiply by size(D_small, 2).
    # This cancels with the division by size(D_small, 2) that we need
    # for the inverse transform using brfft.
    fill!(D_large2.tmp, 0)
    for i in eachindex(w)
        f = (i - 1) * im * D_small.jac
        D_large2.tmp[i] = w[i] * f
    end
    mul!(tmp3_large2, D_large2.brfft_plan, D_large2.tmp)
    # Next, we compute the nonlinear term on the larger grid.
    @. tmp1_large2 = 2 * tmp1_large2 * tmp3_large2
    # Now we do the same for the part proportional to τ.
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(vx)] .= vx ./ size(D_small, 2)
    mul!(tmp2_large2, D_large2.brfft_plan, D_large2.tmp)
    fill!(D_large2.tmp, 0)
    for i in eachindex(w)
        f = (i - 1) * im * D_small.jac
        D_large2.tmp[i] = wx[i] * f
    end
    mul!(tmp3_large2, D_large2.brfft_plan, D_large2.tmp)
    # Next, we add the additional nonlinear term on the larger grid.
    @. tmp1_large2 = tmp1_large2 + τ * tmp2_large2 * tmp3_large2
    return integrate(tmp1_large2, D_large2)
end
function mass_momentum(q, equation::HyperbolizedCubicNLS, semidiscretization::FourierGalerkin, parameters)
    # This could be made more efficient.
    return mass(q, equation, semidiscretization, parameters),
           momentum(q, equation, semidiscretization, parameters)
end

function energy(q, equation::HyperbolizedCubicNLS, ::FourierGalerkin, parameters)
    (; D_small, D_large4, tmp1_large4, tmp2_large4, tmp3_large4) = parameters
    (; β) = equation

    v = get_qi(q, equation, 0)
    w = get_qi(q, equation, 1)
    vx = get_qi(q, equation, 2)
    wx = get_qi(q, equation, 3)

    # ν
    fill!(D_large4.tmp, 0)
    D_large4.tmp[1:length(vx)] .= vx ./ size(D_small, 2)
    mul!(tmp2_large4, D_large4.brfft_plan, D_large4.tmp)

    # v_x
    fill!(D_large4.tmp, 0)
    for i in eachindex(v)
        f = (i - 1) * im * D_small.jac
        D_large4.tmp[i] = v[i] * f
    end
    mul!(tmp3_large4, D_large4.brfft_plan, D_large4.tmp)

    # 2 ν v_x - ν^2
    @. tmp1_large4 = 2 * tmp2_large4 * tmp3_large4 - tmp2_large4^2

    # ω
    fill!(D_large4.tmp, 0)
    D_large4.tmp[1:length(wx)] .= wx ./ size(D_small, 2)
    mul!(tmp2_large4, D_large4.brfft_plan, D_large4.tmp)

    # w_x
    fill!(D_large4.tmp, 0)
    for i in eachindex(w)
        f = (i - 1) * im * D_small.jac
        D_large4.tmp[i] = w[i] * f
    end
    mul!(tmp3_large4, D_large4.brfft_plan, D_large4.tmp)

    # 2 ω w_x - ω^2
    @. tmp1_large4 = tmp1_large4 + 2 * tmp2_large4 * tmp3_large4 - tmp2_large4^2
    kinetic = integrate(tmp1_large4, D_large4)

    # v
    fill!(D_large4.tmp, 0)
    D_large4.tmp[1:length(v)] .= v ./ size(D_small, 2)
    mul!(tmp2_large4, D_large4.brfft_plan, D_large4.tmp)

    # w
    fill!(D_large4.tmp, 0)
    D_large4.tmp[1:length(w)] .= w ./ size(D_small, 2)
    mul!(tmp3_large4, D_large4.brfft_plan, D_large4.tmp)

    # - 0.5 β (v^2 + w^2)^2
    @. tmp1_large4 = (tmp2_large4^2 + tmp3_large4^2)^2
    potential = -0.5 * β * integrate(tmp1_large4, D_large4)

    return kinetic + potential
end

function setup(u_func, equation::HyperbolizedCubicNLS, semidiscretization::FourierGalerkin, tspan, D)
    @assert D isa FourierDerivativeOperator

    # De-aliasing rule for degree p nonlinearities:
    # Use a grid with at least factor (p + 1) / 2 more points.
    # The NLS equation has at most a quartic nonlinearity in the
    # energy, for which we use the factor 5 / 2. However, several
    # other terms require only lower-order nonlinearities, e.g.,
    # quadratic nonlinearities in the mass and momentum. To improve
    # efficiency, we introduce specialized larger grids for them.
    D_small = D
    D_large2 = fourier_derivative_operator(xmin(D), xmax(D),
                    dealiased_number_of_nodes(2, size(D_small, 2)))
    D_large4 = fourier_derivative_operator(xmin(D), xmax(D),
                    dealiased_number_of_nodes(4, size(D_small, 2)))

    x = grid(D_small)
    u0 = u_func.(tspan[begin], x, CubicNLS(equation.β))
    v0 = real.(u0)
    w0 = imag.(u0)
    vx0 = D_small * v0
    wx0 = D_small * w0
    tmp1_small = similar(v0)
    tmp2_small = similar(v0)
    tmp1_large2 = similar(v0, size(D_large2, 2))
    tmp2_large2 = similar(v0, size(D_large2, 2))
    tmp3_large2 = similar(v0, size(D_large2, 2))
    tmp1_large4 = similar(v0, size(D_large4, 2))
    tmp2_large4 = similar(v0, size(D_large4, 2))
    tmp3_large4 = similar(v0, size(D_large4, 2))

    q0 = vcat(D_small.rfft_plan * v0, D_small.rfft_plan * w0,
              D_small.rfft_plan * vx0, D_small.rfft_plan * wx0)
    q_tmp = similar(q0)

    parameters = (; equation, semidiscretization,
                    D_small, tmp1_small, tmp2_small,
                    D_large2, tmp1_large2, tmp2_large2, tmp3_large2,
                    D_large4, tmp1_large4, tmp2_large4, tmp3_large4,
                    q_tmp)
    return (; q0, parameters)
end

function relaxation!(q, tmp, y, z, t, dt,
                     equation::HyperbolizedCubicNLS, parameters,
                     ::NoProjection,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    @. q = q + dt * tmp
    return t + dt
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::HyperbolizedCubicNLS, parameters,
                     ::MassProjection,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    # Simplified projection method using the gradient/direction of
    # the current value
    @. q = q + dt * tmp
    project_mass!(q, equation, parameters, mass_old)
    return t + dt
end
function project_mass!(q, equation, parameters, mass_old)
    (; τ) = equation
    (; D_small, D_large2, tmp1_large2, tmp2_large2) = parameters

    v = get_qi(q, equation, 0)
    w = get_qi(q, equation, 1)
    vx = get_qi(q, equation, 2)
    wx = get_qi(q, equation, 3)

    # q2 = integrate(abs2, v, D1) + integrate(abs2, w, D1)
    # Same computation as for the mass(...)
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(v)] .= v ./ size(D_small, 2)
    mul!(tmp1_large2, D_large2.brfft_plan, D_large2.tmp)

    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(w)] .= w ./ size(D_small, 2)
    mul!(tmp2_large2, D_large2.brfft_plan, D_large2.tmp)

    @. tmp1_large2 = tmp1_large2^2 + tmp2_large2^2
    q2 = integrate(tmp1_large2, D_large2)

    # p2 = integrate(abs2, vx, D1) + integrate(abs2, wx, D1)
    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(vx)] .= vx ./ size(D_small, 2)
    mul!(tmp1_large2, D_large2.brfft_plan, D_large2.tmp)

    fill!(D_large2.tmp, 0)
    D_large2.tmp[1:length(wx)] .= wx ./ size(D_small, 2)
    mul!(tmp2_large2, D_large2.brfft_plan, D_large2.tmp)

    @. tmp1_large2 = tmp1_large2^2 + tmp2_large2^2
    p2 = integrate(tmp1_large2, D_large2)

    c = mass_old
    factor_q = (p2 * (τ - 1) * τ^2 + sqrt(-p2 * q2 * (τ - 1)^2 * τ + c * (q2 + p2 * τ^3))) / (q2 + p2 * τ^3)
    factor_p = (q2 * (1 - τ) + τ * sqrt(-p2 * q2 * (τ - 1)^2 * τ + c * (q2 + p2 * τ^3))) / (q2 + p2 * τ^3)
    @. v = factor_q * v
    @. w = factor_q * w
    @. vx = factor_p * vx
    @. wx = factor_p * wx

    return nothing
end
function relaxation!(q, tmp, y, z, t, dt,
                     equation::HyperbolizedCubicNLS, parameters,
                     ::MassEnergyRelaxation,
                     relaxation_alg, relaxation_tol,
                     mass_old, momentum_old, energy_old)
    # First, we project the mass
    @. y = q + dt * tmp
    project_mass!(y, equation, parameters, mass_old)

    # Next, we search for a solution conserving the energy
    # along the geodesic between q (old value) and y (new value)
    prob = NonlinearProblem{false}(1.0) do gamma, _
        @. z = (1 - gamma) * q + gamma * y
        project_mass!(z, equation, parameters, mass_old)
        energy(z, parameters) - energy_old
    end
    sol = solve(prob, relaxation_alg)
    gamma = sol.u
    if !(0.1 < gamma < 10)
        error("Relaxation failed: gamma = $gamma at time $t. Try using a smaller time step size.")
    end
    @. z = (1 - gamma) * q + gamma * y
    project_mass!(z, equation, parameters, mass_old)
    @. q = z
    return t + gamma * dt
end



#####################################################################
# Numerical experiments reported in the paper

function semidiscrete_conservation()
    fig = Figure(size = (1200, 650)) # default size is (600, 450)

    ax_bbm1 = Axis(fig[1, 1];
                   xlabel = L"Time $t$",
                   ylabel = "Change of invariants",
                   title = "BBM equation")
    ax_kdv1 = Axis(fig[1, 2];
                   xlabel = L"Time $t$",
                   ylabel = "Change of invariants",
                   title = "KdV equation")
    ax_nls1 = Axis(fig[1, 3];
                   xlabel = L"Time $t$",
                   ylabel = "Change of invariants",
                   title = "Cubic NLS equation")
    ax_bbm2 = Axis(fig[2, 1];
                   xlabel = L"Time $t$",
                   ylabel = "Change of invariants",
                   title = "BBM equation")
    ax_kdv2 = Axis(fig[2, 2];
                   xlabel = L"Time $t$",
                   ylabel = "Change of invariants",
                   title = "KdV equation")
    ax_nls2 = Axis(fig[2, 3];
                   xlabel = L"Time $t$",
                   ylabel = "Change of invariants",
                   title = "Cubic NLS equation")

    linkyaxes!(ax_bbm1, ax_kdv1, ax_nls1)
    hideydecorations!(ax_kdv1; grid = false)
    hideydecorations!(ax_nls1; grid = false)

    linkyaxes!(ax_bbm2, ax_kdv2, ax_nls2)
    hideydecorations!(ax_kdv2; grid = false)
    hideydecorations!(ax_nls2; grid = false)

    linkxaxes!(ax_bbm1, ax_bbm2)
    hidexdecorations!(ax_bbm1; grid = false)

    linkxaxes!(ax_kdv1, ax_kdv2)
    hidexdecorations!(ax_kdv1; grid = false)

    linkxaxes!(ax_nls1, ax_nls2)
    hidexdecorations!(ax_nls1; grid = false)

    # Setup callback computing the error of the invariants
    series_t = Vector{Float64}()
    series_mass = Vector{Float64}()
    series_momentum = Vector{Float64}()
    series_energy = Vector{Float64}()
    callback = let series_t = series_t, series_mass = series_mass, series_momentum = series_momentum, series_energy = series_energy
        function (q, parameters, t)
            push!(series_t, t)
            push!(series_mass, mass(q, parameters))
            push!(series_momentum, momentum(q, parameters))
            push!(series_energy, energy(q, parameters))
            return nothing
        end
    end

    @info "BBM equation"
    for (ax, N) in [(ax_bbm1, 2^5 - 1), (ax_bbm2, 2^5)]
        initial_condition = two_solitary_waves
        equation = BBM()
        dt = 5.0e-3
        tspan = (0.0, 400.0)
        alg = KenCarpARK548()

        ax.title = "BBM, $(N) nodes"
        (; xmin, xmax) = domain(initial_condition, equation)
        D = fourier_derivative_operator(xmin, xmax, N)

        (; q0, parameters) = setup(initial_condition, equation,
                                   FourierGalerkin(), tspan, D)

        empty!(series_t)
        empty!(series_mass)
        empty!(series_momentum)
        empty!(series_energy)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, callback)

        lines!(ax, series_t, change(series_energy); label = "energy")
        lines!(ax, series_t, change(series_momentum); label = "momentum")
        lines!(ax, series_t, change(series_mass); label = "mass")

        axislegend(ax; position = :rt, framevisible = false, nbanks = 2)
    end

    @info "KdV equation"
    for (ax, N) in [(ax_kdv1, 2^5 - 1), (ax_kdv2, 2^5)]
        initial_condition = two_solitons
        equation = KdV()
        dt = 1.0e-2
        tspan = (0.0, 350.0)
        alg = KenCarpARK548()

        ax.title = "KdV, $(N) nodes"
        (; xmin, xmax) = domain(initial_condition, equation)
        D = fourier_derivative_operator(xmin, xmax, N)

        (; q0, parameters) = setup(initial_condition, equation,
                                   FourierGalerkin(), tspan, D)

        empty!(series_t)
        empty!(series_mass)
        empty!(series_momentum)
        empty!(series_energy)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, callback)

        lines!(ax, series_t, change(series_energy); label = "energy")
        lines!(ax, series_t, change(series_momentum); label = "momentum")
        lines!(ax, series_t, change(series_mass); label = "mass")

        axislegend(ax; position = :rb, framevisible = false, nbanks = 2)
    end

    @info "NLS equation"
    for (ax, N) in [(ax_nls1, 2^5 - 1), (ax_nls2, 2^5)]
        initial_condition = two_solitons
        equation = CubicNLS(get_β(initial_condition))
        dt = 1.0e-4
        tspan = (0.0, 10.0)
        alg = KenCarpARK548()

        ax.title = "NLS, $(N) nodes"
        (; xmin, xmax) = domain(initial_condition, equation)
        D = fourier_derivative_operator(xmin, xmax, N)

        (; q0, parameters) = setup(initial_condition, equation,
                                   FourierGalerkin(), tspan, D)

        empty!(series_t)
        empty!(series_mass)
        empty!(series_momentum)
        empty!(series_energy)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, callback)

        lines!(ax, series_t, change(series_energy); label = "energy")
        lines!(ax, series_t, change(series_momentum); label = "momentum")
        lines!(ax, series_t, change(series_mass); label = "mass")

        position = isodd(N) ? :rb : :rt
        axislegend(ax; position, framevisible = false, nbanks = 2)
    end

    filename = joinpath(FIGDIR, "semidiscrete_conservation.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function  fully_discrete_conservation_two_waves()
    fig = Figure(size = (1200, 650)) # default size is (600, 450)

    ax_bbm1 = Axis(fig[1, 1];
                   xlabel = L"Time $t$",
                   ylabel = "Change of invariants",
                   title = "BBM, mass-energy relax.")
    ax_kdv1 = Axis(fig[1, 2];
                   xlabel = L"Time $t$",
                   ylabel = "Change of invariants",
                   title = "KdV, mass-energy relax.")
    ax_nls1 = Axis(fig[1, 3];
                   xlabel = L"Time $t$",
                   ylabel = "Change of invariants",
                   title = "NLS, mass-energy relax.")
    ax_bbm2 = Axis(fig[2, 1];
                   xlabel = L"Time $t$",
                   ylabel = "Change of invariants",
                   title = "BBM, full relaxation")
    ax_kdv2 = Axis(fig[2, 2];
                   xlabel = L"Time $t$",
                   ylabel = "Change of invariants",
                   title = "KdV, full relaxation")
    ax_nls2 = Axis(fig[2, 3];
                   xlabel = L"Time $t$",
                   ylabel = "Change of invariants",
                   title = "NLS, full relaxation")

    linkyaxes!(ax_bbm1, ax_kdv1, ax_nls1)
    hideydecorations!(ax_kdv1; grid = false)
    hideydecorations!(ax_nls1; grid = false)

    linkyaxes!(ax_bbm2, ax_kdv2, ax_nls2)
    hideydecorations!(ax_kdv2; grid = false)
    hideydecorations!(ax_nls2; grid = false)

    linkxaxes!(ax_bbm1, ax_bbm2)
    hidexdecorations!(ax_bbm1; grid = false)

    linkxaxes!(ax_kdv1, ax_kdv2)
    hidexdecorations!(ax_kdv1; grid = false)

    linkxaxes!(ax_nls1, ax_nls2)
    hidexdecorations!(ax_nls1; grid = false)

    # Setup callback computing the error of the invariants
    series_t = Vector{Float64}()
    series_mass = Vector{Float64}()
    series_momentum = Vector{Float64}()
    series_energy = Vector{Float64}()
    callback = let series_t = series_t, series_mass = series_mass, series_momentum = series_momentum, series_energy = series_energy
        function (q, parameters, t)
            push!(series_t, t)
            push!(series_mass, mass(q, parameters))
            push!(series_momentum, momentum(q, parameters))
            push!(series_energy, energy(q, parameters))
            return nothing
        end
    end

    alg = KenCarpARK548()

    @info "BBM equation"
    for (ax, relaxation) in [(ax_bbm1, MassEnergyRelaxation()),
                             (ax_bbm2, ProjectionEnergyRelaxation())]
        initial_condition = two_solitary_waves
        equation = BBM()
        N = 2^8
        dt = 0.5
        tspan = (0.0, 5000.0)

        (; xmin, xmax) = domain(initial_condition, equation)
        D = fourier_derivative_operator(xmin, xmax, N)

        (; q0, parameters) = setup(initial_condition, equation,
                                   FourierGalerkin(), tspan, D)

        empty!(series_t)
        empty!(series_mass)
        empty!(series_momentum)
        empty!(series_energy)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, callback,
                               relaxation, relaxation_tol = 1.0e-15)

        lines!(ax, series_t, change(series_energy); label = "energy")
        lines!(ax, series_t, change(series_momentum); label = "momentum")
        lines!(ax, series_t, change(series_mass); label = "mass")

        axislegend(ax; position = :rb, framevisible = false, nbanks = 2)
    end

    @info "KdV equation"
    for (ax, relaxation) in [(ax_kdv1, MassEnergyRelaxation()),
                             (ax_kdv2, ProjectionEnergyRelaxation())]
        initial_condition = two_solitons
        equation = KdV()
        N = 2^8
        dt = 0.1
        tspan = (0.0, 1000.0)

        (; xmin, xmax) = domain(initial_condition, equation)
        D = fourier_derivative_operator(xmin, xmax, N)

        (; q0, parameters) = setup(initial_condition, equation,
                                   FourierGalerkin(), tspan, D)

        empty!(series_t)
        empty!(series_mass)
        empty!(series_momentum)
        empty!(series_energy)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, callback,
                               relaxation, relaxation_tol = 1.0e-15)

        lines!(ax, series_t, change(series_energy); label = "energy")
        lines!(ax, series_t, change(series_momentum); label = "momentum")
        lines!(ax, series_t, change(series_mass); label = "mass")

        axislegend(ax; position = :rb, framevisible = false, nbanks = 2)
    end

    @info "NLS equation"
    for (ax, relaxation) in [(ax_nls1, MassEnergyRelaxation()),
                             (ax_nls2, ProjectionEnergyRelaxation())]
        initial_condition = two_solitons
        equation = CubicNLS(get_β(initial_condition))
        N = 2^8
        dt = 0.01
        tspan = (0.0, 100.0)

        (; xmin, xmax) = domain(initial_condition, equation)
        D = fourier_derivative_operator(xmin, xmax, N)

        (; q0, parameters) = setup(initial_condition, equation,
                                   FourierGalerkin(), tspan, D)

        empty!(series_t)
        empty!(series_mass)
        empty!(series_momentum)
        empty!(series_energy)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, callback,
                               relaxation, relaxation_tol = 1.0e-15)

        lines!(ax, series_t, change(series_energy); label = "energy")
        lines!(ax, series_t, change(series_momentum); label = "momentum")
        lines!(ax, series_t, change(series_mass); label = "mass")

        axislegend(ax; position = :rb, framevisible = false, nbanks = 2)
    end

    filename = joinpath(FIGDIR, "fully_discrete_conservation_two_waves.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function  fully_discrete_conservation_one_wave()
    fig = Figure(size = (1200, 350)) # default size is (600, 450)

    ax_bbm1 = Axis(fig[1, 1];
                   xlabel = L"Time $t$",
                   ylabel = "Change of invariants",
                   title = "BBM, mass-energy relax.")
    ax_kdv1 = Axis(fig[1, 2];
                   xlabel = L"Time $t$",
                   ylabel = "Change of invariants",
                   title = "KdV, mass-energy relax.")
    ax_nls1 = Axis(fig[1, 3];
                   xlabel = L"Time $t$",
                   ylabel = "Change of invariants",
                   title = "NLS, mass-energy relax.")

    linkyaxes!(ax_bbm1, ax_kdv1, ax_nls1)
    hideydecorations!(ax_kdv1; grid = false)
    hideydecorations!(ax_nls1; grid = false)

    # Setup callback computing the error of the invariants
    series_t = Vector{Float64}()
    series_mass = Vector{Float64}()
    series_momentum = Vector{Float64}()
    series_energy = Vector{Float64}()
    callback = let series_t = series_t, series_mass = series_mass, series_momentum = series_momentum, series_energy = series_energy
        function (q, parameters, t)
            push!(series_t, t)
            push!(series_mass, mass(q, parameters))
            push!(series_momentum, momentum(q, parameters))
            push!(series_energy, energy(q, parameters))
            return nothing
        end
    end

    alg = KenCarpARK437()

    @info "BBM equation"
    for (ax, relaxation) in [(ax_bbm1, MassEnergyRelaxation()),]
                            #  (ax_bbm2, ProjectionEnergyRelaxation())]
        initial_condition = solitary_wave
        equation = BBM()
        N = 2^8
        dt = 0.25
        tspan = (0.0, 1.0e4 * dt)

        (; xmin, xmax) = domain(initial_condition, equation)
        D = fourier_derivative_operator(xmin, xmax, N)

        (; q0, parameters) = setup(initial_condition, equation,
                                   FourierGalerkin(), tspan, D)

        empty!(series_t)
        empty!(series_mass)
        empty!(series_momentum)
        empty!(series_energy)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, callback,
                               relaxation, relaxation_tol = 1.0e-14)

        lines!(ax, series_t, change(series_energy); label = "energy")
        lines!(ax, series_t, change(series_momentum); label = "momentum")
        lines!(ax, series_t, change(series_mass); label = "mass")

        axislegend(ax; position = :rt, framevisible = false, nbanks = 2)
    end

    @info "KdV equation"
    for (ax, relaxation) in [(ax_kdv1, MassEnergyRelaxation()),]
        initial_condition = one_soliton
        equation = KdV()
        N = 2^8
        dt = 0.05
        tspan = (0.0, 1.0e4 * dt)

        (; xmin, xmax) = domain(initial_condition, equation)
        D = fourier_derivative_operator(xmin, xmax, N)

        (; q0, parameters) = setup(initial_condition, equation,
                                   FourierGalerkin(), tspan, D)

        empty!(series_t)
        empty!(series_mass)
        empty!(series_momentum)
        empty!(series_energy)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, callback,
                               relaxation, relaxation_tol = 1.0e-14)

        lines!(ax, series_t, change(series_energy); label = "energy")
        lines!(ax, series_t, change(series_momentum); label = "momentum")
        lines!(ax, series_t, change(series_mass); label = "mass")

        axislegend(ax; position = :rt, framevisible = false, nbanks = 2)
    end

    @info "NLS equation"
    for (ax, relaxation) in [(ax_nls1, MassEnergyRelaxation()),]
        initial_condition = one_soliton
        equation = CubicNLS(get_β(initial_condition))
        N = 2^8
        dt = 0.01
        tspan = (0.0, 1.0e4 * dt)

        (; xmin, xmax) = domain(initial_condition, equation)
        D = fourier_derivative_operator(xmin, xmax, N)

        (; q0, parameters) = setup(initial_condition, equation,
                                   FourierGalerkin(), tspan, D)

        empty!(series_t)
        empty!(series_mass)
        empty!(series_momentum)
        empty!(series_energy)
        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, callback,
                               relaxation, relaxation_tol = 1.0e-14)

        lines!(ax, series_t, change(series_energy); label = "energy")
        lines!(ax, series_t, change(series_momentum); label = "momentum")
        lines!(ax, series_t, change(series_mass); label = "mass")

        axislegend(ax; position = :rt, framevisible = false, nbanks = 2)
    end

    filename = joinpath(FIGDIR, "fully_discrete_conservation_one_wave.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function error_growth_multiple_solitons()
    fig = Figure(size = (1200, 800)) # default size is (600, 450)

    ax_kdv1 = Axis(fig[1, 1];
                   xlabel = L"Time $t$",
                   ylabel = L"Error at time $t$",
                   title = "KdV, two solitons",
                   xscale = log10, yscale = log10)
    ax_nls1 = Axis(fig[1, 2];
                   xlabel = L"Time $t$",
                   ylabel = L"Error at time $t$",
                   title = "NLS, two solitons",
                   xscale = log10, yscale = log10)
    ax_kdv2 = Axis(fig[2, 1];
                   xlabel = L"Time $t$",
                   ylabel = L"Error at time $t$",
                   title = "KdV, three solitons",
                   xscale = log10, yscale = log10)
    ax_nls2 = Axis(fig[2, 2];
                   xlabel = L"Time $t$",
                   ylabel = L"Error at time $t$",
                   title = "NLS, three solitons",
                   xscale = log10, yscale = log10)

    alg = KenCarpARK548()

    @info "KdV equation"
    for (ax, initial_condition) in [(ax_kdv1, two_solitons),
                                    (ax_kdv2, three_solitons)]
        equation = KdV()
        if initial_condition === two_solitons
            N = 2^10
            dt = 0.1
            tspan = (0.0, 1400.0)
        else
            N = 2^11
            dt = 0.1
            tspan = (0.0, 1800.0)
        end

        (; xmin, xmax) = domain(initial_condition, equation)
        D = fourier_derivative_operator(xmin, xmax, N)

        (; q0, parameters) = setup(initial_condition, equation,
                                   FourierGalerkin(), tspan, D)

        # Setup callback computing the error
        series_t = Vector{Float64}()
        series_error = Vector{Float64}()
        callback = let series_t = series_t, series_error = series_error, initial_condition = initial_condition
            function (q, parameters, t)
                p_equation = parameters.equation
                p_D_small = parameters.D_small
                p_tmp1_small = parameters.tmp1_small

                push!(series_t, t)

                x = grid(p_D_small)
                p_D_small.tmp .= q ./ size(p_D_small, 2)
                mul!(p_tmp1_small, p_D_small.brfft_plan, p_D_small.tmp)
                @. p_tmp1_small = (p_tmp1_small - initial_condition(t, x, p_equation))^2
                push!(series_error, sqrt(integrate(p_tmp1_small, p_D_small)))

                return nothing
            end
        end

        for (label, relaxation) in [("baseline",
                                     NoProjection()),
                                   (L"$\mathcal{M}, \mathcal{E}$ relaxation",
                                    MassEnergyRelaxation()),
                                   (L"$\mathcal{M}, \mathcal{P}, \mathcal{E}$ relaxation",
                                    ProjectionEnergyRelaxation())]
            empty!(series_t)
            empty!(series_error)
            @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                                   rhs_nonstiff!,
                                   q0, tspan, parameters, alg;
                                   dt, callback,
                                   relaxation, relaxation_tol = 1.0e-15)

            lines!(ax, series_t, series_error; label = label)
        end

        if initial_condition === two_solitons
            t = [1.0e2, tspan[end]]
            lines!(ax, t, t.^2 .* 1.5e-9; label = L"\mathcal{O}(t^2)", linestyle = :dot, color = :gray)

            t = [5.0e2, tspan[end]]
            lines!(ax, t, t.^2 .* 1.5e-11; linestyle = :dot, color = :gray)

            t = [5.0e2, tspan[end]]
            lines!(ax, t, t .* 2.0e-9; label = L"\mathcal{O}(t)", linestyle = :dashdot, color = :gray)
        else
            t = [1.0e2, tspan[end]]
            lines!(ax, t, t.^2 .* 1.0e-9; label = L"\mathcal{O}(t^2)", linestyle = :dot, color = :gray)

            t = [5.0e2, tspan[end]]
            lines!(ax, t, t.^2 .* 1.5e-11; linestyle = :dot, color = :gray)

            t = [5.0e2, tspan[end]]
            lines!(ax, t, t .* 2.0e-9; label = L"\mathcal{O}(t)", linestyle = :dashdot, color = :gray)
        end

        xlims!(ax, 1.0e1, tspan[end])

        axislegend(ax; position = :lt, framevisible = false, nbanks = 1)
    end

    @info "NLS equation"
    for (ax, initial_condition) in [(ax_nls1, two_solitons),
                                    (ax_nls2, three_solitons)]
        equation = CubicNLS(get_β(initial_condition))
        if initial_condition === two_solitons
            N = 2^10
            dt = 1.0e-2
            tspan = (0.0, 100.0)
        else
            N = 2^10
            dt = 1.0e-3
            tspan = (0.0, 100.0)
        end

        (; xmin, xmax) = domain(initial_condition, equation)
        D = fourier_derivative_operator(xmin, xmax, N)

        (; q0, parameters) = setup(initial_condition, equation,
                                   FourierGalerkin(), tspan, D)

        # Setup callback computing the error
        series_t = Vector{Float64}()
        series_error = Vector{Float64}()
        callback = let series_t = series_t, series_error = series_error, initial_condition = initial_condition
            function (q, parameters, t)
                p_equation = parameters.equation
                p_D_small = parameters.D_small
                p_tmp1_small = parameters.tmp1_small
                p_tmp2_small = parameters.tmp2_small

                push!(series_t, t)

                x = grid(p_D_small)
                v = real(q, parameters.equation)
                w = imag(q, parameters.equation)

                p_D_small.tmp .= v ./ size(p_D_small, 2)
                mul!(p_tmp1_small, p_D_small.brfft_plan, p_D_small.tmp)
                p_D_small.tmp .= w ./ size(p_D_small, 2)
                mul!(p_tmp2_small, p_D_small.brfft_plan, p_D_small.tmp)
                for i in eachindex(x, p_tmp1_small, p_tmp2_small)
                    ic = initial_condition(t, x[i], p_equation)
                    p_tmp1_small[i] = (p_tmp1_small[i] - real(ic))^2 + (p_tmp2_small[i] - imag(ic))^2
                end
                push!(series_error, sqrt(integrate(p_tmp1_small, p_D_small)))

                return nothing
            end
        end

        for (label, relaxation) in [("baseline",
                                     NoProjection()),
                                   (L"$\mathcal{M}, \mathcal{E}$ relaxation",
                                    MassEnergyRelaxation()),
                                   (L"$\mathcal{M}, \mathcal{P}, \mathcal{E}$ relaxation",
                                    ProjectionEnergyRelaxation())]
            empty!(series_t)
            empty!(series_error)
            @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                                   rhs_nonstiff!,
                                   q0, tspan, parameters, alg;
                                   dt, callback,
                                   relaxation, relaxation_tol = 1.0e-15)

            lines!(ax, series_t, series_error; label = label)
        end

        if initial_condition === two_solitons
            t = [1.0, tspan[end]]
            lines!(ax, t, t.^2 .* 1.5e-3; label = L"\mathcal{O}(t^2)", linestyle = :dot, color = :gray)

            t = [1.0, tspan[end]]
            lines!(ax, t, t .* 3.0e-5; label = L"\mathcal{O}(t)", linestyle = :dashdot, color = :gray)
        else
            t = [2.0, tspan[end]]
            lines!(ax, t, t.^2 .* 2.0e-4; label = L"\mathcal{O}(t^2)", linestyle = :dot, color = :gray)

            t = [1.0, tspan[end]]
            lines!(ax, t, t .* 1.0e-5; label = L"\mathcal{O}(t)", linestyle = :dashdot, color = :gray)
        end

        xlims!(ax, 1.0, tspan[end])
        ylims!(ax, 5.0e-5, 5.0)

        axislegend(ax; position = :lt, framevisible = false, nbanks = 1)
    end

    filename = joinpath(FIGDIR, "error_growth_multiple_solitons.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function error_growth_multiple_solitons_collocation()
    fig = Figure(size = (1200, 400)) # default size is (600, 450)

    ax_nls1 = Axis(fig[1, 1];
                   xlabel = L"Time $t$",
                   ylabel = L"Error at time $t$",
                   title = "NLS, two solitons",
                   xscale = log10, yscale = log10)
    ax_nls2 = Axis(fig[1, 2];
                   xlabel = L"Time $t$",
                   ylabel = L"Error at time $t$",
                   title = "NLS, three solitons",
                   xscale = log10, yscale = log10)

    alg = KenCarpARK548()

    @info "NLS equation"
    for (ax, initial_condition) in [(ax_nls1, two_solitons),
                                    (ax_nls2, three_solitons)]
        equation = CubicNLS(get_β(initial_condition))
        if initial_condition === two_solitons
            N = 2^10
            dt = 1.0e-2
            tspan = (0.0, 100.0)
        else
            N = 2^10
            dt = 2.0e-3
            tspan = (0.0, 100.0)
        end

        (; xmin, xmax) = domain(initial_condition, equation)
        D = fourier_derivative_operator(xmin, xmax, N)

        (; q0, parameters) = setup(initial_condition, equation,
                                   FourierCollocationConserveMassEnergy(),
                                   tspan, D)

        # Setup callback computing the error
        series_t = Vector{Float64}()
        series_error = Vector{Float64}()
        callback = let series_t = series_t, series_error = series_error, initial_condition = initial_condition
            function (q, parameters, t)
                p_equation = parameters.equation
                p_tmp1 = parameters.tmp1
                p_D2 = parameters.D2

                push!(series_t, t)

                x = grid(p_D2)
                v = real(q, p_equation)
                w = imag(q, p_equation)

                for i in eachindex(x, p_tmp1)
                    ic = initial_condition(t, x[i], p_equation)
                    p_tmp1[i] = (v[i] - real(ic))^2 + (w[i] - imag(ic))^2
                end
                push!(series_error, sqrt(integrate(p_tmp1, p_D2)))

                return nothing
            end
        end

        for (label, relaxation) in [("baseline",
                                     NoProjection()),
                                   (L"$\mathcal{M}, \mathcal{E}$ relaxation",
                                    MassEnergyRelaxation())]
            empty!(series_t)
            empty!(series_error)
            @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                                   rhs_nonstiff!,
                                   q0, tspan, parameters, alg;
                                   dt, callback,
                                   relaxation, relaxation_tol = 1.0e-15)

            lines!(ax, series_t, series_error; label = label)
        end

        if initial_condition === two_solitons
            t = [1.0, tspan[end]]
            lines!(ax, t, t.^2 .* 1.5e-3; label = L"\mathcal{O}(t^2)", linestyle = :dot, color = :gray)

            t = [1.0, tspan[end]]
            lines!(ax, t, t .* 3.0e-5; label = L"\mathcal{O}(t)", linestyle = :dashdot, color = :gray)
        else
            t = [2.0, tspan[end]]
            lines!(ax, t, t.^2 .* 7.0e-3; label = L"\mathcal{O}(t^2)", linestyle = :dot, color = :gray)

            t = [1.0, tspan[end]]
            lines!(ax, t, t .* 2.0e-4; label = L"\mathcal{O}(t)", linestyle = :dashdot, color = :gray)
        end

        xlims!(ax, 1.0, tspan[end])
        ylims!(ax, 5.0e-5, 5.0)

        axislegend(ax; position = :lt, framevisible = false, nbanks = 1)
    end

    filename = joinpath(FIGDIR, "error_growth_multiple_solitons_collocation.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function change_of_invariants_three_solitons_nls()
    fig = Figure(size = (1200, 650)) # default size is (600, 450)

    ax1_galerkin = Axis(fig[1, 1];
                        xlabel = L"Time $t$",
                        ylabel = "Change of invariants",
                        title = "Galerkin, baseline")
    ax1_collocation = Axis(fig[1, 2];
                          xlabel = L"Time $t$",
                          ylabel = "Change of invariants",
                          title = "Collocation, baseline")
    ax2_galerkin = Axis(fig[2, 1];
                        xlabel = L"Time $t$",
                        ylabel = "Change of invariants",
                        title = "Galerkin, mass-energy relax.")
    ax2_collocation = Axis(fig[2, 2];
                          xlabel = L"Time $t$",
                          ylabel = "Change of invariants",
                          title = "Collocation, mass-energy relax.")


    alg = KenCarpARK548()
    setups = [
        (ax1_galerkin, FourierGalerkin(), NoProjection()),
        (ax1_collocation, FourierCollocationConserveMassEnergy(), NoProjection()),
        (ax2_galerkin, FourierGalerkin(), MassEnergyRelaxation()),
        (ax2_collocation, FourierCollocationConserveMassEnergy(), MassEnergyRelaxation())]

    for (ax, semidiscretization, relaxation) in setups
        @info "NLS equation" semidiscretization relaxation
        initial_condition = three_solitons
        equation = CubicNLS(get_β(initial_condition))
        N = 2^10
        dt = 2.0e-3
        tspan = (0.0, 100.0)

        (; xmin, xmax) = domain(initial_condition, equation)
        D = fourier_derivative_operator(xmin, xmax, N)

        (; q0, parameters) = setup(initial_condition, equation,
                                   semidiscretization,
                                   tspan, D)

        # Setup callback computing the error of the invariants
        series_t = Vector{Float64}()
        series_mass = Vector{Float64}()
        series_momentum = Vector{Float64}()
        series_energy = Vector{Float64}()
        callback = let series_t = series_t, series_mass = series_mass, series_momentum = series_momentum, series_energy = series_energy
            function (q, parameters, t)
                push!(series_t, t)
                push!(series_mass, mass(q, parameters))
                push!(series_momentum, momentum(q, parameters))
                push!(series_energy, energy(q, parameters))
                return nothing
            end
        end

        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, callback,
                               relaxation, relaxation_tol = 1.0e-15)

        lines!(ax, series_t, change(series_energy); label = "energy")
        lines!(ax, series_t, change(series_momentum); label = "momentum")
        lines!(ax, series_t, change(series_mass); label = "mass")

        position = ax == ax2_galerkin ? :lt : :lb
        axislegend(ax; position, framevisible = false, nbanks = 3)
    end

    filename = joinpath(FIGDIR, "change_of_invariants_three_solitons_nls.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function error_growth_gray_solitons()
    fig = Figure(size = (1200, 650)) # default size is (600, 450)

    ax1_galerkin = Axis(fig[1, 1];
                        xlabel = L"Time $t$",
                        ylabel = L"Error at time $t$",
                        title = "Galerkin, one gray soliton",
                        xscale = log10, yscale = log10)
    ax1_collocation = Axis(fig[1, 2];
                          xlabel = L"Time $t$",
                          ylabel = L"Error at time $t$",
                          title = "Collocation, one gray soliton",
                          xscale = log10, yscale = log10)
    ax2_galerkin = Axis(fig[2, 1];
                        xlabel = L"Elapsed time $t$",
                        ylabel = L"Error at time $t$",
                        title = "Galerkin, two gray solitons",
                        xscale = log10, yscale = log10)
    ax2_collocation = Axis(fig[2, 2];
                          xlabel = L"Elapsed time $t$",
                          ylabel = L"Error at time $t$",
                          title = "Collocation, two gray solitons",
                          xscale = log10, yscale = log10)

    linkyaxes!(ax1_galerkin, ax1_collocation)
    hideydecorations!(ax1_collocation; grid = false)

    linkyaxes!(ax2_galerkin, ax2_collocation)
    hideydecorations!(ax2_collocation; grid = false)

    alg = KenCarpARK548()

    setups = [
        (ax1_galerkin, one_moving_gray_soliton, FourierGalerkin()),
        (ax1_collocation, one_moving_gray_soliton, FourierCollocationConserveMassEnergy()),
        (ax2_galerkin, two_moving_gray_solitons, FourierGalerkin()),
        (ax2_collocation, two_moving_gray_solitons, FourierCollocationConserveMassEnergy())]

    for (ax, initial_condition, semidiscretization) in setups
        @info "NLS equation" semidiscretization initial_condition
        equation = CubicNLS(get_β(initial_condition))
        if initial_condition === one_moving_gray_soliton
            N = 2^8
            dt = 0.04
            tspan = (0.0, 500.0)
        else
            N = 2^11
            dt = 0.04
            tspan = (-70.0, 70.0)
        end

        (; xmin, xmax) = domain(initial_condition, equation)
        D = fourier_derivative_operator(xmin, xmax, N)

        (; q0, parameters) = setup(initial_condition, equation,
                                   semidiscretization,
                                   tspan, D)

        # Setup callback computing the error
        series_t = Vector{Float64}()
        series_error = Vector{Float64}()
        if semidiscretization isa FourierGalerkin
            callback = let series_t = series_t, series_error = series_error, initial_condition = initial_condition
                function (q, parameters, t)
                    p_equation = parameters.equation
                    p_D_small = parameters.D_small
                    p_tmp1_small = parameters.tmp1_small
                    p_tmp2_small = parameters.tmp2_small

                    push!(series_t, t)

                    x = grid(p_D_small)
                    v = real(q, parameters.equation)
                    w = imag(q, parameters.equation)

                    p_D_small.tmp .= v ./ size(p_D_small, 2)
                    mul!(p_tmp1_small, p_D_small.brfft_plan, p_D_small.tmp)
                    p_D_small.tmp .= w ./ size(p_D_small, 2)
                    mul!(p_tmp2_small, p_D_small.brfft_plan, p_D_small.tmp)
                    for i in eachindex(x, p_tmp1_small, p_tmp2_small)
                        ic = initial_condition(t, x[i], p_equation)
                        p_tmp1_small[i] = (p_tmp1_small[i] - real(ic))^2 + (p_tmp2_small[i] - imag(ic))^2
                    end
                    push!(series_error, sqrt(integrate(p_tmp1_small, p_D_small)))

                    return nothing
                end
            end
        else
            callback = let series_t = series_t, series_error = series_error, initial_condition = initial_condition
                function (q, parameters, t)
                    p_equation = parameters.equation
                    p_tmp1 = parameters.tmp1
                    p_D2 = parameters.D2

                    push!(series_t, t)

                    x = grid(p_D2)
                    v = real(q, p_equation)
                    w = imag(q, p_equation)
                    for i in eachindex(x, p_tmp1)
                        ic = initial_condition(t, x[i], p_equation)
                        p_tmp1[i] = (v[i] - real(ic))^2 + (w[i] - imag(ic))^2
                    end
                    push!(series_error, sqrt(integrate(p_tmp1, p_D2)))

                    return nothing
                end
            end
        end

        if semidiscretization isa FourierGalerkin
            integrators = [
                ("baseline", NoProjection()),
                (L"$\mathcal{M}, \mathcal{E}$ relaxation",
                 MassEnergyRelaxation()),
                (L"$\mathcal{M}, \mathcal{P}, \mathcal{E}$ relaxation",
                 ProjectionEnergyRelaxation())]
        else
            integrators = [
                ("baseline",
                 NoProjection()),
                (L"$\mathcal{M}, \mathcal{E}$ relaxation",
                 MassEnergyRelaxation())]
        end

        for (label, relaxation) in integrators
            empty!(series_t)
            empty!(series_error)
            @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                                   rhs_nonstiff!,
                                   q0, tspan, parameters, alg;
                                   dt, callback,
                                   relaxation, relaxation_tol = 1.0e-15)

            lines!(ax, series_t .- tspan[begin], series_error; label = label)
        end

        if initial_condition === one_moving_gray_soliton
            t = [1.0, 500.0]
            lines!(ax, t, t .* 2.0e-8; label = L"\mathcal{O}(t)", linestyle = :dashdot, color = :gray)
            lines!(ax, t, t.^2 .* 5.0e-7; label = L"\mathcal{O}(t^2)", linestyle = :dot, color = :gray)
        else
            t = [2.0, 140.0]
            lines!(ax, t, t .* 2.0e-6; label = L"\mathcal{O}(t)", linestyle = :dashdot, color = :gray)
            lines!(ax, t, t.^2 .* 5.0e-5; label = L"\mathcal{O}(t^2)", linestyle = :dot, color = :gray)
        end

        axislegend(ax; position = :lt, framevisible = false, nbanks = 1)
    end

    filename = joinpath(FIGDIR, "error_growth_gray_solitons.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function change_of_invariants_gray_solitons()
    fig = Figure(size = (1200, 700)) # default size is (600, 450)

    ax1_galerkin = Axis(fig[1, 1];
                        xlabel = L"Time $t$",
                        ylabel = "Change of invar.",
                        title = "Galerkin, one gray soliton")
    ax1_collocation = Axis(fig[1, 2];
                          xlabel = L"Time $t$",
                          ylabel = "Change of invar.",
                          title = "Collocation, one gray soliton")
    ax2_galerkin = Axis(fig[2, 1];
                        xlabel = L"Elapsed time $t$",
                        ylabel = "Change of invar.",
                        title = "Galerkin, two gray solitons")
    ax2_collocation = Axis(fig[2, 2];
                          xlabel = L"Elapsed time $t$",
                          ylabel = "Change of invar.",
                          title = "Collocation, two gray solitons")

    linkyaxes!(ax1_galerkin, ax1_collocation)
    hideydecorations!(ax1_collocation; grid = false)

    linkyaxes!(ax2_galerkin, ax2_collocation)
    hideydecorations!(ax2_collocation; grid = false)

    alg = KenCarpARK548()

    setups = [
        (ax1_galerkin, one_moving_gray_soliton, FourierGalerkin()),
        (ax1_collocation, one_moving_gray_soliton, FourierCollocationConserveMassEnergy()),
        (ax2_galerkin, two_moving_gray_solitons, FourierGalerkin()),
        (ax2_collocation, two_moving_gray_solitons, FourierCollocationConserveMassEnergy())]

    for (ax, initial_condition, semidiscretization) in setups
        @info "NLS equation" semidiscretization initial_condition
        equation = CubicNLS(get_β(initial_condition))
        if initial_condition === one_moving_gray_soliton
            N = 2^8
            dt = 0.04
            tspan = (0.0, 500.0)
        else
            N = 2^11
            dt = 0.04
            tspan = (-70.0, 70.0)
        end

        (; xmin, xmax) = domain(initial_condition, equation)
        D = fourier_derivative_operator(xmin, xmax, N)

        (; q0, parameters) = setup(initial_condition, equation,
                                   semidiscretization,
                                   tspan, D)

        # Setup callback computing the error of the invariants
        series_t = Vector{Float64}()
        series_mass = Vector{Float64}()
        series_momentum = Vector{Float64}()
        series_energy = Vector{Float64}()
        callback = let series_t = series_t, series_mass = series_mass, series_momentum = series_momentum, series_energy = series_energy
            function (q, parameters, t)
                push!(series_t, t)
                push!(series_mass, mass(q, parameters))
                push!(series_momentum, momentum(q, parameters))
                push!(series_energy, energy(q, parameters))
                return nothing
            end
        end

        @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                               rhs_nonstiff!,
                               q0, tspan, parameters, alg;
                               dt, callback,
                               relaxation = MassEnergyRelaxation(),
                               relaxation_tol = 1.0e-15)

        lines!(ax, series_t .- tspan[begin], change(series_energy); label = "energy")
        lines!(ax, series_t .- tspan[begin], change(series_momentum); label = "momentum")
        lines!(ax, series_t .- tspan[begin], change(series_mass); label = "mass")

        if initial_condition === one_moving_gray_soliton
            position = :lt
        else
            position = :lb
        end
        axislegend(ax; position, framevisible = false, nbanks = 3)
    end

    filename = joinpath(FIGDIR, "change_of_invariants_gray_solitons.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function hyperbolized_nls()
    initial_condition = three_solitons
    τ = 1.0e-9
    N = 2^10
    semidiscretization = FourierGalerkin()
    dt = 1.0e-3
    tspan = (0.0, 100.0)
    alg = KenCarpARK548()
    relaxation = MassEnergyRelaxation()
    kwargs = (; relaxation_tol = 1.0e-15,)

    # Setup callback computing the error
    series_t = Vector{Float64}()
    series_mass = Vector{Float64}()
    series_momentum = Vector{Float64}()
    series_energy = Vector{Float64}()
    series_error = Vector{Float64}()
    callback = let series_t = series_t, series_mass = series_mass, series_momentum = series_momentum, series_energy = series_energy, series_error = series_error, initial_condition = initial_condition
        function (q, parameters, t)
            push!(series_t, t)
            push!(series_mass, mass(q, parameters))
            push!(series_momentum, momentum(q, parameters))
            push!(series_energy, energy(q, parameters))

            p_equation = parameters.equation
            p_equation_ic = CubicNLS(p_equation.β)
            p_D_small = parameters.D_small
            p_tmp1_small = parameters.tmp1_small
            p_tmp2_small = parameters.tmp2_small

            x = grid(p_D_small)
            v = real(q, p_equation)
            w = imag(q, p_equation)

            p_D_small.tmp .= v ./ size(p_D_small, 2)
            mul!(p_tmp1_small, p_D_small.brfft_plan, p_D_small.tmp)
            p_D_small.tmp .= w ./ size(p_D_small, 2)
            mul!(p_tmp2_small, p_D_small.brfft_plan, p_D_small.tmp)
            for i in eachindex(x, p_tmp1_small, p_tmp2_small)
                ic = initial_condition(t, x[i], p_equation_ic)
                p_tmp1_small[i] = (p_tmp1_small[i] - real(ic))^2 + (p_tmp2_small[i] - imag(ic))^2
            end
            push!(series_error, sqrt(integrate(p_tmp1_small, p_D_small)))

            return nothing
        end
    end

    # Initialization of physical and numerical parameters
    equation = HyperbolizedCubicNLS(get_β(initial_condition), τ)
    (; xmin, xmax) = domain(initial_condition, CubicNLS(get_β(initial_condition)))

    D = fourier_derivative_operator(xmin, xmax, N)

    (; q0, parameters) = setup(initial_condition, equation,
                               semidiscretization, tspan, D)
    @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                           rhs_nonstiff!,
                           q0, tspan, parameters, alg;
                           dt, callback, relaxation, kwargs...)

    fig = Figure(size = (1200, 300)) # default size is (600, 450)

    ax_invariants = Axis(fig[1, 1];
                         xlabel = L"Time $t$",
                         ylabel = "Change of invar.")
    lines!(ax_invariants, series_t, change(series_energy); label = "energy")
    lines!(ax_invariants, series_t, change(series_momentum); label = "momentum")
    lines!(ax_invariants, series_t, change(series_mass); label = "mass")
    axislegend(ax_invariants; position = :lt, framevisible = false)

    ax_error = Axis(fig[1, 2];
                    xlabel = L"Time $t$",
                    ylabel = L"Error at time $t$",
                    xscale = log10, yscale = log10)
    lines!(ax_error, series_t .- tspan[begin], series_error;
           label = L"$\mathcal{M}, \mathcal{E}$ relaxation")

    t = [2.0, tspan[end]]
    lines!(ax_error, t, t.^2 .* 2.0e-4; label = L"\mathcal{O}(t^2)", linestyle = :dot, color = :gray)
    t = [1.0, tspan[end]]
    lines!(ax_error, t, t .* 1.0e-5; label = L"\mathcal{O}(t)", linestyle = :dashdot, color = :gray)
    xlims!(ax_error, 1.0, tspan[end])
    ylims!(ax_error, 5.0e-5, 5.0)
    axislegend(ax_error; position = :lt, framevisible = false, nbanks = 1)

    filename = joinpath(FIGDIR, "hyperbolized_nls.pdf")
    save(filename, fig)
    @info "Results saved to $filename"
    return nothing
end

function comparison_bai_et_al(; semidiscretization = FourierGalerkin(),
                                tspan = (0.0, 1.0),
                                alg = KenCarpARK548(),
                                dt = 1 / 512,
                                relaxation = NoProjection(),
                                N = 2^10,
                                kwargs...)
    # Initialization of physical and numerical parameters
    xmin = -40.0
    xmax = +40.0
    β = 2
    equation = CubicNLS(β)
    function initial_condition(t, x, equation)
        # this is a single moving soliton
        return cis(-2x - 3t) * sech(x + 4t)
    end

    D = fourier_derivative_operator(xmin, xmax, N)

    (; q0, parameters) = setup(initial_condition, equation,
                               semidiscretization, tspan, D)
    # Run this once to trigger compilation
    solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
               rhs_nonstiff!,
               q0, tspan, parameters, alg;
               dt, relaxation, kwargs...)
    # Having compiled the code, run the timed simulation
    @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                           rhs_nonstiff!,
                           q0, tspan, parameters, alg;
                           dt, relaxation, kwargs...)

    if semidiscretization isa FourierGalerkin
        exact = initial_condition.(sol.t[end], grid(D), equation)
        exact_v = real.(exact)
        exact_w = imag.(exact)
        v = real(sol.u[end], equation)
        w = imag(sol.u[end], equation)

        p_D_small = parameters.D_small
        p_tmp1_small = parameters.tmp1_small
        p_tmp2_small = parameters.tmp2_small

        p_D_small.tmp .= v ./ size(p_D_small, 2)
        mul!(p_tmp1_small, p_D_small.brfft_plan, p_D_small.tmp)
        p_D_small.tmp .= w ./ size(p_D_small, 2)
        mul!(p_tmp2_small, p_D_small.brfft_plan, p_D_small.tmp)

        diff_v = p_tmp1_small - exact_v
        diff_w = p_tmp2_small - exact_w
        l2_error = sqrt(integrate(abs2, diff_v, D) +
                        integrate(abs2, diff_w, D))
        h1_error = sqrt(integrate(abs2, diff_v, D) -
                        integrate(diff_v .* (D * diff_v), D) +
                        integrate(abs2, diff_w, D) -
                        integrate(diff_w .* (D * diff_w), D))
        @info "Errors at the final time" l2_error h1_error
    elseif semidiscretization isa FourierCollocation
        exact = initial_condition.(sol.t[end], grid(D), equation)
        exact_v = real.(exact)
        exact_w = imag.(exact)
        v = real(sol.u[end], equation)
        w = imag(sol.u[end], equation)
        diff_v = v - exact_v
        diff_w = w - exact_w
        l2_error = sqrt(integrate(abs2, diff_v, D) +
                        integrate(abs2, diff_w, D))
        h1_error = sqrt(integrate(abs2, diff_v, D) -
                        integrate(diff_v .* (D * diff_v), D) +
                        integrate(abs2, diff_w, D) -
                        integrate(diff_w .* (D * diff_w), D))
        @info "Errors at the final time" l2_error h1_error
    end

    return nothing
end

function comparison_andrews_farrell(; N = 100,
                                      tspan = (0.0, 2.0e4),
                                      alg = KenCarpARK548(),
                                      dt = 1.0,
                                      relaxation = MassEnergyRelaxation(),
                                      kwargs...)
    # Setup callback computing the error
    series_t = Vector{Float64}()
    series_momentum = Vector{Float64}()
    series_energy = Vector{Float64}()
    callback = let series_t = series_t, series_momentum = series_momentum, series_energy = series_energy
        function (q, parameters, t)
            push!(series_t, t)
            push!(series_momentum, momentum(q, parameters))
            push!(series_energy, energy(q, parameters))

            return nothing
        end
    end

    # Initialization of physical and numerical parameters
    equation = BBM()
    xmin = -50.0
    xmax = +50.0
    function initial_condition(t, x, equation)
        c = (1 + sqrt(5)) / 2
        A = 3 * (c - 1)
        K = 0.5 * sqrt(1 - 1 / c)
        x_t = mod(x - c * t - xmin, xmax - xmin) + xmin

        # There are two normalizations of the BBM equation:
        # 1. u_t - u_{txx} + u_x + u u_x = 0
        # return A / cosh(K * x_t)^2
        # 2. u_t - u_{txx} + u u_x = 0
        return 1 + A / cosh(K * x_t)^2
    end

    D = fourier_derivative_operator(xmin, xmax, N)

    semidiscretization = FourierGalerkin()
    (; q0, parameters) = setup(initial_condition, equation,
                               semidiscretization, tspan, D)
    # Run this once to trigger compilation
    solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
               rhs_nonstiff!,
               q0, (0.0, 2 * dt), parameters, alg;
               dt, callback, relaxation, kwargs...)
    # Having compiled the code, run the timed simulation
    empty!(series_t)
    empty!(series_momentum)
    empty!(series_energy)
    @time sol = solve_imex(rhs_stiff!, operator(rhs_stiff!, parameters),
                           rhs_nonstiff!,
                           q0, tspan, parameters, alg;
                           dt, callback, relaxation, kwargs...)

    fig = Figure(size = (1200, 400)) # default size is (600, 450)

    ax_sol = Axis(fig[1, 1]; xlabel = L"x", ylabel = L"Solution $u$")
    lines!(ax_sol, grid(D), get_u(sol.u[begin], equation, parameters);
           label = "initial data")
    lines!(ax_sol, grid(D), get_u(sol.u[end], equation, parameters);
           label = "numerical solution")
    lines!(ax_sol, grid(D), initial_condition.(tspan[end], grid(D), equation);
           label = "exact solution")
    axislegend(ax_sol; position = :lt, framevisible = false)

    ax_invariants = Axis(fig[1, 2]; xlabel = L"Time $t$", ylabel = "Rel. change of invariants")
    lines!(ax_invariants, series_t, relative_change(series_energy); label = "energy")
    lines!(ax_invariants, series_t, relative_change(series_momentum); label = "momentum")
    axislegend(ax_invariants; position = :rt, framevisible = false)

    return fig
end
