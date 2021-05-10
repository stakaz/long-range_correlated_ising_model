using Random
using JLD2
using FileIO
using LinearAlgebra
using StatsBase: counts, weights

include("lattice_utils.jl")
include("hoshen_kopelman_clustering.jl")
include("disorder_generator.jl")

mutable struct IsingModel{T,N,F <: NTuple{N,LinearBC},TJ <: Number,Tβ <: Number} 
	Λ::Array{T,N}
	bc::F
	empty_site::T
	Nempty::Int
	V::Int
	J::TJ
	β::Tβ
	Γ::Array{Int,N}
	labels::Array{Int,1}
	max_label::Int
	FT1::Array{Complex{Float64},N}
	RNG::Random.MersenneTwister
end
IsingModel(Λ::Array{T,N}, bc::F, seed::Int = 0) where {T,N,F} = IsingModel(Λ, bc, zero(T), count(i -> i == zero(T), Λ), prod(size(Λ)), 1, 0.1, similar(Λ, Int), Array{Int}(undef, length(Λ)), 1, calc_FT_mode(Λ, unit(Λ, 1)), Random.MersenneTwister(seed))

"""
		sw_update!(M::IsingModel{T}, empty_site::T = zero(T))

Swendsen-Wang update of the model `M`
"""
function sw_update!(M::IsingModel{T}, empty_site::T = zero(T); p_bond = 0.0) where {T}
	M.max_label = hk_cluster!(M.Λ, M.Γ, M.labels, M.bc, (x, y, i, d) -> (x == y && rand(M.RNG) < p_bond), empty_site)
	M.max_label = hk_relabel!(M.Γ, M.labels, M.max_label)
	new_values = rand(M.RNG, Array{T}([-1,1]), M.max_label)
	
	@inbounds for i ∈ eachindex(M.Λ)
		M.Λ[i] == empty_site || (M.Λ[i] = new_values[M.Γ[i]])
	end
	return nothing
end

"Energy of the Ising Model `M`"
function energy(M::IsingModel)
	s::Int = 0
	@inbounds @simd for i ∈ CartesianIndices(size(M.Λ))
		@inbounds for d ∈ 1:ndims(M.Λ)
			s += M.Λ[i] * neighbor_for(M.Λ, i, d, M.bc[d])
		end
	end
	return -M.J * s
end

"Magnetization of the Ising Model `M`"
function magnetization(M::IsingModel)
	s::Int = 0
	@inbounds @simd for i ∈ eachindex(M.Λ)
		s += M.Λ[i]
	end
	return s
end

"Mean size of the stohastic clusters of the Ising Model `M`"
stohastic_cluster_mean(M::IsingModel) = (M.V - M.Nempty) / M.max_label
"Maximal size of the stohastic clusters of the Ising Model `M`"
stohastic_cluster_max(M::IsingModel) = maximum(counts(M.Γ, 1:M.max_label))

"Maximal size of the geometric clusters of the Ising Model `M`"
function geometric_cluster_max(M::IsingModel)
	M.max_label = hk_cluster!(M.Λ, M.Γ, M.labels, M.bc, (x, y, i, d) -> (x == y), M.empty_site)
	M.max_label = hk_relabel!(M.Γ, M.labels, M.max_label)
	return stohastic_cluster_max(M::IsingModel)
end
### call only AFTER!!! geometric_cluster_max, because the latter generates the clusters
"Mean size of the geometric clusters of the Ising Model `M`"
geometric_cluster_mean(M::IsingModel) = stohastic_cluster_mean(M::IsingModel)

"Precalcultion of the Fourier transformed exponentials with vector `k`"
calc_FT_mode(Λ::AbstractArray, k::Vector) = return reshape([exp(im * dot([Tuple(CartesianIndices(Λ)[i])...] .- 1, k .* (2π ./ size(Λ)))) for i ∈ eachindex(Λ)], size(Λ)...)

"Fourier mode in (1,0,...) direction for the Ising Model `M`"
correlation_FT1_mode(M::IsingModel) = abs(sum(M.Λ .* M.FT1))^2

alias_to_observable = Dict{Symbol,Function}(
:E => energy,
:M => magnetization,
:Csmean => stohastic_cluster_mean,
:Csmax => stohastic_cluster_max,
:Cgmean => geometric_cluster_mean,
:Cgmax => geometric_cluster_max,
:FT1 => correlation_FT1_mode,
)
observable_to_alias = Dict{Function,Symbol}(reverse(p) for p ∈ pairs(alias_to_observable))

"Initialize the `NamedTuple` for the observables given in `aliases` with `Nmeas` number of measurements"
function init_observables(model::IsingModel, aliases::Array{Symbol}, Nmeas::Int)
	return NamedTuple{Tuple(aliases)}(Array{typeof(alias_to_observable[i](model))}(undef, Nmeas) for i ∈ aliases)
end

"Calculate observables given as Array of `aliases` and store in `D` at row `m`"
function calc_observables!(M::IsingModel, D::NamedTuple, m::Int, aliases::Array{Symbol})
	for a ∈ aliases
		D[a][m] = alias_to_observable[a](M)
	end
end

"""
    generate_defected_ising_lattice(L, dim, seed, pd, a;T::Type=Int)
Generate a (correlated) defected lattice of size `L^dim` with correlation exponent `a`, defect concentration `pd`, gaussian width `sigma2`. Potentially convert elements to type `T`.

"""
function generate_defected_ising_lattice(L, dim, seed, pd, a; T::Type = Int,)
	RNG = Random.MersenneTwister()
	if pd == 0.0
		println("return pure ising lattice")
		return reshape(convert.(T, rand(RNG, [-1,1], L^dim)), ntuple(d -> L, dim))
	end

	if a == Inf
		println("return uncorrelated ising lattice")
		Λ = reshape(convert.(T, rand(RNG, [-1,1], L^dim)), ntuple(d -> L, dim))
		for i ∈ eachindex(Λ)
			rand(RNG) < pd && (Λ[i] = zero(T))
		end
		return Λ
	end

	println("return correlated ising lattice")

	Λ = generate_discrete_correlated_disorder(ntuple(d -> L, dim); p = 1 - pd, a, RNG)
	for i ∈ eachindex(Λ)
		if isapprox(Λ[i], 0.0) 
			Λ[i] = 0.0
		else
		rand(RNG) > 0.5 ? Λ[i] = 1.0 : Λ[i] = -1.0
		end
	end
	return Λ
end

"""
	run_simulation(ising::IsingModel; kwargs...)

Performs a series of simulations at each temperature from `β_array`.
Start with a prethermalization and thermalized betwwen each temperature.

The saved result consists of:
- `data`: the measured observables as `Array{NamedTuple}`
- `ising`: the final state of the model
- `config`: the parameters passed to `run_simulation` function
- `total_time`: elapsed time
A NamedTuple `(;data, config, total_time, cfg_params)` is returned.

### Arguments
- `β_array = 0.2:0.1:0.5`: the β range for the simulation
-	`Ntherm::Int = 0`: number of thermalization sweeps before each β
-	`Nmeas::Int = 1`: number of measurements at each β
-	`Nbetween::Int = 1`: number of sweeps between two subsequent measurements (1 means measure after each sweep)
-	`obs_aliases::Array{Symbol} = [:E, :M, :Csmax, :Csmean, :Cgmax, :Cgmean, :FT1]`: observables to measure. Possible aliases are: $(keys(alias_to_observable)). `Cgmax` **must** come before `Cgmean` and is needed for the latter.
- `output_dir = "./observables/"`: directory for the output
- `name_prefix = "ising_"`: prefix for the generated name
- `name_suffix = "_raw_observables"`: suffix for the generated name
- `save_data = true`: whether to save the data
- `cfg_params = (;)`: parameters of the cfg which was used (provided by the user)
- `verbose = 0`: defined the level of verbosity (0-2)
"""
function run_simulation(
	ising::IsingModel;
	β_array = 0.2:0.1:0.5,
	Ntherm::Int = 0,
	Npretherm::Int = 0,
	Nmeas::Int = 0,
	Nbetween::Int = 1,
	obs_aliases::Array{Symbol} = [:E, :M, :Csmax, :Csmean, :Cgmax, :Cgmean, :FT1],
	output_dir::String = "./observables/",
	name_prefix::String = "ising_",
	name_suffix::String = "_raw_observables",
	save_data::Bool = true,
	cfg_params::NamedTuple = NamedTuple(),
	verbose::Int = 0,
	compress = true,
	)
	### preparation
	L = size(ising.Λ, 1)
	dim = length(size(ising.Λ))

	## generate names and directories
	save_data && mkpath(output_dir)
	result_file = "$(output_dir)/$(name_prefix)d$(dim)_L$(L)$(name_suffix).jld2"

	time1 = time_ns()

	config = (β_array = β_array, Ntherm = Ntherm, Nmeas = Nmeas, Nbetween = Nbetween, Npretherm = Npretherm, obs_aliases = obs_aliases, verbose = verbose)

	### initialization
	ising.β = β_array[1]
	p_bond = 1 - exp(-2 * β_array[1] * ising.J)
	data = Array{NamedTuple}(undef, 0)

	### pre-thermalization
	verbose ≥ 1 && println("start simulations")
	verbose ≥ 2 && println("  start $Npretherm pre-thermalization sweeps")
	for sweep ∈ 1:Npretherm; sw_update!(ising; p_bond = p_bond) end
	
	for β ∈ β_array
		verbose ≥ 1 && println("  start simulation at β = $β")
		ising.β = β
		p_bond = 1 - exp(-2 * β * ising.J)

		data_at_β = init_observables(ising, obs_aliases, Nmeas)
		### thermalization
		verbose ≥ 2 && println("    start $Ntherm thermalization sweeps")
		for sweep ∈ 1:Ntherm
			sw_update!(ising; p_bond = p_bond)
		end

		### measurements
		verbose ≥ 2 && println("    start $Nmeas measurements")
		for meas ∈ 1:Nmeas
			for sweep ∈ 1:Nbetween
				sw_update!(ising; p_bond = p_bond)
			end
			calc_observables!(ising, data_at_β, meas, obs_aliases)
		end
		push!(data, merge((β = β, L = size(ising.Λ)), data_at_β))
	end
	verbose ≥ 1 && println("end simulations")

	time2 = time_ns()

	total_time = (meas_time = (time2 - time1) / 1.0e9)

	if save_data
		verbose ≥ 1 && println("saving data to $(result_file)")
		jldopen(result_file, true, true, true, IOStream; compress = compress) do file
			file["data"] = data
			file["config"] = config
			file["total_time"] = total_time
			file["cfg_params"] = cfg_params
		end
		verbose ≥ 1 && println("saving data finished")
	end
	return (;data, config, total_time, cfg_params)
end
