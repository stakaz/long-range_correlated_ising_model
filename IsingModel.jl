using Random
using LinearAlgebra
using Colors
using Crayons
using Formatting
using StatsBase: counts, weights

abstract type LatticeBC end
abstract type LinearBC <: LatticeBC end

struct PeriodicBC <: LinearBC end
struct APeriodicBC <: LinearBC end
struct FreeBC <: LinearBC end

@inline apply_bc(::PeriodicBC, x) = x
@inline apply_bc(::APeriodicBC, x) = -x
@inline apply_bc(::FreeBC, x) = zero(x)

abstract type LatticeModel end

mutable struct IsingModel{T,N,F <: NTuple{N,LinearBC},TJ <: Number,Tβ <: Number} <: LatticeModel
	Λ::Array{T,N}
	bc::F
	empty_site::T
	Nempty::Int
	V::Int
	J::TJ
	β::Tβ
	cΛ::Array{Int,N}
	labels::Array{Int,1}
	max_label::Int
	FT1::Array{Complex{Float64},N}
	RNG::Random.MersenneTwister
end
IsingModel(Λ::Array{T,N}, bc::F, seed::Int = 0) where {T,N,F} = IsingModel(Λ, bc, zero(T), count(i -> i == zero(T), Λ), prod(size(Λ)), 1, 0.1, similar(Λ, Int), Array{Int}(undef, length(Λ)), 1, calc_FT_mode(Λ, unit(Λ, 1)), Random.MersenneTwister(seed))

"""
    display_lattice(Λ::AbstractArray [, prepare]; kwargs...)

Prints a 1, 2 or 3 dimentional array where each diffenret value gets a unique background color.

# Arguments
- `prepare::Function=x->"\$(x)"`: apply on each array element before printing, return something printable
- `title::String=""`: optional title above the array output
- `minwidth::Int = 2`: minimal width (number of characters) for each array element

"""
function display_lattice(Λ::AbstractArray, prepare::Function = x -> "$(x)"; title::String = "", minwidth::Int = 2)
	L = size(Λ)
	width = max(maximum(length.(prepare.(Λ))), minwidth - 1) + 1
	unique_species = sort(unique(prepare.(Λ)))
	colors_raw = distinguishable_colors(length(unique_species);lchoices = range(60; stop = 100, length = 15))

	color_dict = Dict(unique_species[i] => reinterpret(UInt32, convert(RGB24, colors_raw[i])) for i in 1:length(unique_species))
	fmt = "%$(width)s"
	println(typeof(Λ))
	title ≠ "" && println(title)
	if ndims(Λ) == 1
		for i in 1:L[1]
			print(Crayon(background = color_dict[prepare(Λ[i])]), sprintf1(fmt, prepare(Λ[i])))
			print(Crayon(reset = true), "\n")
		end
	elseif ndims(Λ) == 2
		for i in 1:L[1]
			for j in 1:L[2]
				print(Crayon(background = color_dict[prepare(Λ[i,j])]), sprintf1(fmt, prepare(Λ[i,j])))
			end
			print(Crayon(reset = true), "\n")
		end
	elseif ndims(Λ) == 3
		for k in 1:L[1]
			println("level $k")
			for i in 1:L[2]
				for j in 1:L[3]
					print(Crayon(background = color_dict[prepare(Λ[i,j,k])]), sprintf1(fmt, prepare(Λ[i,j,k])))
				end
				print(Crayon(reset = true), "\n")
			end
			print(Crayon(reset = true), "\n")
		end
	else
		println("display_lattice not defined for ndims = $(ndims(Λ))")
	end
	print("\n")
end

"""
unit(dims, dir::Integer)

Returns an Array as a unit vector in direction `dir` in `dims` dimentions.
`dims` can be an `Integer` or an `AbstractArray`
"""
unit(dims::Integer, dir::Integer) = [dir == i ? 1 : 0 for i ∈ 1:dims]
unit(Λ::AbstractArray, dir::Integer) = unit(ndims(Λ), dir)

"""
		neighbor_back(Λ::AbstractArray, i, dir, bc = x -> x)

Returns the next nearest neigbor of site `i` on a lattice `Λ` backward in direction `dir`.
When the element is at boundary, applies `bc` to it.
"""
@generated function neighbor_back(Λ::AbstractArray{T,N}, i, dir, bc = PeriodicBC()) where {T,N}
	quote
		$(Expr(:meta, :inline))
		@inbounds begin
			if i[dir] == 1
				return apply_bc(bc, Λ[Base.Cartesian.@ncall $N CartesianIndex d -> (d ≠ dir ? i[d] : size(Λ, d))])
			else
				return Λ[Base.Cartesian.@ncall $N CartesianIndex d -> (d ≠ dir ? i[d] : i[d] - 1)]
			end
		end
	end
end

"""
		neighbor_for(Λ::AbstractArray, i, dir, bc = x -> x)

Returns the next nearest neigbor of site `i` on a lattice `Λ` forward in direction `dir`.
When the element is at boundary, applies `bc` to it.
"""
@generated function neighbor_for(Λ::AbstractArray{T,N}, i, dir, bc = PeriodicBC()) where {T,N}
	quote
		$(Expr(:meta, :inline))
		@inbounds begin
			if i[dir] == size(Λ, dir)
				return apply_bc(bc, Λ[Base.Cartesian.@ncall $N CartesianIndex d -> (d ≠ dir ? i[d] : 1)])
			else
				return Λ[Base.Cartesian.@ncall $N CartesianIndex d -> (d ≠ dir ? i[d] : i[d] + 1)]
			end
		end
	end
end

"""
		neighbor_ind_back(Λ::AbstractArray, i, dir)

Returns the `CartesianIndex` of the next nearest neigbor of site `i` on a lattice `Λ` backward in direction `dir`.
"""
@generated function neighbor_ind_back(Λ::AbstractArray{T,N}, i, dir) where {T,N}
	quote
		$(Expr(:meta, :inline))
		@inbounds begin
			return Base.Cartesian.@ncall $N CartesianIndex d -> (d ≠ dir ? i[d] : (i[dir] == 1 ? size(Λ, d) : i[d] - 1))
		end
	end
end

"""
		neighbor_ind_for(Λ::AbstractArray, i, dir)

Returns the `CartesianIndex` of the next nearest neigbor of site `i` on a lattice `Λ` forward in direction `dir`.
"""
@generated function neighbor_ind_for(Λ::AbstractArray{T,N}, i, dir) where {T,N}
	quote
		$(Expr(:meta, :inline))
		@inbounds begin
			return Base.Cartesian.@ncall $N CartesianIndex d -> (d ≠ dir ? i[d] : (i[dir] == size(Λ, d) ? 1 : i[d] + 1))
		end
	end
end

"Precalcultion of the Fourier transformed exponentials with vector `k`"
calc_FT_mode(Λ::AbstractArray, k::Vector) = return reshape([exp(im * dot([Tuple(CartesianIndices(Λ)[i])...] .- 1, k .* (2π ./ size(Λ)))) for i ∈ eachindex(Λ)], size(Λ)...)

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
stohastic_cluster_max(M::IsingModel) = maximum(counts(M.cΛ, 1:M.max_label))

"Maximal size of the geometric clusters of the Ising Model `M`"
function geometric_cluster_max(M::IsingModel)
	M.max_label = hk_cluster!(M.Λ, M.cΛ, M.labels, M.bc, (x, y, i, d) -> (x == y), M.empty_site)
	M.max_label = hk_relabel!(M.cΛ, M.labels, M.max_label)
	return stohastic_cluster_max(M::IsingModel)
end
### call only AFTER!!! geometric_cluster_max, because the latter generates the clusters
"Mean size of the geometric clusters of the Ising Model `M`"
geometric_cluster_mean(M::IsingModel) = stohastic_cluster_mean(M::IsingModel)

"Fourier mode in (1,0,...) direction for the Ising Model `M`"
correlation_FT1_mode(M::IsingModel) = abs(sum(M.Λ .* M.FT1))^2

"""
hk_find!(x::Int, labels::Array{Int,1})

Finds the proper label for the culster number `x` in `labels`
"""
@inline function hk_find!(x::Int, labels::Array{Int,1})
	@inbounds while labels[x] ≠ x
		p = labels[x]
		labels[x] = labels[p]
		x = p
	end
  return x
end
	
    """
hk_union!(x, y, labels)

Unite cluster labels `x` and `y` in `labels`
"""
hk_union!(x, y, labels) = @inbounds labels[hk_find!(x, labels)] = hk_find!(y, labels)

"""
hk_cluster!(Λ::Array{T, N}, cΛ::Array{Int, N}, labels::Array{Int,1}, bc::NTuple{N, Function}, bond_condition::Function , empty_site::T = zero(T))

Clusterize the lattice `Λ` and writes the cluster labels to `cΛ`, the proper labels to `labels` and the maximum label counter to `max_label`.

`bond_condition` is a function which takes `(x,y,i,d)` as arguments and returns a bool.
`x` and `y` are the values on the neighboring sites,`i` is the index of `x` and `d` is the direction of the bond.
`bc` is a tuple of boundary conditions for each direction.
`empty_site` describes which sites to skip (empty).
"""
@generated function hk_cluster!(Λ::Array{T,N}, cΛ::Array{Int,N}, labels::Array{Int,1}, bc::NTuple{N,LinearBC}, bond_condition::Function, empty_site::T = zero(T)) where {T,N}
	return quote
		@inbounds begin
			for i ∈ 1:length(cΛ)
				cΛ[i] = 0
				labels[i] = i
			end
			max_label = 0
			for i ∈ CartesianIndices(size(cΛ))
				Λ[i] ≠ empty_site || continue
				Base.Cartesian.@nexprs $N d -> (begin
					nn = neighbor_ind_for(cΛ, i, d)
					if cΛ[i] == 0
						if bond_condition(Λ[i], (nn[d] == 1 ? apply_bc(bc[d], Λ[nn]) : Λ[nn]), i, d)
							if cΛ[nn] == 0
								cΛ[i] = cΛ[nn] = max_label += 1
							else
								cΛ[i] = cΛ[nn]
							end
						else
							cΛ[i] = max_label += 1
						end
					else
						if bond_condition(Λ[i], (nn[d] == 1 ? apply_bc(bc[d], Λ[nn]) : Λ[nn]), i, d)
							if cΛ[nn] == 0
								cΛ[nn] = cΛ[i]
							else
								hk_union!(cΛ[i], cΛ[nn], labels)
							end
						end
					end
				end)
			end
			return max_label
		end
	end
end

"""
hk_relabel!(cΛ::Array{T, N}, old_labels::Array{Int}, max_label::Int)

Relabels the cluster lattice `cΛ` with subsequent labels and returns the number of clusters.
"""
function hk_relabel!(cΛ::Array{T,N}, old_labels::Array{Int}, max_label::Int) where {T,N}
	label = 0
	labels = fill(0, max_label)
	
	@inbounds for i ∈ 1:length(cΛ)
		cΛ[i] ≠ 0 || continue
		current = hk_find!(cΛ[i], old_labels)
		if labels[current] == 0
			labels[current] = label += 1
		end
		cΛ[i] = labels[current]
	end
	return label
end

"""
sw_update!(M::IsingModel{T}, empty_site::T = zero(T))

Swendsen-Wang update of the model `M`
"""
function sw_update!(M::IsingModel{T}, empty_site::T = zero(T); p_bond = 0.0) where {T}
	M.max_label = hk_cluster!(M.Λ, M.cΛ, M.labels, M.bc, (x, y, i, d) -> (x == y && rand(M.RNG) < p_bond), empty_site)
	M.max_label = hk_relabel!(M.cΛ, M.labels, M.max_label)
	new_values = rand(M.RNG, Array{T}([-1,1]), M.max_label)
	
	@inbounds for i ∈ eachindex(M.Λ)
		M.Λ[i] == empty_site || (M.Λ[i] = new_values[M.cΛ[i]])
	end
	return nothing
end


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
function init_observables(model::LatticeModel, aliases::Array{Symbol}, Nmeas::Int)
	return NamedTuple{Tuple(aliases)}(Array{typeof(alias_to_observable[i](model))}(undef, Nmeas) for i ∈ aliases)
end

"Calculate observables given as Array of `aliases` and store in `D` at row `m`"
function calc_observables!(M::LatticeModel, D::NamedTuple, m::Int, aliases::Array{Symbol})
	for a ∈ aliases
		D[a][m] = alias_to_observable[a](M)
	end
end

"""
	run_simulation_with_β_series(ising::IsingModel; kwargs...)

First, try to find the simulation temperature as the peak of (currently) dlnm observable.
Uses Optim for that and βmin, βmax as boundaries.
While simulations for the βsim search uses `100 * L * (d-1)` number of measurements.
Then performs a simulation at βsim, writes the result and returns the observables.

The saved result consists of:
- `data`: the measured observables as `Array{NamedTuple}`
- `ising`: the final state of the model
- `config`: the parameters passed to `run_simulation` function
- `total_time`: elapsed time

### Arguments
- `β_array = 0.2:0.1:0.5`: the β range for the simulation
-	`Ntherm::Int = 0`: number of thermalization sweeps before each β
-	`Nmeas::Int = 1`: number of measurements at each β
-	`Nbetween::Int = 1`: number of sweeps between two subsequent measurements
-	`obs_aliases::Array{Symbol} = [:E, :M, :Csmax, :Csmean, :Cgmax, :Cgmean, :FT1]`: observables to measure. Possible aliases are: $(keys(alias_to_observable)). `Cgmax` **must** come before `Cgmean` and is needed for the latter.
- `out_dir = "./observables/"`: directory for the output
- `name_prefix = "ising_"`: prefix for the generated name
- `name_suffix = "_raw_observables"`: suffix for the generated name
- `save_data = true`: whether to save the data
- `cfg_params`: parameters of the cfg which was used
"""
function run_simulation(
	ising::IsingModel;
	β_array = 0.2:0.1:0.5,
	Ntherm::Int = 0,
	Npretherm::Int = 0,
	Nmeas::Int = 0,
	Nbetween::Int = 1,
	obs_aliases::Array{Symbol} = [:E, :M, :Csmax, :Csmean, :Cgmax, :Cgmean, :FT1],
	out_dir::String = "./observables/",
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
	save_data && mkpath(out_dir)
	result_file = "$(out_dir)/$(name_prefix)d$(dim)_L$(L)$(name_suffix).jld2"
	error_file = "$(out_dir)/ERRORS/$(name_prefix)d$(dim)_L$(L)$(name_suffix).dat"

	time1 = time_ns()

	config = (β_array = β_array, Ntherm = Ntherm, Nmeas = Nmeas, Nbetween = Nbetween, Npretherm = Npretherm, obs_aliases = obs_aliases, verbose = verbose)

		### initialization
	ising.β = β_array[1]
	p_bond = 1 - exp(-2 * β_array[1] * ising.J)
	data = Array{NamedTuple}(undef, 0)

	### pre-thermalization
	verbose ≥ 1 && println("start simulations")
	verbose ≥ 2 && println("  $Npretherm pre-thermalization sweeps")
	for sweep ∈ 1:Npretherm; sw_update!(ising; p_bond = p_bond) end
	
	for β ∈ β_array
		verbose ≥ 1 && println("  simulation at β = $β")
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
	return data
end

IM = IsingModel(rand(Int8[-1,1], 8, 8), (PeriodicBC(), PeriodicBC()), 1)

D = run_simulation(IM; β_array = [0.44], Ntherm = 1000, Nmeas = 1000, save_data = false, verbose = 3)
