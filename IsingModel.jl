using Random
using LinearAlgebra

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
unit(dims, dir::Integer)

Returns an Array as a unit vector in direction `dir` in `dims` dimentions.
`dims` can be an `Integer` or an `AbstractArray`
"""
unit(dims::Integer, dir::Integer) = [dir == i ? 1 : 0 for i ∈ 1:dims]
unit(Λ::AbstractArray, dir::Integer) = unit(ndims(Λ), dir)

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


IM = IsingModel(rand(Int8[-1,1], 8, 8), (PeriodicBC(), PeriodicBC()), 1)

IM.Λ