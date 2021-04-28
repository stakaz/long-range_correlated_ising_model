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


    IM = IsingModel(rand(Int8[-1,1], 8, 8), (PeriodicBC(), PeriodicBC()), 1)

IM.Λ