using Random
 
abstract type LatticeModel end

mutable struct IsingModel{T, N, F<:NTuple{N, LinearBC}, TJ<:Number, Tβ<:Number} <: LatticeModel
	Λ::Array{T, N}
	bc::F
	empty_site::T
	Nempty::Int
	V::Int
	J::TJ
	β::Tβ
	cΛ::Array{Int, N}
	labels::Array{Int, 1}
	max_label::Int
	FT1::Array{Complex{Float64}, N}
	RNG::Random.MersenneTwister
end
IsingModel(Λ::Array{T, N}, bc::F, seed::Int = 0) where {T, N, F} = IsingModel(Λ, bc, zero(T), count(i->i==zero(T), Λ), prod(size(Λ)), 1, 0.1, similar(Λ, Int), Array{Int}(undef, length(Λ)), 1, calc_FT_mode(Λ, unit(Λ, 1)), Random.MersenneTwister(seed))