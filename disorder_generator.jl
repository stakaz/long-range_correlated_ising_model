using Statistics
using StatsBase
using LinearAlgebra
using FastTransforms
using Distributions
using Random 

periodic_dist(x, L) = x ≤ L / 2 ? x : L - x

"""
    generate_corr_matrix(dims::Tuple, C::Function; T::Type = Float64)

Calculate the function `C` for each distance `r` and returns an array where each C(r) is stored.
The distnace `r` is measured from upper left corner and implies periodic boundary conditions.

# Arguments:
- `dims::Tuple`: dimensions of the desired lattice
- `C::Function`: desired correlation function `C(r)`
- `T::Type = Float64`: type of the retured array
"""
function generate_corr_matrix(dims::Tuple, C::Function; T::Type = Float64)
	Λ = Array{T}(undef, dims)
	fill_periodic_distances!(Λ, C)
	return Λ
end

"""
    fill_periodic_distances!(Λ::AbstractArray, C::Function)

Helper function for the `generate_corr_matrix` function.
The type of `Λ` is known on call and therefore can this inner function works faster.

# Arguments:
- `Λ::AbstractArray`: array to save the `C(r)` to
- `C::Function`: desired correlation function `C(r)`
"""
function fill_periodic_distances!(Λ::AbstractArray, C::Function)
	for i ∈ CartesianIndices(Λ)
		r = norm(periodic_dist.(Tuple(i) .- 1, size(Λ)))
		Λ[i] = C(r)
	end
end

"""
    generate_continuous_disorder(dims::Tuple, S::AbstractArray)

Produces an array with dimensions `dims` with correlated disorder where the correlation follows the function `C(r)`

# Arguments:
- `dims::Tuple`: dimensions of the desired lattice
- `C::Function`: desired correlation function `C(r)`
- `RNG = Random.MersenneTwister(1)`: a random number generator for the continuous disorder 
"""
function generate_continuous_disorder(dims::Tuple, S::AbstractArray, RNG)
	φq = sqrt.(max.(S, 0)) .* rand.(RNG, Normal(0, √(2 * prod(dims)))) # 2V = 2prod(dims) is used because the random numbers are produced only in real space 
	φ = real.(ifft(φq))
	return φ
end

"""
    generate_spectral_density(dims::Tuple, C::Function)

Produces a spectral density array with dimensions `dims` of the array with `C(r)` entries

# Arguments:
- `dims::Tuple`: dimensions of the desired lattice
- `C::Function`: desired correlation function `C(r)`
"""
generate_spectral_density(dims::Tuple, C::Function) = real.(fft(generate_corr_matrix(dims, C)))

"""
    transform_to_discrete_disorder(Λ::AbstractArray, p::Number; T = Int8(1), F = Int8(0))

Produces a truncated array similar to `Λ` but only with `T` and `F` values.
The concentration of `T` is given through `p`.

# Arguments:
- `Λ::AbstractArray`: expect an array with continuous correlated numbers
- `p::Number`: concentration of `T` values
- `T = Int8(1)`: values with concentration `p` to fill into the returned array 
- `F = Int8(0)`: values with concentration `1 - p` to fill into the returned array 
- `RNG = Random.MersenneTwister(1)`: a random number generator for the continuous disorder 
"""
transform_to_discrete_disorder(Λ::AbstractArray, p::Number; T = Int8(1), F = Int8(0)) = [i ≤ quantile(Normal(0, 1), p) ? T : F for i ∈ Λ]

function generate_discrete_correlated_disorder(dims::Tuple; p::Number = 0.5, a = 1.0, C = (r, a) -> (1 + r^2)^(-a / 2), T = Int8(1), F = Int8(0), RNG = Random.MersenneTwister(1))
	S = generate_spectral_density(dims, r -> C(r, a))
	Λc = generate_continuous_disorder(dims, S, RNG)
	Λd = transform_to_discrete_disorder(Λc, p; T, F)
	return Λd
end