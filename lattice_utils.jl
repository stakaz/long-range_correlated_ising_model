using Colors
using Crayons
using Formatting

export display_lattice
export neighbor_back
export neighbor_for
export neighbor_ind_back
export neighbor_ind_for
export unit
export apply_bc

export LatticeBC
export LinearBC
export PeriodicBC
export APeriodicBC
export FreeBC

abstract type LatticeBC end
abstract type LinearBC <: LatticeBC end

struct PeriodicBC <: LinearBC end
struct APeriodicBC <: LinearBC end
struct FreeBC <: LinearBC end

@inline apply_bc(::PeriodicBC, x) = x
@inline apply_bc(::APeriodicBC, x) = -x
@inline apply_bc(::FreeBC, x) = zero(x)

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
