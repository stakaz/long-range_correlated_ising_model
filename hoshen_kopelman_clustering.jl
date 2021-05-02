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
		hk_cluster!(Λ::Array{T, N}, Γ::Array{Int, N}, labels::Array{Int,1}, bc::NTuple{N, Function}, bond_condition::Function , empty_site::T = zero(T))

Clusterize the lattice `Λ` and writes the cluster labels to `Γ`, the proper labels to `labels` and the maximum label counter to `max_label`.

`bond_condition` is a function which takes `(x,y,i,d)` as arguments and returns a bool.
`x` and `y` are the values on the neighboring sites,`i` is the index of `x` and `d` is the direction of the bond.
`bc` is a tuple of boundary conditions for each direction.
`empty_site` describes which sites to skip (empty).
"""
@generated function hk_cluster!(Λ::Array{T,N}, Γ::Array{Int,N}, labels::Array{Int,1}, bc::NTuple{N,LinearBC}, bond_condition::Function, empty_site::T = zero(T)) where {T,N}
	return quote
		@inbounds begin
			for i ∈ 1:length(Γ)
				Γ[i] = 0
				labels[i] = i
			end
			max_label = 0
			for i ∈ CartesianIndices(size(Γ))
				Λ[i] ≠ empty_site || continue
				Base.Cartesian.@nexprs $N d -> (begin
					nn = neighbor_ind_for(Γ, i, d)
					if Γ[i] == 0
						if bond_condition(Λ[i], (nn[d] == 1 ? apply_bc(bc[d], Λ[nn]) : Λ[nn]), i, d)
							if Γ[nn] == 0
								Γ[i] = Γ[nn] = max_label += 1
							else
								Γ[i] = Γ[nn]
							end
						else
							Γ[i] = max_label += 1
						end
					else
						if bond_condition(Λ[i], (nn[d] == 1 ? apply_bc(bc[d], Λ[nn]) : Λ[nn]), i, d)
							if Γ[nn] == 0
								Γ[nn] = Γ[i]
							else
								hk_union!(Γ[i], Γ[nn], labels)
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
		hk_relabel!(Γ::Array{T, N}, old_labels::Array{Int}, max_label::Int)
		
Relabels the cluster lattice `Γ` with subsequent labels and returns the number of clusters.
"""
function hk_relabel!(Γ::Array{T,N}, old_labels::Array{Int}, max_label::Int) where {T,N}
	label = 0
	labels = fill(0, max_label)
	
	@inbounds for i ∈ 1:length(Γ)
		Γ[i] ≠ 0 || continue
		current = hk_find!(Γ[i], old_labels)
		if labels[current] == 0
			labels[current] = label += 1
		end
		Γ[i] = labels[current]
	end
	return label
end