include("ising_model.jl")


using QuadGK

### set parameters
dim = 2
a = 2.0
pd = 0.0
L = 32 
seed = 1

V = L^dim

### temperautre range
β_array = 0.4:0.001:0.5

### init Ising model and run simulaiton
IM = IsingModel(generate_defected_ising_lattice(L, dim, seed, pd, a), ntuple(d -> PeriodicBC(), dim), seed)

D = run_simulation(IM; β_array = β_array, Npretherm = 1000, Ntherm = 100, Nmeas = 1000, save_data = true, cfg_params = (; L, a, pd, seed, dim), verbose = 1)

### calcualte observables
e_mean = [mean(d.E) / V for d ∈ D.data]
m_mean = [mean(abs.(d.M)) / V for d ∈ D.data]
χ_mean = [(mean(d.M.^2) - mean(abs.(d.M)).^2) * d.β / V for d ∈ D.data]

### exact results for 2d
K(β; J = 1) = 1 / (sinh(2 * β * J) * sinh(2β * J))
IntΘ(β; J = 1) = quadgk(Θ -> 1 / sqrt(1 - 4 * K(β;J = J) * (1 + K(β;J = J))^-2 * sin(Θ)^2), 0, π / 2; rtol = 1e-8)[1]

u∞(β; J = 1) = -J * coth(2β * J) * (1 + 2 / π * (2 * tanh(2β * J)^2 - 1) * IntΘ(β; J = J))
m∞(β; J = 1) = β > log(1 + √2) / 2 / J ? (1 - sinh(2β * J)^(-4))^(1 / 8) : 0.0

### plot results
using Plots

plot(β_array, e_mean, label = "e")
plot!(β_array, m_mean, label = "m")
plot!(β_array, u∞.(β_array), label = "e exact")
plot!(β_array, m∞.(β_array),label = "m exact", xlabel = "β", ylabel = "e, m")
plot!(twinx(), β_array, χ_mean, label = "χ", legend = :right, ylabel = "χ")

### check correlated disorder 
IM = IsingModel(generate_defected_ising_lattice(16, 2, 1, 0.2, Inf), ntuple(d -> PeriodicBC(), dim), seed)
display_lattice(IM.Λ)

IM = IsingModel(generate_defected_ising_lattice(16, 2, 1, 0.2, 2.0), ntuple(d -> PeriodicBC(), dim), seed)
display_lattice(IM.Λ)

IM = IsingModel(generate_defected_ising_lattice(16, 2, 1, 0.2, 1.0), ntuple(d -> PeriodicBC(), dim), seed)
display_lattice(IM.Λ)
