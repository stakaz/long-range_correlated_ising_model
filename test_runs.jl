include("ising_model.jl")
include("disorder_generator.jl")

using QuadGK

dim = 2
a = 2.0
pd = 0.0
L = 32 
seed = 1

V = L^dim

β_array = 0.4:0.001:0.5

IM = IsingModel(generate_defected_ising_lattice(L, dim, seed, pd, a), ntuple(d -> PeriodicBC(), dim), seed)

D = run_simulation(IM; β_array = β_array, Npretherm = 1000, Ntherm = 1000, Nmeas = 10000, save_data = false, cfg_params = (; L, a, pd, seed, dim), verbose = 1)
# display_lattice(IM.Λ)

e_mean = [mean(d.E) / V for d ∈ D.data]
m_mean = [mean(abs.(d.M)) / V for d ∈ D.data]
χ_mean = [(mean(d.M.^2) - mean(abs.(d.M)).^2) * d.β / V for d ∈ D.data]

### exact results for 2d
K(β; J = 1) = 1 / (sinh(2 * β * J) * sinh(2β * J))
IntΘ(β; J = 1) = quadgk(Θ -> 1 / sqrt(1 - 4 * K(β;J = J) * (1 + K(β;J = J))^-2 * sin(Θ)^2), 0, π / 2; rtol = 1e-8)[1]

u∞(β; J = 1) = -J * coth(2β * J) * (1 + 2 / π * (2 * tanh(2β * J)^2 - 1) * IntΘ(β; J = J))
m∞(β; J = 1) = β > log(1 + √2) / 2 / J ? (1 - sinh(2β * J)^(-4))^(1 / 8) : 0.0

using Plots

plot(β_array, e_mean, label = "e")
plot!(β_array, m_mean, label = "m")
plot!(β_array, u∞.(β_array), label = "e exact")
plot!(β_array, m∞.(β_array),label = "m exact", xlabel = "β", ylabel = "e, m")
plot!(twinx(), β_array, χ_mean, label = "χ", legend = :right, ylabel = "χ")