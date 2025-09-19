# include("../../src/src.jl") 

using QuantumOptics
using LsqFit
using ProgressBars
using LaTeXStrings
using Plots

τ_x_model(t, p) = 1 * exp.(- t / p[1])

function get_τ(x,y)
    τ = curve_fit(τ_x_model, x, y, [1.0]).param[1]
    return τ
end

function H_eff_qo(N,Δ,K,ϵ_1,ϵ_2)
    basis = FockBasis(N)
    A = destroy(basis)
    return -Δ*A'*A + K*(A'^2)*(A^2) - ϵ_1*(A + A') - ϵ_2*(A^2 + A'^2)
end

function wigner_qo(ψ,xlim,plim,meshstep=0.1)
    wigner_f = wigner(ψ,-xlim:meshstep:xlim,-plim:meshstep:plim)
    return wigner_f' #transpose so that x is in x-axis
end

function x_exp(t::Float64, psi::Ket)
    i = findfirst(isequal(t), KT_array)
    x_exp_average[i] += real((psi'*X*psi)/norm(psi)^2)
end

N = 100 # Maximum photon number
N_traj = 10
K = 1
Δ = 0*K
ϵ_1 = 0*K
ϵ_2 = 8*K

H_temp = H_eff_qo(N,Δ,K,ϵ_1,ϵ_2)

# κ1/K = 0.025 and nth = 0.01
κ_1 =  0.025 
n_th = 0.3
κ_p, κ_m = κ_1*n_th, κ_1*(1 + n_th) 


# KT_f = 2πK*T_x, numbers from nick frattini
KT_x = 300e-6 # in seconds
KT_f = 2*π*(320e3)*KT_x
KT_array = Vector(range(0,2*KT_f,length=100))


##############
# Simulation #
##############
basis = FockBasis(N)
A = destroy(basis)
X = (A + A')/√(2)
J = [ A,  A' ]
rates = [κ_m, κ_p]

x_well, p_well = dblwell_minimas(K,ϵ_1,ϵ_2)[1], 0
α = (x_well + im*p_well)/√(2) 
Ψ₀ = coherentstate(basis, α)

# x_exp_average = zeros(Float64, length(KT_array))
for _=ProgressBar(1:N_traj)
    t_array, Ψt = timeevolution.mcwf(KT_array, Ψ₀, H_temp, J; rates=rates,maxiters=1e8)
    for (n,ψ) in enumerate(Ψt)
        j = findfirst(isequal(t_array[n]), KT_array)
        x_exp_average[j] += real( ψ'*X*ψ )
    end
end

# x_exp_average = x_exp_average ./ N_traj
τ_x = round(get_τ(KT_array, x_exp_average./ √(2*ϵ_2/K)),sigdigits=3)

KT_f = round(KT_f, sigdigits=2)
scatter(KT_array, x_exp_average ./ √(2*ϵ_2/K), ylabel=L"\overline{\langle X \rangle}/\sqrt{2 ϵ_2/K }", xlabel=L"tK", title="Calculated "*L"\tau K = "*"$τ_x, \n Paper val, "*L"\tau K \sim "*"$KT_f at ϵ_2/K=$ϵ_2")
plot!(KT_array,exp.(-KT_array/τ_x),lw=2,label="exp. fit")