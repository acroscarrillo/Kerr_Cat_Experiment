using QuantumOptics

function H_eff_qo(N,Δ,K,ϵ_1,ϵ_2)
    basis = FockBasis(N)
    A = destroy(basis)
    return -Δ*A'*A + K*(A'^2)*(A^2) - ϵ_1*(A + A') - ϵ_2*(A^2 + A'^2)
end

function wigner_qo(ψ,xlim,plim,meshstep=0.1)
    wigner_f = wigner(ψ,-xlim:meshstep:xlim,-plim:meshstep:plim)
    return wigner_f' #transpose so that x is in x-axis
end


N = 100 # Maximum photon number
T = [0:10:800;]

K = 1
Δ = 0*K
ϵ_1 = 0*K
ϵ_2 = 10*K

H_temp = H_eff_qo(N,Δ,K,ϵ_1,ϵ_2)

# κ1/K = 0.025 and nth = 0.01
κ_1 =  2.5
n_th = 1
κ_p, κ_m = κ_1*n_th, κ_1*(1 + n_th) 

basis = FockBasis(N)
A = destroy(basis)
J = [ A,  A' ]
rates = [κ_m, κ_p]

x_well, p_well = dblwell_minimas(K,ϵ_1,ϵ_2)[1], 0
α = (x_well + im*p_well)/√(2) 
Ψ₀ = coherentstate(basis, α)
ρ₀ = Ψ₀ ⊗ dagger(Ψ₀)

tout, ρt_master = timeevolution.master(T, ρ₀, H_temp, J,rates=rates)

before = heatmap(wigner_qo(ρt_master[1], 6,6))
after = heatmap(wigner_qo(ρt_master[end], 6,6))
l = @layout [before after]
plot(before, after, layout = l, size=(900,400))

anim = @animate for ρ in ProgressBar(ρt_master)
    heatmap(wigner_qo(  ρ, 6,6 ),title="Wigner(ρ(t)) for ϵ_2 /K = $ϵ_2")
end
gif(anim, "master_sim_wigner.gif", fps = 15)
