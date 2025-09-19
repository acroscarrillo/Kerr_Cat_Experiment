using QuantumOptics
using EasyFit

function get_τ(x,y)
    return fitexp(x,y).b
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

N = 100 # Maximum photon number

K = 1
Δ = 0*K
ϵ_1 = 0*K
ϵ_2 = 4*K

H_temp = H_eff_qo(N,Δ,K,ϵ_1,ϵ_2)

# κ1/K = 0.025 and nth = 0.01
κ_1 =  0.025 
n_th = 0.1
κ_p, κ_m = κ_1*n_th, κ_1*(1 + n_th) 


# KT_f = 2πK*T_x, numbers from nick frattini
KT_x = 250e-6 # in seconds
KT_f = 2*π*(320e3)*T_x
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

t_array, Ψt = timeevolution.mcwf(KT_array, Ψ₀, H_temp, J; rates=rates, display_beforeevent=true, display_afterevent=true,maxiters=1e7)


x_array = []
anim = @animate for (n,Ψ) in ProgressBar(enumerate(Ψt))
    t = round(t_array[n],sigdigits=2)
    right = heatmap(wigner_qo(  Ψ, 6,6 ),title="Wigner(Ψ(Kt=$t)) for ϵ_2 /K = $ϵ_2", cbarlims=(-0.3,0.3))

    push!(x_array, real(Ψ'*(A + A')*Ψ / √(2)))
    left = scatter(t_array[1:n],x_array ./ √(2*ϵ_2/K) ,xlim=(t_array[1]-0.1,t_array[end]),ylim=((-1.15)*1,1*(1.15)),ylabel=L"\langle X \rangle /\sqrt{2 ϵ_2/K }", xlabel = L"tK",legend=false,ms=2,left_margin = 6Plots.mm,bottom_margin = 3Plots.mm,)
    plot!(t_array[1:n],x_array./ √(2*ϵ_2/K))
    hline!([1,-1],linestyle=:dash)

    l = @layout [left right]
    plot(left, right, layout = l,size=(800,300),dpi=300)
end
gif(anim, "traject_sim_wigner.gif", fps = 15)
savefig("traject_sim_wigner.gif")