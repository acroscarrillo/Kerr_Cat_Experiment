# include("../../src/src.jl") 

using DataFrames # this is like pandas
using CSV 
using ProgressBars
using Plots
using LaTeXStrings


N = 100 # Maximum photon number
K = 1
Δ = 0*K
ϵ_1 = 0*K
ϵ_2 = 8*K

# κ1/K = 0.025 and nth = 0.01
κ_1 =  0.025 
n_th = 0.1
κ_p, κ_m = κ_1*n_th, κ_1*(1 + n_th) 


# KT_f = 2πK*T_x, numbers from nick frattini
KT_x = 230e-6 # in seconds
# KT_x = 300e-6 # in seconds
KT_f = 2*π*(320e3)*KT_x
N_δt = Int(  1e5  )
KT_array = Vector(range(0,4*KT_f, length=N_δt))
δt = KT_array[2]-KT_array[1]


C_ops = [√(κ_p)*a(N)', √(κ_m)*a(N)]
X = ( a(N) + a(N)' ) / √(2)

# define ψ_0 as a coherent state at right well
x_well, p_well = dblwell_minimas(K,ϵ_1,ϵ_2)[1], 0
ψ_0 = coherent_state(N,x_well,p_well)

# load propagator in memory
U_T_temp = exp(  -im * δt * H_eff_nh(N,Δ,K,ϵ_1,ϵ_2,κ_p,κ_m)  )

# start Trajectory
x_array, ψ_t_array, t_array = H_eff_obs_traj(δt,N_δt, ψ_0, U_T_temp, C_ops, X; return_N = 200)

anim = @animate for (n,t) in ProgressBar(enumerate(t_array))
    t = round(t_array[n],sigdigits=2)
    right = heatmap(wigner_func( ψ_t_array[n,:], 6,6 ),title="Wigner(Ψ(Kt=$t)) for ϵ_2 /K = $ϵ_2", cbarlims=(-0.3,0.3))

    left = scatter(t_array[1:n],x_array[1:n] ./ √(2*ϵ_2/K) ,xlim=(t_array[1]-0.1,t_array[end]),ylim=((-1.15)*1,1*(1.15)),ylabel=L"\langle X \rangle /\sqrt{2 ϵ_2/K }", xlabel = L"tK",legend=false,ms=2,left_margin = 6Plots.mm,bottom_margin = 3Plots.mm)
    plot!(t_array[1:n],x_array[1:n] ./ √(2*ϵ_2/K))
    hline!([1,-1],linestyle=:dash)

    l = @layout [left right]
    plot(left, right, layout = l,size=(800,300),dpi=300)
end
gif(anim, "traject_sim_wigner.gif", fps = 15)