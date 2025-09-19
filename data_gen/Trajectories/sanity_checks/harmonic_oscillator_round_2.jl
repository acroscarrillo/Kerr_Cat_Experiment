# include("../../src/src.jl") # import src.jl which has creation/annahilation operators defined

# Performance notes with typical parameters:
# @btime U_T(N=25),   6.610 ms (3612 allocations: 12.00 MiB)
# @btime U_T(N=50),   48.208 ms (8971 allocations: 87.18 MiB)
# @btime U_T(N=75),   531.844 ms (13185 allocations: 288.44 MiB)
# @btime U_T(N=100),  2.236 s (17442 allocations: 678.10 MiB)
# @btime U_T(N=150),  9.271 s (26130 allocations: 2.23 GiB)
# @btime U_T(N=200),  23.489 s (34949 allocations: 5.32 GiB)

using DataFrames # this is like pandas
using CSV 
using ProgressBars
using Plots
using LaTeXStrings
using LsqFit

τ_x_model(t, p) = 1 * exp.(- t / p[1])

function get_τ(x,y)
    τ = curve_fit(τ_x_model, x, y, [100.0]).param[1]
    return τ
end


#####################
# Global parameters #
#####################

N = 100 #it works just fine for tests
N_traj = 1000
N_saves = 1000

# everything in units of ω_0
ω_0 = 1
κ_1 =  0.25 
n_th = 0.1
κ_p, κ_m = κ_1*n_th, κ_1*(1 + n_th) 
κ_p, κ_m  = κ_p, κ_m 

C_ops = ([√(κ_p)*ComplexF64.(a(N)'), √(κ_m)*ComplexF64.(a(N))])
X = ( a(N) + a(N)' ) / √(2)
N_op = a(N)'*a(N)

# Time considerations
T_f =  10 * 1/(κ_1)   # unit of ω_0
N_algo_steps = Int( 1e3 )
t_array =  Vector( range(0,T_f, length=N_algo_steps) ) 
δt = t_array[2] - t_array[1] 

##################################################################################
################################ Exact propagator ################################
##################################################################################

# Load propagator in memory
U_T_temp = exp( -im * δt * H_0_nh(N,ω_0,[0],κ_p,κ_m) )

######################
# ψ_0 Coherent state #
######################
ψ_0 =  coherent_state(N,1,0) 

# start Trajectories
avg_X_τ_array = zeros(N_saves)
avg_N_τ_array = zeros(N_saves)
for n=ProgressBar(1:N_traj)
    x_array, ψ_t_array = traj_ignore_nh(δt,N_algo_steps, ψ_0, U_T_temp, C_ops, X; return_N = N_saves)
    N_array, ψ_t_array = traj_ignore_nh(δt,N_algo_steps, ψ_0, U_T_temp, C_ops, N_op; return_N = N_saves)
    avg_X_τ_array += x_array ./ N_traj
    avg_N_τ_array += N_array ./ N_traj
end
t_2_plot = Vector( range(0,T_f, length=N_saves) )  ./ 1/(1/κ_1) 

ttl = " Exact propagator. \n ψ_0 = coherent_state(N=$N,x=1,p=0). \n N_traj=$N_traj, ω_0 = $ω_0, κ_1=$κ_1, n_th=$n_th" 

left = scatter(t_2_plot,avg_X_τ_array, ylabel=L"\overline{\langle X \rangle}", xlabel=L"t/τ",ms=2,label="trajectories",markerstrokewidth=0,xticks = Vector(range(0,t_2_plot[end],length=6)),ylim=(-avg_X_τ_array[1]*1.15,avg_X_τ_array[1]*1.15) )
plot!(t_2_plot, exp.(-t_2_plot*(1/2)),lw=3,label=L"\exp(-t \kappa_1 /2)",linestyle=:dash)
hline!([1/ℯ],lw=3, ls=:dash,label=L"1/e" )
vline!([2*π/(ω_0) * κ_1 ],lw=3, ls=:dash,label=L"T \kappa_1" )
plot!(t_2_plot, cos.(t_2_plot*ω_0/κ_1).*exp.(-t_2_plot*(1/2)),lw=3,label=L"\cos(t ω_0)\exp(-t \kappa_1 /2)",linestyle=:dash)


right = scatter(t_2_plot,avg_N_τ_array, ylabel=L"\overline{\langle N \rangle}", xlabel=L"t",ms=2,label="trajectories",markerstrokewidth=0,xticks = Vector(range(0,t_2_plot[end],length=6)), ylim=(0,avg_N_τ_array[1]))
plot!(t_2_plot, (0.5-n_th).*exp.(-t_2_plot*(1)) .+ n_th,lw=3,label=L"(0.5-n_{th})\exp(-t \kappa_1) + n_{th}",linestyle=:dash)
hline!([1/ℯ],lw=3, ls=:dash,label=L"1/e" )
# vline!([ -log((1/ℯ-n_th)/(0.5-n_th)) ], lw=3, ls=:dash,label="-log((1/ℯ-n_th)/(0.5-n_th))" )


l = @layout [left right]
plot(left,right, layout=l,plot_title=ttl, top_margin=20Plots.mm,titlefontsize=10)


##################################################################################
############################## Numerical propagator ##############################
##################################################################################

# Load propagator in memory
U_T_temp = U_T_nh(N, ω_0, [0], 0, 0, 0, 4*π/δt, 0, 0)
U_T_temp = U_T_temp*exp(-im*δt* H_0_nh(N,0,[0],κ_p,κ_m) )

######################
# ψ_0 Coherent state #
######################
ψ_0 =  coherent_state(N,1,0) 

# start Trajectories
avg_X_τ_array = zeros(N_saves)
avg_N_τ_array = zeros(N_saves)
for n=ProgressBar(1:N_traj)
    x_array, ψ_t_array = traj_ignore_nh(δt,N_algo_steps, ψ_0, U_T_temp, C_ops, X; return_N = N_saves)
    N_array, ψ_t_array = traj_ignore_nh(δt,N_algo_steps, ψ_0, U_T_temp, C_ops, N_op; return_N = N_saves)
    avg_X_τ_array += x_array ./ N_traj
    avg_N_τ_array += N_array ./ N_traj
end
t_2_plot = Vector( range(0,T_f, length=N_saves) )  ./ 1/(1/κ_1) 

ttl = " Numerical propagator. \n ψ_0 = coherent_state(N=$N,x=1,p=0). \n N_traj=$N_traj, ω_0 = $ω_0, κ_1=$κ_1, n_th=$n_th" 

left = scatter(t_2_plot,avg_X_τ_array, ylabel=L"\overline{\langle X \rangle}", xlabel=L"t/τ",ms=2,label="trajectories",markerstrokewidth=0,xticks = Vector(range(0,t_2_plot[end],length=6)),ylim=(-avg_X_τ_array[1]*1.15,avg_X_τ_array[1]*1.15) )
plot!(t_2_plot, exp.(-t_2_plot*(1/2)),lw=3,label=L"\exp(-t \kappa_1 /2)",color=:orange,linestyle=:dash)
plot!(t_2_plot, -exp.(-t_2_plot*(1/2)),lw=3,label=false,linestyle=:dash,color=:orange)
hline!([1/ℯ],lw=3, ls=:dash,label=L"1/e" )
vline!([2*π/(ω_0) * κ_1 ],lw=3, ls=:dash,label=L"T \kappa_1" )
plot!(t_2_plot, cos.(t_2_plot*ω_0/κ_1).*exp.(-t_2_plot*(1/2)),lw=3,label=L"\cos(t ω_0)\exp(-t \kappa_1 /2)",linestyle=:dash)


right = scatter(t_2_plot,avg_N_τ_array, ylabel=L"\overline{\langle N \rangle}", xlabel=L"t",ms=2,label="trajectories",markerstrokewidth=0,xticks = Vector(range(0,t_2_plot[end],length=6)), ylim=(0.05,avg_N_τ_array[1]))
plot!(t_2_plot, (0.5-n_th).*exp.(-t_2_plot*(1)) .+ n_th,lw=3,label=L"(0.5-n_{th})\exp(-t \kappa_1) + n_{th}",linestyle=:dash)
hline!([1/ℯ],lw=3, ls=:dash,label=L"1/e" )
# vline!([ -log((1/ℯ-n_th)/(0.5-n_th)) ], lw=3, ls=:dash,label="-log((1/ℯ-n_th)/(0.5-n_th))" )


l = @layout [left right]
plot(left,right, layout=l,plot_title=ttl, top_margin=20Plots.mm,titlefontsize=10)



##################################################################################
############################## Wigner function anim. #############################
##################################################################################

x_array, ψ_t_array = traj_ignore_nh(δt,N_algo_steps, ψ_0, U_T_temp, C_ops, X; return_N = N_saves)
anim = @animate for n=ProgressBar(1:N_saves)
    heatmap(wigner_func( ψ_t_array[n,:], 6,6 ), title="$n")
end
gif(anim, "traj_floq.gif", fps = 15)





p_n,v_n = eigen(ρ_t_final)
wigner_temp = zeros(121,121)
for n=1:100
    wigner_temp += p_n[n]*wigner_func(v_n[:,n],6,6)
end
heatmap(wigner_temp)