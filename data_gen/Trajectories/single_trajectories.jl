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

function renormalise_U(U)
    λ_n, V_n = eigen(U)  
    λ_n = λ_n ./ norm.(λ_n)
    return  V_n * diagm(λ_n) * V_n'
end

N = 200 #it works just fine for tests
N_traj = 200
N_saves = 1000

################
# H_eff params #
################

# Define H_eff parameter space, in units of K
K = 1
Δ = -0*K
ϵ_1 = 0*K
ϵ_2 = 4*K

#####################
# Trajectories code #
#####################

# Define Trajectories parameter space in units of ω_0
ω_0 = 1
# g_n = [0.00075, 1.27*10^(-7)].*ω_0 # exact match with H_eff (garcia mata fig 1)
# g_n = [-0.0035, -3.33e-5].*ω_0 #
# g_n = [-0.0034, -5.0892e-5].*ω_0 # experimental values (these has K=678kHz)
g_n = [-0.00331793, -3.33333333e-5].*ω_0 # experimental values (these has K=520kHz)
K = (10*g_n[1]^2)/(3*ω_0) - 3*g_n[2]/2

Ω_1 =  (ϵ_1*K*2)*ω_0 
Ω_2 = 3*(ϵ_2*K)/(2*g_n[1])*ω_0


# Define open sys parameters
# κ1/K = 0.025 and nth = 0.01
# κ_1 =  0.025 * K # in s^-1
# n_th = 0.01
κ_1 =  0.0 * K # in s^-1
n_th = 0.0
κ_p, κ_m = κ_1*n_th, κ_1*(1 + n_th) 
κ_p, κ_m  = κ_p, κ_m 

# Time considerations
ω_0_exp = 6e9 # in seconds
T_f = (ω_0_exp/(2*π))*(600e-6) # unit of ω_0
T_drive =  2*π/(2*ω_0)  # units of ω_0
δt = T_drive
t_array = Vector( range(0,T_f,step = δt) )
N_algo_steps = Int( length(t_array) )

steps_2_save = Vector( 1:200:N_algo_steps  )


C_ops = ([√(κ_p)*ComplexF64.(a(N)'), √(κ_m)*ComplexF64.(a(N))])
X = ( a(N) + a(N)' ) / √(2)


ω_1, ω_2 = ω_a(ω_0,g_n,Ω_2), 2*ω_a(ω_0,g_n,Ω_2)
ϵ_1, ϵ_2 = Ω_1/2, g_n[1]*Π(ω_0,Ω_2,ω_2)

# define ψ_0 as a coherent state at right well
x_well, p_well = dblwell_minimas(K,ϵ_1,ϵ_2)[1], 0
ψ_0 =  coherent_state(N,x_well,p_well) 

# load propagator in memory
U_T_temp = U_T_nh(N, ω_0, g_n, Ω_1, ω_1, Ω_2, ω_2, 0, 0)
U_T_temp = U_T_temp # * exp(-im*δt* H_0_nh(N,0,[0],κ_p,κ_m) )

# start Trajectories
ρ_t_final = zeros(ComplexF64,N,N)
avg_x_τ_array = zeros(length(steps_2_save))
for n=ProgressBar(1:N_traj)
    x_array, ψ_t_array = traj_ignore_nh(δt,N_algo_steps, ψ_0, U_T_temp, C_ops, X, steps_2_save)
    avg_x_τ_array += x_array ./ N_traj
    ρ_t_final += ψ_t_array[end,:]* ψ_t_array[end,:]'./ N_traj
end

τ_x = get_τ(range(0,ω_0_T_f,length(avg_x_τ_array)), avg_x_τ_array./ √(2*ϵ_2/K))
τ_x_μs = round(1e6.*τ_x./(2*π*320e3),sigdigits=5)

paper_τ_x_μs = round(1e6.*KT_f./(2*π*320e3),sigdigits=5)

scatter(KT_array,avg_x_τ_array./ √(2*ϵ_2/K), ylabel=L"\overline{\langle X \rangle}/\sqrt{2 ϵ_2/K }", xlabel=L"tK", title="Calculated "*L"\tau = "*"$τ_x_μs (μs), \n Paper val, "*L"\tau \sim "*"$paper_τ_x_μs (μs) at ϵ_2/K=$ϵ_2",ms=1,label="avg")


plot!(KT_array,exp.(-KT_array/τ_x),lw=2,label="exp. fit")
plot!(KT_array,exp.(-KT_array/KT_f),lw=2,label="paper. fit")

p_n,v_n = eigen(ρ_t_final)
wigner_temp = zeros(121,121)
for n=1:100
    wigner_temp += p_n[n]*wigner_func(v_n[:,n],6,6)
end
heatmap(wigner_temp)




x_array, ψ_t_array = traj_ignore_nh(δt,N_algo_steps, ψ_0, U_T_temp, C_ops, X, steps_2_save)
anim = @animate for n=ProgressBar(1:length(steps_2_save))
    heatmap(wigner_func( ψ_t_array[n,:], 6,6 ), title="$n")
end
gif(anim, "traj_floq.gif", fps = 15)



@gif for n=ProgressBar(1:length(steps_2_save))
    plot(norm.( ψ_t_array[n,:]), title="$n",ylim=(0,1))
end
