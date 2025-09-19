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

function avg_δp(U_T)
    N = size(U_T)[1]
    δ_p = 0
    for _=1:1000
        δ_p += ( 1 - norm(U_T*rand_ψ(N))^2 ) / 1000
    end
    return  δ_p
end

function get_δN(U_T,δt_thresh = 0.1)
    N = 1
    U_T_temp = copy(U_T)
    while avg_δp(U_T_temp) < δt_thresh
        U_T_temp = U_T_temp*U_T_temp
        N += 1
    end
    return N
end

N = 30 #it works just fine for tests
N_traj = 200

################
# H_eff params #
################

# Define H_eff parameter space, in units of K
K = 1
Δ = -0*K
ϵ_1_max = 15*K
ϵ_2_cut = 10*K

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

Ω_1_array = Vector( range(0,ϵ_1_max*K*2, length=20)).*ω_0 
ϵ_1_array = Vector( range(0,ϵ_1_max, length=length(Ω_1_array))).*ω_0 
Ω_2 = 3*(ϵ_2_cut*K)/(2*g_n[1])*ω_0
Ω_1_array = Ω_1_array[2:end]
ϵ_1_array = ϵ_1_array[2:end]

# Define open sys parameters
# κ1/K = 0.025 and nth = 0.01
κ_1 =  0.025 * K   # in s^-1
n_th = 0.1
κ_p, κ_m = κ_1*n_th, κ_1*(1 + n_th) 


# KT_f = 2πK*T_x, numbers from nick frattini
ω_0_exp = 6e9 # in seconds
T_f = (K*ω_0_exp)*(600e-6) # in seconds
T_drive =  2*π/(2*ω_0_exp) 
N_drives = T_f/T_drive
KT_array = Vector(range(0,4*KT_f, length=N_δt))
δt = 
N_saves = 200

C_ops = [√(κ_p)*a(N)', √(κ_m)*a(N)]
X = ( a(N) + a(N)' ) / √(2)


τ_x_array = zeros(length(Ω_1_array))
pbar = ProgressBar(total=length(Ω_1_array)*N_traj)
for (n,Ω_1) in enumerate(Ω_1_array)
    # define parameters
    ω_1, ω_2 = ω_a(ω_0,g_n,Ω_2), 2*ω_a(ω_0,g_n,Ω_2)
    ϵ_1, ϵ_2 = Ω_1/2, g_n[1]*Π(ω_0,Ω_2,ω_2)

    # define ψ_0 as a coherent state at right well
    x_well, p_well = dblwell_minimas(K,ϵ_1,ϵ_2)[2], 0
    ψ_0 = coherent_state(N,x_well,p_well)

    # load propagator in memory
    U_T_temp = U_T(N, ω_0, g_n, Ω_1, ω_1, Ω_2, ω_2, κ_p, κ_m)

    # start Trajectories
    avg_x_τ_array = zeros(N_saves)
    KT_array = zeros(N_saves)
    for n=1:N_traj
        x_array, ψ_t_array, t_array = trajectory(δt,N_δt, ψ_0, U_T_temp, C_ops, X; return_N = N_saves)
        avg_x_τ_array += x_array./ N_traj
        KT_array = t_array
        update(pbar)
    end

    τ_x_array[n] = get_τ(KT_array, avg_x_τ_array./ √(2*ϵ_2/K))
end 

width_pts = 246.0  # or any other value
inches_per_points = 1.0/72.27
width_inches = width_pts *inches_per_points
width_px= width_inches*100  # or  width_inches*DPI
plot(ϵ_2_array,1e6.*τ_x_array./(2*π*320e3), ylabel=L"T_x (\mu s)", xlabel=L"ϵ_2/K",lw=2,title=L"N="*"$N,   "*L"N_{traj}="*"$N_traj,   "*L"\kappa_1="*"$κ_1,   "*L"n_{th}="*"$n_th",yscale=:log)
scatter!(ϵ_2_array,1e6.*τ_x_array./(2*π*320e3), ms=2,legend=false,dpi=600,size = (width_px,width_px*0.6),grid=false,xtickfontsize=8,ytickfontsize=8,guidefont=font(8),widen=false,tickdirection=:out,right_margin = 4Plots.mm,titlefontsize=8,legendfontsize=8,left_margin = 0Plots.mm,bottom_margin = 0Plots.mm,fontfamilys = "Times New Roman",tickfontfamily = "Times New Roman",background_color_subplot=:white,foreground_color_legend = nothing,label=false)