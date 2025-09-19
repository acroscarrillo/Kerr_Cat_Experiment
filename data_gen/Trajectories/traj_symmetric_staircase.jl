include("../../src/src.jl") # import src.jl which has creation/annahilation operators defined

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


N = 20 #it works just fine for tests
N_traj = 1
N_T = Int( 1e4 )

################
# H_eff params #
################

# Define H_eff parameter space, in units of K
K = 1
Δ = -0*K
ϵ_1_cut = 0*K
ϵ_2_max = 14*K

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

Ω_2_array = Vector( range(0, 3*(ϵ_2_max*K)/(2*g_n[1]), length=100)).*ω_0 
ϵ_2_array = Vector( range(0, ϵ_2_max, length=length(Ω_2_array)))
Ω_2_array = Ω_2_array[2:end]
ϵ_2_array = ϵ_2_array[2:end]

Ω_1 = 0

# Define open sys parameters
#κ1/K = 0.025 and nth = 0.01
κ_1 =  0.00025
n_th = 0.001
κ_p, κ_m = κ_1*n_th, κ_1*(1 +n_th) 

C_ops = [κ_p*a(N), κ_m*a(N)']
obs = a(N) + a(N)'

# Define data form: avg_x_τ | τ | ϵ_1 | ϵ_2 
pbar = ProgressBar(total= N_T * length(Ω_2_array))
avg_x_τ_array = zeros( N_T, length(Ω_2_array) )
counter = 1
for (j,Ω_2) in ProgressBar(enumerate(Ω_2_array))
    # define parameters
    ω_1, ω_2 = ω_a(ω_0,g_n,Ω_2), 2*ω_a(ω_0,g_n,Ω_2)
    ϵ_1, ϵ_2 = Ω_1/2, g_n[1]*Π(ω_0,Ω_2,ω_2)

    # define ψ_0 as a coherent state at right well
    x_well, p_well = dblwell_minimas(K,ϵ_1,ϵ_2)[2], 0
    ψ_0 = coherent_state(N,x_well,p_well)

    # load propagator in memory
    U_T_temp = U_T(N, ω_0, g_n, Ω_1, ω_1, Ω_2, ω_2, κ_p, κ_m)

    x_temp = zeros(N_T)
    for n=1:N_traj
        x_temp += obs_traj(N_T, ψ_0, U_T_temp, C_ops, obs) ./ N_traj
        update(pbar)
    end
    avg_x_τ_array[:,j] = x_temp
end

width_pts = 246.0  # or any other value
inches_per_points = 1.0/72.27
width_inches = width_pts *inches_per_points
width_px= width_inches*100  # or  width_inches*DPI
heatmap(ϵ_2_array,1:N_T,avg_x_τ_array,xlab=L"ϵ_2/K",ylab=L"\tau",title=L"ϵ_1/K="*"$ϵ_1_cut",  colorbar_title="\n"*L"\langle X \rangle",ylim=(0,N_T),dpi=600,grid=false,xtickfontsize=8,ytickfontsize=8,guidefont=font(8),widen=false,tickdirection=:out,right_margin = 4Plots.mm,titlefontsize=8,legendfontsize=8,left_margin = 0Plots.mm,bottom_margin = 0Plots.mm,fontfamilys = "Times New Roman",tickfontfamily = "Times New Roman",background_color_subplot=:white,foreground_color_legend = nothing,label=false)#,size = (width_px,width_px*0.6))

# size = (width_px,width_px*0.6)