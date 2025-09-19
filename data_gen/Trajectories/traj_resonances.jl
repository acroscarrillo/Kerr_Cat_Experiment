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


N = 50 #it works just fine for tests
N_traj = 100

################
# H_eff params #
################

# Define H_eff parameter space, in units of K
K = 1
Δ = -0*K
ϵ_1_array = Vector(range(0,16,length=200)).*K
ϵ_2_array = Vector(range(0,18,length=100)).*K
ϵ_1_array = ϵ_1_array[2:end]
ϵ_2_array = ϵ_2_array[2:end]
κ_p, κ_m = 0.01, 0.01

#####################
# Trajectories code #
#####################

# Define Floquet parameter space
ω_0 = 1
# g_n = [0.00075, 1.27*10^(-7)].*ω_0 # exact match with H_eff (garcia mata fig 1)
# g_n = [-0.0035, -3.33e-5].*ω_0 #
# g_n = [-0.0034, -5.0892e-5].*ω_0 # experimental values (these has K=678kHz)
g_n = [-0.00331793, -3.33333333e-5].*ω_0 # experimental values (these has K=520kHz)
K = (10*g_n[1]^2)/(3*ω_0) - 3*g_n[2]/2

Ω_1_array = Vector(range(0,Ω_1_max_by_ω_0(K,ϵ_1_array),length=1000)).*ω_0 
Ω_2_array = Vector(range(0,Ω_2_max_by_ω_0(K,g_n,ϵ_2_array),length=100)).*ω_0
Ω_1_array = Ω_1_array[2:end]
Ω_2_array = Ω_2_array[2:end]


# Define data form: x_τ | ϵ_1 | ϵ_2 
traj_data_array = zeros( length(Ω_1_array)*length(Ω_2_array), 3 )

counter = 1
pbar = ProgressBar(total=N_traj*length(Ω_1_array)*length(Ω_2_array))
Ω_1, Ω_2 = 0, 0
for Ω_1 in Ω_1_array
    for Ω_2 in Ω_2_array
        # define parameters
        ω_1, ω_2 = ω_a(ω_0,g_n,Ω_2), 2*ω_a(ω_0,g_n,Ω_2)
        ϵ_1, ϵ_2 = Ω_1/2, g_n[1]*Π(ω_0,Ω_2,ω_2)

        # define ψ_0 as a coherent state at right well
        x_well, p_well = dblwell_minimas(K,ϵ_1,ϵ_2)[1], 0
        ψ_0 = coherent_state(N,x_well,p_well)

        # load propagator in memory
        U_T_temp = U_T(N, ω_0, g_n, Ω_1, ω_1, Ω_2, ω_2,κ_p,κ_m)

        x_temp = zeros(N_traj)
        for n=1:N_traj
            ...
        end
        update(pbar)
    end
end

# Put data in convenient DataFrame object and save it
df_floquet = DataFrame(floquet_data_array, ["ΔE_n","ϵ_1","ϵ_2","n_photons","Floquet?"])
# CSV.write("data/floquet_crossings_comparison.csv", df_floquet)

# Combine both sims into a single dataframe
# df_comparison = DataFrame(vcat(floquet_data_array,H_eff_data_array), ["ΔE_n","ϵ_1","ϵ_2","n_photons","Floquet"]) 

# CSV.write("data/crossings_comparison.csv", df_comparison)