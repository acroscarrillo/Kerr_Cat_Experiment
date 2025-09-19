# include("../../src/src.jl") 

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

N = 100 # Maximum photon number
N_traj = 200
K = 1
Δ = 0*K
ϵ_1 = 0*K
ϵ_2 = 3*K

# κ1/K = 0.025 and nth = 0.01
κ_1 =  0.025 
n_th = 0.1
κ_p, κ_m = κ_1*n_th, κ_1*(1 + n_th) 


# KT_f = (2πK)*T_x, numbers from nick frattini
KT_x = 600e-6 # in seconds
KT_f = (2*π*320e3)*KT_x
N_δt = Int(  5e5  )
KT_array = Vector(range(0,4*KT_f, length=N_δt))
δt = KT_array[2]-KT_array[1]
N_saves = 200

# define ψ_0 as a coherent state at right well
x_well, p_well = dblwell_minimas(K,ϵ_1,ϵ_2)[1], 0
ψ_0 = coherent_state(N,x_well,p_well)


C_ops = [√(κ_p)*a(N)', √(κ_m)*a(N)]
X = ( a(N) + a(N)' ) / √(2)


# load propagator in memory
U_T_temp = exp(  -im * δt * H_eff_nh(N,Δ,K,ϵ_1,ϵ_2,κ_p,κ_m)  )

# start Trajectories
ρ_t_final = zeros(ComplexF64,N,N)
avg_x_τ_array = zeros(N_saves)
KT_array = zeros(N_saves)
for n=ProgressBar(1:N_traj)
    x_array, ψ_t_array, t_array = trajectory(δt,N_δt, ψ_0, U_T_temp, C_ops, X; return_N = N_saves)
    avg_x_τ_array += x_array./ N_traj
    KT_array = t_array
    ρ_t_final += ψ_t_array[end,:]* ψ_t_array[end,:]'./ N_traj
end

τ_x = get_τ(KT_array, avg_x_τ_array./ √(2*ϵ_2/K))
τ_x_μs = round(1e6.*τ_x./(2*π*320e3),sigdigits=5)

paper_τ_x_μs = round(1e6.*KT_f./(2*π*320e3),sigdigits=5)

scatter(KT_array,avg_x_τ_array./ √(2*ϵ_2/K), ylabel=L"\overline{\langle X \rangle}/\sqrt{2 ϵ_2/K }", xlabel=L"tK", title="Calculated "*L"\tau = "*"$τ_x_μs (μs), \n Paper val, "*L"\tau \sim "*"$paper_τ_x_μs (μs) at ϵ_2/K=$ϵ_2",ms=1,label="avg")
plot!(KT_array,exp.(-KT_array/τ_x),lw=2,label="exp. fit")
plot!(KT_array,exp.(-KT_array/KT_f),lw=2,label="paper. fit")

# p_n,v_n = eigen(ρ_t_final)
# wigner_temp = zeros(121,121)
# for n=1:100
#     wigner_temp += p_n[n]*wigner_func(v_n[:,n],6,6)
# end
# heatmap(wigner_temp)