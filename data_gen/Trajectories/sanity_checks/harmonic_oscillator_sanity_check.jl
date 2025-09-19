# include("../../src/src.jl") 

using DataFrames # this is like pandas
using CSV 
using ProgressBars
using Plots
using LaTeXStrings
using LsqFit

τ_x_model(t, p) = 1 * exp.(- t * p[1])

function get_τ(x,y)
    τ = curve_fit(τ_x_model, x, y, [100.0]).param[1]
    return τ
end

function H_harmonic_nh(N,ω,κ_p,κ_m)
    A = a(N) # annahilation op. up to dim N
    h_part =  ω*A'*A 
    nh_part = - im*0.5*(  (κ_m)*A'*A + (κ_p)*A*A' )
    return  h_part + nh_part
end

N = 10 # Maximum photon number
N_traj = 200
ω = 1


# κ1/K = 0.025 and nth = 0.01
κ_1 =  0.1
n_th = 0.000000001
κ_p, κ_m = κ_1*n_th, κ_1*(1 + n_th) 


# KT_f = (2πK)*T_x, numbers from nick frattini
N_δt = Int(  1e6  )
t_array = Vector(range(0,4*1/(κ_1/2), length=N_δt))
δt = t_array[2]-t_array[1]
N_saves = 200

ψ_0 = zeros(N)
ψ_0[2] = 1
ψ_0 = coherent_state(N,1,0)


C_ops = [√(κ_p)*a(N)', √(κ_m)*a(N)]
X = ( a(N) + a(N)' ) / √(2)
N_op = a(N)'*a(N)


# load propagator in memory
U_T_temp = exp(  -im * δt * H_harmonic_nh(N,ω,κ_p,κ_m)  )

# start Trajectories
ρ_t_final = zeros(ComplexF64,N,N)
avg_x_τ_array = zeros(N_saves)
t_array = zeros(N_saves)
for n=ProgressBar(1:N_traj)
    x_array, ψ_t_array, t_array = H_eff_obs_traj(δt,N_δt, ψ_0, U_T_temp, C_ops, X; return_N = N_saves)
    avg_x_τ_array += x_array./ N_traj
    t_array = t_array
    ρ_t_final += ψ_t_array[end,:]* ψ_t_array[end,:]'./ N_traj
end

τ_x = get_τ(t_array, avg_x_τ_array)
τ_x_μs = round(τ_x,sigdigits=5)


scatter(t_array,avg_x_τ_array, ylabel=L"\overline{\langle X \rangle}/\sqrt{2 ϵ_2/K }", xlabel=L"tK", title="Calculated "*L"\tau = "*"$τ_x_μs",ms=2,label="avg")
plot!(t_array,avg_x_τ_array, ylabel=L"\overline{\langle X \rangle}/\sqrt{2 ϵ_2/K }", xlabel=L"tK", title="Calculated "*L"\tau = "*"$τ_x_μs",ms=2,label="avg")


plot!(t_array,exp.(-t_array*τ_x),lw=2,label="exp. fit")

# p_n,v_n = eigen(ρ_t_final)
# wigner_temp = zeros(121,121)
# for n=1:N
#     wigner_temp += real.(p_n[n]*wigner_func(v_n[:,n],6,6))
# end
# heatmap(wigner_temp)

x_array, ψ_t_array, t_array = trajectory(δt,N_δt, ψ_0, U_T_temp, C_ops, X; return_N = N_saves)

scatter(x_array)