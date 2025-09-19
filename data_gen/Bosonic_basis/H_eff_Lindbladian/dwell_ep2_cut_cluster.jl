include("../../../src/src.jl") # import src.jl which has creation/annahilation operators defined

# @btime eigen(H_super_op( -H_eff(35,0,1,0,12) ) + C_ops_super) 2.177 s (54 allocations: 128.60 MiB)

using NPZ
using Plots
using LaTeXStrings
using LsqFit

function f_2_fit(t,p)
    τ, O = p[1], p[2]
    exp.(-t/τ) + O*(1 .-exp.(-t/τ))
end

function get_fit_params(P_array, t_array,p0=[10,0.75])
    fit = curve_fit(f_2_fit, t_array, P_array, p0)
    return fit.param
end

# function expo_time_heatmap(τ_array,t_steps=1000)
#     temp_mat = zeros(length(τ_array),t_steps)
# end

# cut at ϵ_2/K = 7.7
exp_data_cut = npzread("data_gen/Bosonic_basis/H_eff_Lindbladian/exp_cut_at_ep2_7_7.npz")

heatmap_mat_exp = exp_data_cut["arr_2"] # μs, but irrelant for now
ϵ_1_array_exp = exp_data_cut["arr_0"]
t_array_exp = exp_data_cut["arr_1"]

τ_array = zeros(length(ϵ_1_array_exp))
O_array = zeros(length(ϵ_1_array_exp))
for (j,ϵ_1) in enumerate(ϵ_1_array_exp)
    temp_cut = heatmap_mat_exp[:,j]
    τ_temp, O_temp = get_fit_params(temp_cut, t_array_exp)
    τ_array[j], O_array[j] = τ_temp, O_temp
end

heatmap(ϵ_1_array_exp,t_array_exp,heatmap_mat_exp,c=:seismic,clim=(0,1),widen=false,legend=false,ylab=L"t",xlab=L"\epsilon_1/K")
plot!(ϵ_1_array_exp,τ_array,lw=3,c=:green,ls=:solid)

# top = plot(ϵ_1_array_exp,τ_array/maximum(τ_array),ylabel=L"\tau/\tau_\max",label="exp")
# ϵ_1_array = Vector(range(0,15,length=200)).*K
# plot!(ϵ_1_array,T_1_data_array[128,:]/maximum(T_1_data_array[128,:]),label="theo")
# bot = plot(ϵ_1_array_exp,O_array,ylabel=L"O",legend=false)
# l = @layout [top; bot]
# plot(top, bot, layout=l,plot_title="Fitting "*L"\exp(-t/\tau) + O(1-\exp(-t/\tau))",dpi=600)

# Define H_eff parameter space, in units of K
N = 50 # scales as N^2

# Define H_eff parameter space, in units of K
K = 1
Δ = -0*K
ϵ_1_array = Vector(range(0,15,length=200)).*K
ϵ_2_array = Vector(range(0,12,length=199)).*K

# ϵ_2/K = 7.7
j, ϵ_2 =  128, ϵ_2_array[128]

T_1_data_array = -npzread("data_gen/Bosonic_basis/H_eff_Lindbladian/cluster_stuff/dwell_T_1_data_array.npz")'