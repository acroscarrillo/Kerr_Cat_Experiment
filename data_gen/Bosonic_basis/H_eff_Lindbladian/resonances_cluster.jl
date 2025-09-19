include("../../../src/src.jl") # import src.jl which has creation/annahilation operators defined

# @btime eigen(H_super_op( -H_eff(35,0,1,0,12) ) + C_ops_super) 2.177 s (54 allocations: 128.60 MiB)

using DataFrames # this is like pandas
using CSV 
using ProgressBars
using Plots
using LaTeXStrings
using NPZ

K = 1
Δ = -0*K
ϵ_1_array = Vector(range(0,15,length=200)).*K
ϵ_2_array = Vector(range(0,12,length=199)).*K

T_1_data_array = zeros( length(ϵ_1_array), length(ϵ_2_array) )
for (n, ϵ_1) in enumerate(ϵ_1_array)
    T_1_temp = npzread("data_gen/Bosonic_basis/H_eff_Lindbladian/cluster_stuff/ep_1_$ϵ_1.npz")
    T_1_data_array[n,:] = T_1_temp
end
npzwrite("data_gen/Bosonic_basis/H_eff_Lindbladian/T_1_data_array.npz",T_1_data_array)

heatmap(ϵ_1_array,ϵ_2_array,log.(-T_1_data_array'))

plot(ϵ_2_array,log.(-T_1_data_array))

heatmap(ϵ_1_array,ϵ_2_array,log.(-T_1_data_array'))

plot(log.(-T_1_data_array')[end,:])