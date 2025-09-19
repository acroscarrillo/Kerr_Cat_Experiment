include("../../../src/src.jl") # import src.jl which has creation/annahilation operators defined

# @btime eigen(H_super_op( -H_eff(35,0,1,0,12) ) + C_ops_super) 2.177 s (54 allocations: 128.60 MiB)

using DataFrames # this is like pandas
using CSV 
using ProgressBars
using Plots
using LaTeXStrings
using NPZ

# Define H_eff parameter space, in units of K
N = 50 # scales as N^2

# Define H_eff parameter space, in units of K
K = 1
Δ = -0*K
ϵ_1_array = Vector(range(0,15,length=200)).*K
ϵ_2_array = Vector(range(0,12,length=199)).*K

T_1_data_array = npzread("data_gen/Bosonic_basis/H_eff_Lindbladian/cluster_stuff/dwell_zero_temp_T_1_data_array.npz")

ϵ_2_T_max = zeros(length(ϵ_2_array))
for (j,ϵ_2) in enumerate(ϵ_2_array)
    ϵ_2_T_max[j] = ϵ_1_array[argmax(T_1_data_array[:,j])]
end
plot(ϵ_2_T_max, ϵ_2_array)

heatmap(log.(-T_1_data_array)')

@gif for (j,k_2) in enumerate(k_2_array)
    top = heatmap(k_1_array,k_2_array,log.(T_1_data_array)',widen=false)
    scatter!(k_2_T_max[1:j],k_2_array[1:j],color=:lime,label=false,ms=2,markerstrokewidth=0)

    k_2_temp = round(k_2,sigdigits=2)
    bot = plot(k_1_array,log.(T_1_data_array[:,j]),ylab=L"\log \  T_x",xlab=L"ϵ_1/K",title="ϵ_2/K = $k_2_temp",grid=false,ylim=(minimum(log.(T_1_data_array)),1.1*maximum(log.(T_1_data_array[:,j]))))
    vline!([k_2_T_max[j]],c=:lime,legend=false)

    l = @layout [top ; bot]
    plot(top,bot, layout= l,dpi=100)
end