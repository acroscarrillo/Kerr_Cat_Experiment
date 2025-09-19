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
k_4 = 1
Δ = -0
k_1_array = Vector(range(0,10,length=200))
k_2_array = Vector(range(5,15,length=199))


T_1_data_array = npzread("data_gen/Bosonic_basis/H_eff_Lindbladian/cluster_stuff/chem_T_1_data_array.npz")


k_2_array = k_2_array[54:160]
T_1_data_array = T_1_data_array[:,54:160]

k_2_T_max = zeros(length(k_2_array))
for (j,k_2) in enumerate(k_2_array)
    k_2_T_max[j] = k_1_array[argmax(T_1_data_array[:,j])]
end
plot(k_2_T_max, k_2_array)

@gif for (j,k_2) in enumerate(k_2_array)
    top = heatmap(k_1_array[1:end-1],k_2_array,log.(abs.(grad_T_array)),widen=false,c=:seismic,xlim=(0,10),ylab=L"\log \  || \partial_{\epsilon_1} T_x ||",xlab=L"k_1/K")
    # top = heatmap(k_1_array,k_2_array,log.(T_1_data_array)',widen=false,c=:seismic)
    scatter!(k_2_T_max[1:j],k_2_array[1:j],color=:lime,label=false,ms=2,markerstrokewidth=0)
    hline!([k_2],lw=2,legend=false,c=:white)

    k_2_temp = round(k_2,sigdigits=3)
    bot = plot(k_1_array,log.(T_1_data_array[:,j]),ylab=L"\log \  T_x",xlab=L"k_1/K",title="ϵ_2/K = $k_2_temp",grid=false,ylim=(minimum(log.(T_1_data_array)),1.1*maximum(log.(T_1_data_array[:,j]))))
    vline!([k_2_T_max[j]],c=:lime,legend=false)

    l = @layout [top ; bot]
    plot(top,bot, layout= l,dpi=100)
end


function scaled_shifted_diff(array)
    N = length(array)
    temp = zeros(N-1)
    for n=1:N-1
        temp[n] =  (array[n+1] - array[n])/(0.5*(array[n+1] + array[n]))
    end
    return temp
end

grad_T_array = zeros(length(k_2_array), length(k_1_array) -1)
for (j,k_2) in enumerate(k_2_array)
    grad_T_array[j,:] .= shifted_diff(T_1_data_array[:,j])
end


top = heatmap(k_1_array,k_2_array,log.(T_1_data_array)',widen=false,c=:seismic,xlim=(0,10),ylab=L"\log \  T_x",xlab=L"k_1/K",)
bot = heatmap(k_1_array[1:end-1],k_2_array,log.(abs.(grad_T_array)),widen=false,c=:seismic,xlim=(0,10),ylab=L"\log \  || \partial_{\epsilon_1} T_x ||",xlab=L"k_1/K",)
l = @layout [top ; bot]
plot(top,bot, layout= l,dpi=600)