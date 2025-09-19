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

T_1_data_array = -npzread("data_gen/Bosonic_basis/H_eff_Lindbladian/cluster_stuff/dwell_T_1_data_array.npz")

ϵ_2_T_max = zeros(length(ϵ_2_array))
for (j,ϵ_2) in enumerate(ϵ_2_array)
    ϵ_2_T_max[j] = ϵ_1_array[argmax(T_1_data_array[:,j])]
end
heatmap(ϵ_1_array,ϵ_2_array,log.(T_1_data_array'),widen=false,c=:seismic)
plot!(ϵ_2_T_max, ϵ_2_array,c=:black,legend=false,lw=3)

@gif for (j,ϵ_2) in enumerate(ϵ_2_array)
    top = heatmap(ϵ_1_array,ϵ_2_array,log.(T_1_data_array)',widen=false)
    scatter!(ϵ_2_T_max[1:j],ϵ_2_array[1:j],color=:lime,label=false,ms=2,markerstrokewidth=0)

    ϵ_2_temp = round(ϵ_2,sigdigits=2)
    bot = plot(k_1_array,log.(T_1_data_array[:,j]),ylab=L"\log \  T_x",xlab=L"ϵ_1/K",title="ϵ_2/K = $ϵ_2_temp",grid=false,ylim=(minimum(log.(T_1_data_array)),1.1*maximum(log.(T_1_data_array[:,j]))))
    vline!([ϵ_2_T_max[j]],c=:lime,legend=false)

    l = @layout [top ; bot]
    plot(top,bot, layout= l,dpi=100)
end



j,ϵ_2 = 199, 12.0
top = heatmap(ϵ_1_array,ϵ_2_array,log.(T_1_data_array)',widen=false)
scatter!(ϵ_1_array[1:j],ϵ_2_T_max[1:j],color=:lime,label=false,ms=2,markerstrokewidth=0)

ϵ_2_temp = round(ϵ_2,sigdigits=2)
bot = plot(k_1_array,log.(T_1_data_array[:,j]),ylab=L"\log \  T_x",xlab=L"ϵ_1/K",title="ϵ_2/K = $ϵ_2_temp",grid=false,ylim=(11.5,12))
vline!([ϵ_2_T_max[j]],c=:lime,legend=false)

l = @layout [top ; bot]
plot(top,bot, layout= l,dpi=100)





function scaled_shifted_diff(array)
    N = length(array)
    temp = zeros(N-1)
    for n=1:N-1
        temp[n] =  (array[n+1] - array[n])
    end
    return temp
end

grad_T_array = zeros(length(ϵ_2_array), length(ϵ_1_array) -1)
for (j,ϵ_2) in enumerate(ϵ_2_array)
    grad_T_array[j,:] .= shifted_diff(T_1_data_array[:,j]) / (ϵ_1_array[2]-ϵ_1_array[1])
end

grad_grad_T_array = zeros(length(ϵ_2_array), length(ϵ_1_array) -2)
for (j,ϵ_2) in enumerate(ϵ_2_array)
    grad_grad_T_array[j,:] .= shifted_diff(grad_T_array[:,j])
end

top = heatmap(ϵ_1_array,ϵ_2_array,log.(T_1_data_array)',widen=false,c=:seismic,xlim=(0,15),ylab=L"ϵ_2/K",xlab=L"ϵ_1/K",title=L"\log \ T_x")
bot = heatmap(ϵ_1_array[1:end-1],ϵ_2_array,log.(abs.(grad_T_array)),widen=false,c=:seismic,xlim=(0,15),title=L"\log \  || \partial_{\epsilon_1} T_x ||",ylab=L"ϵ_2/K",xlab=L"ϵ_1/K",)
l = @layout [top ; bot]
plot(top,bot, layout= l,dpi=600)