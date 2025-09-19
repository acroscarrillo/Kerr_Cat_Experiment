include("../../../src/src.jl") # import src.jl which has creation/annahilation operators defined

# @btime eigen(H_super_op( -H_eff(35,0,1,0,12) ) + C_ops_super) 2.177 s (54 allocations: 128.60 MiB)

using NPZ
using Plots
using LaTeXStrings

# Define H_eff parameter space, in units of K
N = 50 # scales as N^2

# Define H_eff parameter space, in units of K
K = 1
Δ = -0*K
ϵ_1_array = Vector(range(0,15,length=200)).*K
ϵ_2_array = Vector(range(0,12,length=199)).*K

T_1_data_array = -npzread("data_gen/Bosonic_basis/H_eff_Lindbladian/cluster_stuff/dwell_T_1_data_array.npz")'

width_px= 340.39020340390204
heatmap(ϵ_1_array,ϵ_2_array,log10.(T_1_data_array), size = (width_px,width_px*0.6),xlab=L"ϵ_1/K",ylab=L"ϵ_2/K",cbartitle=L"\log\left( KT \right)",xtickfontsize=8,ytickfontsize=8,guidefont=font(8),colorbar=true,dpi=600,widen=false,tickdirection=:out,right_margin = 0Plots.mm,colorbar_tickfontsize=8,left_margin = 0Plots.mm,bottom_margin = 0Plots.mm,fontfamilys = "Times New Roman",tickfontfamily = "Times New Roman",title="Theory",titlefontsize=8,xlim=/(minimum(ϵ_1_array),maximum(ϵ_1_array)),ylim=(minimum(ϵ_2_array),maximum(ϵ_2_array)),c=:seismic)


savefig("data_plotting/paper_plots/figures/resos_heatmap_Lind_theo.png")
savefig("data_plotting/paper_plots/figures/resos_heatmap_Lind_theo.pdf")
savefig("data_plotting/paper_plots/figures/resos_heatmap_Lind_theo.svg")

