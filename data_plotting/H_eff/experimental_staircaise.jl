using DataFrames # this is like pandas
using CSV 
using ProgressBars
using LaTeXStrings # latex support for sexy figs
using NPZ
# using Forecast
using Plots

ϵ_1_array = Vector( range(0,13.69,length=300) )
ϵ_2_array = Vector( range(2.5,12.5,length=300) )

# Experiment
exp_data = npzread("data/experimental/DecayData.npz")

K_exp = 0.505 # MHz or 505KHz

heatmap_mat_exp = exp_data["arr_2"] # μs, but irrelant for now

ϵ_1_array_exp = exp_data["arr_0"][1,:] # MHz
ϵ_2_array_exp = exp_data["arr_1"][:,1] # MHz

ϵ_1_array_K = ϵ_1_array_exp./K_exp # Units of Kerr
ϵ_2_array_K = ϵ_2_array_exp./K_exp # Units of Kerr


heatmap_exp = heatmap(ϵ_1_array_exp,ϵ_2_array_exp,heatmap_mat_exp,xlab=L"\epsilon_1/K",ylab=L"\epsilon_2/K",title="Experiment",titlefontsize=12,xtickfontsize=8,ytickfontsize=8,guidefont=font(12),height=400,colorbar=true,dpi=600, size=(720,400),clim=(0,400))

heatmap_exp = heatmap(ϵ_1_array_K,ϵ_2_array_K,heatmap_mat_exp,xlab=L"\epsilon_1/K",ylab=L"\epsilon_2/K",title="Experiment",titlefontsize=12,xtickfontsize=8,ytickfontsize=8,guidefont=font(12),height=400,colorbar=true,dpi=600, size=(720,400),clim=(0,400))


###################
# Analytics plots #
###################
# plot(ϵ_1_array,(2*ϵ_1_array).^(2/3),ylim=(2.5,12),linestyle=:solid,legend=false,lw=2,xlab=L"\epsilon_1/K",ylab=L"\epsilon_2/K",dpi=600)

findnearest(A::AbstractArray,t) = findmin(abs.(A.-t))[2]


l = @layout [a b; c d]
plots_array = []
for n=1:4
    stair_1 = []
    ϵ_2_stair_1 = []
    ϵ_1_stair_1 = []
    for (j,ϵ_1) in enumerate(ϵ_1_array_K)
        ϵ_2 = (ϵ_1/n)^2
        if (ϵ_2 > ϵ_2_array_K[1]) && (ϵ_2 < ϵ_2_array_K[end])
            ϵ_2_indx = findnearest(ϵ_2_array_K,ϵ_2)
            push!(ϵ_2_stair_1,ϵ_2_array_K[ϵ_2_indx])
            push!(ϵ_1_stair_1,ϵ_1)
            push!(stair_1,heatmap_mat_exp[ϵ_2_indx,j])
        end
    end
    # plot!(ϵ_1_stair_1,ϵ_2_stair_1,c=:white)
    scatter(ϵ_1_stair_1,stair_1,legend=false,ylab=L"T_{exp} \ (\mu s)",xlab=L"\epsilon_1/K",dpi=600,label="Data",title=L"n="*string(n),ms=1)

    vline!([√(ϵ_2_array_K[end-7])*n],c=:black,label="Chaos")

    model = loess(Float32.(ϵ_1_stair_1[1:end-9]), Float32.(stair_1[1:end-9]),span=0.35)
    us = range(extrema(Float32.(ϵ_1_stair_1[1:end-9]))...; step = 0.1)
    vs = predict(model, us)
    push!(plots_array,plot!(us, vs, label="Fit (LOESS)",legend=true,legendfontsize=6))
end


#pt = 0.0138889 
width_pts = 246.0  # or any other value
inches_per_points = 1.0/72.27
width_inches = width_pts *inches_per_points
width_px= width_inches*100  # or  width_inches*DPI
plot(plots_array..., layout = l, size = (2*width_px,width_px*0.5))
# savefig("delete.png")