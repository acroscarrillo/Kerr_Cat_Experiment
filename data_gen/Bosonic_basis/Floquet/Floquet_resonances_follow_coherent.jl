include("../../src/src.jl") # import src.jl which has creation/annahilation operators defined

using DataFrames # this is like pandas
using CSV 
using ProgressBars
using Plots
using LaTeXStrings

N = 30    

###################
# Units suffering #
###################
# experimental data
ω_0_exp = 6000 # MHz times 2π but it doesnt matter as it will divide out
g_3 = 3 * (-6.8) #MHz
g_4 = 4 * (-0.05) #MHz
# g_3 = 3 * (-7) #MHz
# g_4 = 4 * (-0.1) #MHz
ϵ_1_max = 2*14 #MHz
ϵ_2_max = 18 #MHz
K_exp = 0.500 #MHz not used anywhere, for reference only 

# experimental data in units of ω_0 
g_3 = g_3/ω_0_exp 
g_4 = g_4/ω_0_exp 
ϵ_1_max = ϵ_1_max/ω_0_exp
ϵ_2_max = ϵ_2_max/ω_0_exp
ω_0 = ω_0_exp/ω_0_exp

#################################################
# Define Floquet parameter space in units of ω_0 #
#################################################
# Define Floquet parameter space in units of ω_0
# ω_1 = 1
# g_n = [0.00075, 1.27*10^(-7)].*ω_0
g_n = [g_3,g_4]
# g_n = [-0.0025, -6.667*10^(-5)].*ω_0
K = (10*g_n[1]^2)/(3*ω_0) - 3*g_n[2]/2
Ω_1_array = Vector( range(0, (ϵ_1_max)/2, length=150) )[2:end]
Ω_2_array = Vector( range(0, 3*ϵ_2_max/(2*g_n[1]), length=200) )[2:end]


Ω_1_maxx = Ω_1_array[end]
Ω_2_maxx = Ω_2_array[end]

ϵ_1_max = 2*Ω_1_maxx # check: in ω_0 units like in above 
ϵ_2_max = Ω_2_maxx*2*g_n[1]/(3*ω_0) # check: in ω_0 units like in above 


# Define data form
data_array = zeros( (N-1)*length(Ω_1_array)*length(Ω_2_array), 5 ) # data form: ϵ_n | Δnn | ω_0 | Ω_1 | ω_1 | Ω_2 | ω_2 | K | ϵ_2 | ϵ_1 | N 

matrix_color = zeros(length(Ω_1_array),length(Ω_2_array))
r_array = zeros(length(Ω_1_array),length(Ω_2_array))

df_groundstates = DataFrame([[],[],[]], ["ϵ_1", "ϵ_2", "ψ"])
df_cohstates = DataFrame([[],[],[]], ["ϵ_1", "ϵ_2", "ψ_α"])


ϵ_1_array = zeros(length(Ω_1_array))
ϵ_2_array = zeros(length(Ω_2_array))

counter = 1
pbar = ProgressBar(total=length(Ω_1_array)*length(Ω_2_array))
Ω_1, Ω_2 = 0, 0
for (i,Ω_1) in enumerate(Ω_1_array)
    for (j,Ω_2) in enumerate(Ω_2_array)

        # calculate simulation parameters
        ω_1, ω_2 = ω_a(ω_0,g_n,Ω_2), 2*ω_a(ω_0,g_n,Ω_2)
        Π_temp = 2*Ω_2/(3*ω_2)
        ϵ_1, ϵ_2 = Ω_1/2, g_n[1]*Π_temp
        ϵ_1_array[i], ϵ_2_array[j] = ϵ_1, ϵ_2

        # Floquet 
        ϵ_n, ϕ_n = qen_qmodes(N, ω_0, g_n, Ω_1, ω_1, Ω_2, ω_2)

        x_well, p_well = dblwell_minimas(K,ϵ_1,ϵ_2)[1], 0
        coh_stt = coherent_state(N,x_well,p_well)

        overlaps_array = norm.( (coh_stt'*ϕ_n) ) # get matrix of overlaps

        levels_order_array = mapslices(argmax,overlaps_array,dims=2) # vector of levels positions, i.e. the groundstate position is levels_order_array[1].

        ϕ_0 = ϕ_n[:,levels_order_array[1]]
        push!(df_groundstates, [ϵ_1, ϵ_2, ϕ_0])
        push!(df_cohstates, [ϵ_1, ϵ_2, coh_stt])

        ϵ_0 = ϵ_n[levels_order_array[1]]
        ϵ_n = sort( mod.( ϵ_n .- ϵ_0, ω_2/2) ) # sort them all properly
        Δnn_array = zeros((N-1))
        for n=1:(N-1)
            Δnn_array[n] = mod( ϵ_n[n+1] - ϵ_n[n],  ω_2/2)
            data_array[counter,:] .= ϵ_n[n+1], Δnn_array[n], n, ϵ_2, ϵ_1
            counter += 1
        end

        temp_cross = zeros((N-1))
        for n=1:((N-1))
            temp_cross[n] =  mod( ϵ_n[n+1] - ϵ_n[n],  ω_2/2)
        end

        matrix_color[i,j] = minimum(temp_cross)

        r_array[i,j] = avg_spacing_ratio_min(Δnn_array)
    
        update(pbar)
    end
end

# heatmap of minimum 
ϵ_1_array_K = (ϵ_1_array*ω_0_exp)./(K*ω_0_exp) #back to units of K baby
ϵ_2_array_K = (ϵ_2_array*ω_0_exp)./(K*ω_0_exp) #back to units of K baby
heatmap(ϵ_1_array_K,ϵ_2_array_K,log.(matrix_color'), xlab=L"\epsilon_1/K",ylab=L"\epsilon_2/K",title=L"N="*string(N))

heatmap(ϵ_1_array[2:end],ϵ_2_array[60:70],log.(matrix_color[2:end,60:70]'), xlab=L"\epsilon_1/\omega_0",ylab=L"\epsilon_2/\omega_0",title=L"N="*string(N)*L".\omega_0="*string(ω_0_exp)*"MHz.")

# plot of r, level spacings
heatmap(ϵ_1_array_K,ϵ_2_array_K,r_array',xlab=L"\epsilon_1/K",ylab=L"\epsilon_2/K",title="Level ratio, "*L"r"*", landscape at "*L"N="*string(N)*L".\omega_0="*string(ω_0_exp)*"MHz.")

# Put data in convenient DataFrame object and save it
df_floquet = DataFrame(data_array, ["ϵ_n","Δnn","n","ϵ_2","ϵ_1"]) 

# CSV.write("data/floquet_resonances.csv", df_floquet)

l = @layout [a b c d; e f g h]
ϵ_2_cuts = [ϵ_2_array[1],ϵ_2_array[75],ϵ_2_array[125],ϵ_2_array[end]]
anim1 = @animate for ϵ_1 in ProgressBar(ϵ_1_array)
    plots_array = []

    # quasimodes
    for ϵ_2_cut in ϵ_2_cuts
        df_temp = filter(row -> row.ϵ_1 == ϵ_1 && row.ϵ_2 ==  ϵ_2_cut, df_groundstates)
        wigner_temp = wigner_func(df_temp.ψ[1]', 10,4)

        plot_temp = heatmap(-10:0.1:10,-4:0.1:4,wigner_temp,title=L"\epsilon_2="*string(round(100*ϵ_2_cut/ϵ_2_array[end]))*"%,  "*L"\epsilon_1="*string(round(100*ϵ_1/ϵ_1_array[end]))*"%",clim=(-0.3,0.3),dpi=600,colorbar=false,titlefontsize=6)

        df_temp = filter(row -> row.ϵ_1 == ϵ_1 && row.ϵ_2 ==  ϵ_2_cut, df_cohstates)
        x_avg = real(df_temp.ψ_α[1]'*(a(N)+a(N)')/√(2)*df_temp.ψ_α[1])
        plot!([1.2*x_avg],seriestype=:vline,c=:white,legend=false)

        push!(plots_array,plot_temp)
    end

    # coherent states
    for ϵ_2_cut in ϵ_2_cuts
        df_temp = filter(row -> row.ϵ_1 == ϵ_1 && row.ϵ_2 ==  ϵ_2_cut, df_cohstates)

        x_avg = real(df_temp.ψ_α[1]'*(a(N)+a(N)')/√(2)*df_temp.ψ_α[1])
        
        wigner_temp = wigner_func(df_temp.ψ_α[1]', 10,4)

        plot_temp = heatmap(-10:0.1:10,-4:0.1:4,wigner_temp,title=L"\epsilon_2="*string(round(100*ϵ_2_cut/ϵ_2_array[end]))*"%,  "*L"\epsilon_1="*string(round(100*ϵ_1/ϵ_1_array[end]))*"%",clim=(-0.3,0.3),dpi=600,colorbar=false,titlefontsize=6)

        plot!([1.2*x_avg],seriestype=:vline,c=:white,legend=false)

        push!(plots_array,plot_temp)
    end

    plot(plots_array..., layout = l)
end
gif(anim1, "anim_fps15.gif", fps = 10)

# l = @layout [a b; c d]
# ϵ_1_cuts = [ϵ_1_array[1],ϵ_1_array[75],ϵ_1_array[125],ϵ_1_array[end]]
# anim2 = @animate for ϵ_2 in ProgressBar(ϵ_2_array)
#     plots_array = []
#     for ϵ_1_cut in ϵ_1_cuts
#         df_temp = filter(row -> row.ϵ_2 == ϵ_2 && row.ϵ_1 ==  ϵ_1_cut, df_groundstates)
#         wigner_temp = wigner_func(df_temp.ψ[1][1]', 7,4)

#         plot_temp = heatmap(-7:0.1:7,-4:0.1:4,wigner_temp,title=L"\epsilon_1="*string(round(100*ϵ_1_cut/ϵ_1_array[end]))*"%,  "*L"\epsilon_2="*string(round(100*ϵ_2/ϵ_2_array[end]))*"%",clim=(-0.3,0.3))

#         push!(plots_array,plot_temp)
#     end
#     plot(plots_array..., layout = l)
# end
# gif(anim2, "anim_fps15.gif", fps = 10)

linea_negra(ϵ_1,ϵ_2) = ϵ_2^2 + 2*ϵ_1*sqrt(ϵ_2)
linea_fija(ϵ_1,ϵ_2) = 4*ϵ_2
linea_primera(ϵ_1,ϵ_2) = 4*ϵ_1*sqrt(ϵ_2)
anim = @animate for ϵ_2 in ProgressBar(ϵ_2_array)
    df_temp = filter(row -> row.ϵ_2 == ϵ_2, df_floquet)
    scatter(df_temp.ϵ_1,df_temp.ϵ_n,ms=1,ylims=(0.9,1),legend=false,title=L"\epsilon_2="*string(round(100*ϵ_2/ϵ_2_array[end]))*"%",markerstrokewidth=0,c=:black,dpi=600)
    plot!(df_temp.ϵ_1, linea_negra.(df_temp.ϵ_1,ϵ_2),linestyle=:dash,color=:red)
    plot!(df_temp.ϵ_1, linea_fija.(df_temp.ϵ_1,ϵ_2),linestyle=:dash,color=:red)
    plot!(df_temp.ϵ_1, linea_primera.(df_temp.ϵ_1,ϵ_2),linestyle=:dash,color=:blue)
end
gif(anim, "anim_fps15.gif", fps = 2)

# anim = @animate for i in ProgressBar(range(1,length(unique(df_floquet.ϵ_2))))
#     ϵ_2 = unique(df_floquet.ϵ_2)[i]
#     df_temp = filter(row -> row.ϵ_2 == ϵ_2 && row.n==1, df_floquet)
#     plot(df_temp.ϵ_1,df_temp.ϵ_n,ms=0.5,ylims=(0,1),legend=false,title=L"\epsilon_2="*string(round(100*ϵ_2/ϵ_2_array[end]))*"%",markerstrokewidth=0)
#     for n=2:(N-1)
#         df_temp = filter(row -> row.ϵ_2 == ϵ_2 && row.n==n, df_floquet)
#         plot!(df_temp.ϵ_1,df_temp.ϵ_n,ms=0.5,ylims=(0,1),legend=false,title=L"\epsilon_2="*string(round(100*ϵ_2/ϵ_2_array[end]))*"%",markerstrokewidth=0)
#     end
# end
# gif(anim, "anim_fps15.gif", fps = 10)


coh_stt = coherent_state(100,4)