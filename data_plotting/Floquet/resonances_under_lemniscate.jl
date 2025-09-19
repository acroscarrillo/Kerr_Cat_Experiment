include("../../src/src.jl") # import src.jl which has creation/annahilation operators defined

using LaTeXStrings # latex support for sexy figs
using DataFrames # this is like pandas
using CSV 
using ProgressBars
using Plots

# load data file
df_floquet =  DataFrame(CSV.File("data/floquet_crossings_comparison.csv"))

# focus on the first
df_floquet = filter(row ->  row.ΔE_n < 1000, df_floquet)

# n_photons cutoff
n_photons_cutoff = 150
df_floquet = filter(row ->  row.n_photons < n_photons_cutoff, df_floquet)


# Define parameter space
ϵ_1_array_floq = unique(df_floquet.ϵ_1)
ϵ_2_array_floq = unique(df_floquet.ϵ_2)

ylbl = L"\epsilon_2/K"
xlbl = L"\epsilon_1/K"

# Crossings
lemnis_surf(ϵ_1,ϵ_2) = ϵ_2^2 + 2*ϵ_1*sqrt(ϵ_2)

reso_htmp_array = zeros(length(ϵ_2_array_floq),length(ϵ_1_array_floq))
for (i,ϵ_2) in ProgressBar(enumerate(ϵ_2_array_floq))
    df_temp_1 = filter(row ->  row.ϵ_2 == ϵ_2, df_floquet)
    for (j,ϵ_1) in enumerate(ϵ_1_array_floq)

        y_max = 1.3 * lemnis_surf(ϵ_1,ϵ_2)
        y_min = 0.7 * lemnis_surf(ϵ_1,ϵ_2)
        df_temp_2 = filter(row ->  row.ΔE_n < y_max && row.ΔE_n > y_min,df_temp_1)
        df_temp_2 = filter(row -> row.ϵ_1 == ϵ_1, df_temp_2)
        ΔE_lemnis = sort(df_temp_2.ΔE_n)
        
        if length(ΔE_lemnis) > 1
            gap_array = zeros(length(ΔE_lemnis)-1)
            for n=1:(length(ΔE_lemnis)-1)
                gap_array[n] = ΔE_lemnis[n+1]-ΔE_lemnis[n]
            end
            reso_htmp_array[i,j] = minimum(gap_array)
        end

    end
end

width_pts = 246.0  # or any other value
inches_per_points = 1.0/72.27
width_inches = width_pts *inches_per_points
width_px= width_inches*100  # or  width_inches*DPI
heatmap(ϵ_1_array_floq,ϵ_2_array_floq,log.(reso_htmp_array),size = (width_px,width_px*0.6),xlab=xlbl,ylab=ylbl,title="Crossings under the barrier: #photons cutoff = $n_photons_cutoff",dpi=600,grid=false,xtickfontsize=8,ytickfontsize=8,guidefont=font(8),widen=false,tickdirection=:out,right_margin = 0Plots.mm,titlefontsize=8,legendfontsize=8,left_margin = 0Plots.mm,bottom_margin = 0Plots.mm,fontfamilys = "Times New Roman",tickfontfamily = "Times New Roman",background_color_subplot=:white,foreground_color_legend = nothing,label=false)