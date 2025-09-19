include("../../src/main.jl")

using DataFrames # this is like pandas
using CSV 
using ProgressBars
using LaTeXStrings # latex support for sexy figs


function find_barrier_height(ϵ_1,ϵ_2)
    H_p0_cut(x) = H_cl(x,0,ϵ_1,ϵ_2)
    H_p0_cut_derivative(x) = gradient(H_p0_cut, x)[1]
    all_zeros = find_zeros(H_p0_cut_derivative,x_array[1],x_array[end])
    if length(all_zeros)==3
        x_barrier = sort(all_zeros)[2]
        E_barrier = H_p0_cut(x_barrier)
        return E_barrier, x_barrier
    else 
        return nothing, nothing
    end
end


function left_well_area_MC(ϵ_1,ϵ_2,E_barrier,x_barrier,x_array,p_array)
    n=0
    for x in x_array
        for p in p_array
            if (H_cl(x,p,ϵ_1,ϵ_2)-E_barrier <= 0) && (x <= x_barrier) # left well
                n += 1
            end
        end
    end
    total_points = length(x_array) * length(p_array) 
    total_area = (x_array[end]-x_array[1])*(p_array[end]-p_array[1])
    return total_area*(n/total_points)
end


function right_well_area_MC(ϵ_1,ϵ_2,E_barrier,x_barrier,x_array,p_array)
    n=0
    for x in x_array
        for p in p_array
            if (H_cl(x,p,ϵ_1,ϵ_2)-E_barrier <= 0) && (x >= x_barrier) # left well
                n += 1
            end
        end
    end
    total_points = length(x_array) * length(p_array) 
    total_area = (x_array[end]-x_array[1])*(p_array[end]-p_array[1])
    return total_area*(n/total_points)
end


function contour_line(mat,ϵ_1_array,ϵ_2_array,val, tol)
    indxs =  findall(val+tol .>= mat .>= val-tol)
    x_2_plot = []
    y_2_plot = []
    for ind in indxs
        i,j = ind[1], ind[2]
        push!(x_2_plot,ϵ_1_array[i])
        push!(y_2_plot,ϵ_2_array[j])
    end
    return x_2_plot, y_2_plot
end

ϵ_1_array = Vector( range(0,12.5,length=300) )
ϵ_2_array = Vector( range(0,12.5,length=300) )

################
# Area Heatmap #
################
x_array = Vector(range(-10,10,length=1000)) #it's a bit choppy but oh well
p_array = Vector(range(-6,6,length=1000))

# left_well_area_matrix = zeros((length(ϵ_2_array),length(ϵ_1_array)))
# right_well_area_matrix = zeros((length(ϵ_2_array),length(ϵ_1_array)))
# for (j,ϵ_1) in ProgressBar(enumerate(ϵ_1_array))
#     for (k,ϵ_2) in enumerate(ϵ_2_array)

#         E_barrier, x_barrier = find_barrier_height(ϵ_1,ϵ_2)

#         if E_barrier !== nothing
#             left_well_area_matrix[j,k] = left_well_area_MC(ϵ_1,ϵ_2,E_barrier,x_barrier,x_array,p_array)
#             right_well_area_matrix[j,k] = right_well_area_MC(ϵ_1,ϵ_2,E_barrier,x_barrier,x_array,p_array)
#         else
#             left_well_area_matrix[j,k] = -0.5
#             right_well_area_matrix[j,k] = -0.5
#         end

#     end
# end

# contour(ϵ_1_array,ϵ_2_array,left_well_area_matrix')
# contour!(ϵ_1_array,ϵ_2_array,right_well_area_matrix')

###################
# Analytics plots #
###################
# plot(ϵ_1_array,(2*ϵ_1_array).^(2/3),ylim=(2.5,12),linestyle=:solid,legend=false,lw=2,xlab=L"\epsilon_1/K",ylab=L"\epsilon_2/K",dpi=600)

plot(ϵ_1_array,(ϵ_1_array/1).^2,ylim=(2.5,12),legend=false,linewidth=2,alpha=1,linestyle=:solid,dpi=600,xlab=L"\epsilon_1/K",ylab=L"\epsilon_2/K")
plot!(ϵ_1_array,(ϵ_1_array/2).^2,ylim=(2.5,12),legend=false,linewidth=2,alpha=1,linestyle=:solid)
plot!(ϵ_1_array,(ϵ_1_array/3).^2,ylim=(2.5,12),legend=false,linewidth=2,alpha=1,linestyle=:solid)
plot!(ϵ_1_array,(ϵ_1_array/4).^2,ylim=(2.5,12),legend=false,linewidth=2,alpha=1,linestyle=:solid)


###################
# Equi-Area plots #
###################
plot!(contour_line(left_well_area_matrix,ϵ_1_array,ϵ_2_array,2*π*0, 0.05),ylim=(0,12),lw=2,alpha=1,c=:black)

plot!(contour_line(left_well_area_matrix,ϵ_1_array,ϵ_2_array,2*π*0.5, 0.05),ylim=(0,12),lw=2,alpha=1,c=:black)
plot!(contour_line(right_well_area_matrix,ϵ_1_array,ϵ_2_array,2*π*0.5, 0.05),ylim=(0,12),lw=2,alpha=1,c=:black)

plot!(contour_line(left_well_area_matrix,ϵ_1_array,ϵ_2_array,2*π*1.5, 0.05),ylim=(0,12),lw=2,alpha=1,c=:black)
plot!(contour_line(right_well_area_matrix,ϵ_1_array,ϵ_2_array,2*π*1.5, 0.05),ylim=(0,12),lw=2,alpha=1,c=:black)

plot!(contour_line(left_well_area_matrix,ϵ_1_array,ϵ_2_array,2*π*2.5, 0.05),ylim=(0,12),lw=2,alpha=1,c=:black)
plot!(contour_line(right_well_area_matrix,ϵ_1_array,ϵ_2_array,2*π*2.5, 0.05),ylim=(0,12),lw=2,alpha=1,c=:black)

plot!(contour_line(left_well_area_matrix,ϵ_1_array,ϵ_2_array,2*π*3.5, 0.05),ylim=(0,12),lw=2,alpha=1,c=:black)
plot!(contour_line(right_well_area_matrix,ϵ_1_array,ϵ_2_array,2*π*3.5, 0.05),ylim=(0,12),lw=2,alpha=1,c=:black)

plot!(contour_line(left_well_area_matrix,ϵ_1_array,ϵ_2_array,2*π*4.5, 0.05),ylim=(0,12),lw=2,alpha=1,c=:black)
plot!(contour_line(right_well_area_matrix,ϵ_1_array,ϵ_2_array,2*π*4.5, 0.05),ylim=(0,12),lw=2,alpha=1,c=:black)

plot!(contour_line(left_well_area_matrix,ϵ_1_array,ϵ_2_array,2*π*5.5, 0.05),ylim=(0,12),lw=2,alpha=1,c=:black)
plot!(contour_line(right_well_area_matrix,ϵ_1_array,ϵ_2_array,2*π*5.5, 0.05),ylim=(0,12),lw=2,alpha=1,c=:black)



# plot!(ϵ_1_array[1:160],(-4.7*(ϵ_1_array[1:160] .- 7.5)).^(1/2) .+ 5)


# savefig("figs/important_figs/H_eff/analytics_only.png")
