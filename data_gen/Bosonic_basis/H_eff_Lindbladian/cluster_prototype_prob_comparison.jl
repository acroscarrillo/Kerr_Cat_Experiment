# !/usr/bin/env julia

# @btime eigen(H_super_op( -H_eff(35,0,1,0,12) ) + C_ops_super) 2.177 s (54 allocations: 128.60 MiB)

using LinearAlgebra
using NPZ
using Roots
using Flux
using LsqFit
using ProgressBars

⦼(A,B) = kron(A,B)

function a(N::Int)
    a = zeros(Float64, (N, N))
    for n in 1:N-1
        a[n,n+1] = sqrt(n)
    end
    return a
end

function H_eff(N,Δ,K,ϵ_1,ϵ_2)
    A = a(N) # annahilation op. up to dim N
    return Δ*A'*A - K*(A'^2)*(A^2) + ϵ_1*(A + A') + ϵ_2*(A^2 + A'^2)
end

function C_super_ops(C_ops)
    N = size(C_ops[1])[1]
    D_temp = zeros(N^2,N^2)
    for C in C_ops
        D_temp += transpose(C') ⦼ C 
        D_temp += -0.5 * I(N) ⦼ (C'*C)
        D_temp += -0.5 * transpose(C'*C) ⦼ I(N)
    end
    return D_temp
end

function H_super_op(H)
    N = size(H)[1]
    return  -im * ( I(N) ⦼ H - transpose(H) ⦼ I(N) )
end

function smallest_nonzero_real(vec)
    temp_real = abs.(real.(vec))
    temp_non_zero = temp_real[ temp_real .> 1e-10 ]
    return minimum(temp_non_zero)
end

function coherent_state(N,α)
    A = a(N)
    vac = zeros(N)
    vac[1] = 1
    coh_state =  exp(-α*A+(α*A)')*vac
    return coh_state/norm(coh_state)
end

function coherent_state(N,x,p)
    α =  (x + im*p)/√(2) 
    return coherent_state(N,α)
end

function f_2_fit(t,p)
    τ, O = p[1], p[2]
    return (1 .- exp.(-t/τ)) .+ O*(1 .- exp.(-t/τ))
end

function f_2_fit_alt(t,p)
    τ, A, O = p[1], p[2], p[3]
    return A*(1 .- exp.(-t/τ)) .+ O
end

function get_fit_params(P_array, t_array, p0=[10.0,1])
    fit = curve_fit(f_2_fit, t_array, P_array, p0,lower=[0.0,0.5])
    return fit.param
end

function get_fit_params_alt(P_array, t_array, p0=[10.0,0.5,0.5])
    fit = curve_fit(f_2_fit_alt, t_array, P_array, p0,lower=[0.0,0.0,0.0])
    return fit.param
end



function find_dwell_minimas(ϵ_1,ϵ_2)
    x_array = Vector(-20:0.01:20)
    H_p0_cut(x) = H_cl(x,0,ϵ_1,ϵ_2)
    H_p0_cut_derivative(x) = gradient(H_p0_cut, x)[1]
    all_zeros = find_zeros(H_p0_cut_derivative,x_array[1],x_array[end])
    if length(all_zeros)==3
        minimas = sort(all_zeros)
        return minimas[1], minimas[3]
    else 
        return nothing, nothing
    end
end


##############
# L_mat code #
##############
job_index = ARGS[1] 
n = parse(Int, job_index) # To program!
display("Job correctly initalised, n=$n")

N = 50 # scales as N^2

# Define H_eff parameter space, in units of K
K = 1
Δ = -0*K
ϵ_1_array = Vector(range(0,15,length=200)).*K
ϵ_2_array = Vector(range(0,12,length=199)).*K
n = 128
ϵ_2 = ϵ_2_array[n]

# Define collapse ops 
κ_1 = 0.025 #in units of K
n_th = 0.05
κ_p, κ_m = κ_1*n_th, κ_1*(1 + n_th) 

# Define X op
X_op = (a(N) + a(N)')/√(2)

# EXPONENTIAL time array
t_array = exp.( Vector(range(0,log(2e5),length=100)))
pushfirst!(t_array,0)

display("constructing C_super_op...")
C_ops = ([ √(κ_p) * a(N)', √(κ_m) * a(N) ])
C_ops_super = C_super_ops(C_ops)

# Generate H_eff data within parameter space
sim_time = round(length(ϵ_1_array) * 2.177 / 3600,sigdigits=2 )
display("For N=35, this simulation will take: $sim_time h.")

# start on the left.  well
ψ_0 = coherent_state(N,-√(2*ϵ_2),0)
ρ_0 = ψ_0*ψ_0'
ρ_0_vec = reshape(ρ_0, N^2)

# Define data form
X_data_array_old = zeros( length(ϵ_1_array), length(t_array) ) 
X_data_array_new = zeros( length(ϵ_1_array), length(t_array) ) 
pbar = ProgressBar(total = length(ϵ_1_array)*length(t_array))
for (j,ϵ_1) in enumerate(ϵ_1_array)
    H_temp_super = H_super_op( -H_eff(N,Δ,K,ϵ_1,ϵ_2) )
    L_temp = H_temp_super + C_ops_super
    λ_n, ϕ_mat = eigen(L_temp) 
    ϕ_mat_inv = inv(ϕ_mat)

    for (k,t) in enumerate(t_array)
        exp_tL = ϕ_mat * diagm( exp.(t*λ_n) ) * ϕ_mat_inv
        ρ_t_vec = exp_tL * ρ_0_vec
        ρ_t_mat = reshape(ρ_t_vec,(N,N))

        P_temp = ( √(2*ϵ_2) + real( tr(X_op*ρ_t_mat) ) )/ (2*√(2*ϵ_2))
        if P_temp > 1
            P_temp = 1
        elseif P_temp < 0 
            P_temp = 0 
        end          
        X_data_array_old[j,k] = P_temp

        min_L, min_R = find_dwell_minimas(ϵ_1,ϵ_2)
        P_up = ( real( tr(X_op*ρ_t_mat) ) - min_L )/ (min_R-min_L)
        if P_up > 1
            P_up = 1
        elseif P_up < 0 
            P_up = 0 
        end          
        X_data_array_new[j,k] = P_up

        update(pbar)
    end
end


npzwrite("data_gen/Bosonic_basis/H_eff_Lindbladian/cluster_prototype_prob_comparison_data_new.npz", X_data_array_new)
npzwrite("data_gen/Bosonic_basis/H_eff_Lindbladian/cluster_prototype_prob_comparison_data_old.npz", X_data_array_old)

fit_array_new = zeros( length(ϵ_1_array),2 )
fit_array_old = zeros( length(ϵ_1_array),2 )
for (j,ϵ_1) in enumerate(ϵ_1_array)
    fit_array_new[j,:] .= get_fit_params(X_data_array_new[j,:], t_array)
    fit_array_old[j,:] .= get_fit_params(X_data_array_old[j,:], t_array)
end 

fit_array_new_alt = zeros( length(ϵ_1_array),3 )
fit_array_old_alt = zeros( length(ϵ_1_array),3 )
for (j,ϵ_1) in enumerate(ϵ_1_array)
    fit_array_new_alt[j,:] .= get_fit_params_alt(X_data_array_new[j,:], t_array)
    fit_array_old_alt[j,:] .= get_fit_params_alt(X_data_array_old[j,:], t_array)
end 

npzwrite("data_gen/Bosonic_basis/H_eff_Lindbladian/cluster_prototype_prob_comparison_fit_new.npz", fit_array_new)
npzwrite("data_gen/Bosonic_basis/H_eff_Lindbladian/cluster_prototype_prob_comparison_fit_old.npz", fit_array_old)


display("Saved! Exiting...")



top = heatmap(ϵ_1_array,t_array,X_data_array_new',c=cgrad(:curl, rev = false),size = (width_px,width_px*0.5), dpi=600,right_margin = 2Plots.mm,xtickfontsize=8,ytickfontsize=8,tickdirection=:out,fontfamilys = "Times New Roman",tickfontfamily = "Times New Roman",axislabelfontfamily="Times New Roman",ylabelfontsize=8,xlabelfontsize=8,widen=false,ylab=L"t \ (\mu s)",xlab=L"ϵ_1/K",colorbartitle=L"P_\textrm{down}",ylim=(0,30000),title="new")
plot!(ϵ_1_array,fit_array_new,lw=1.5,c=:black,label=false)
plot!(ϵ_1_array,fit_array_new,lw=0.5,c=:white,ls=:dash,label=false)

bot = heatmap(ϵ_1_array,t_array,X_data_array_old',c=cgrad(:curl, rev = false),size = (width_px,width_px*0.5), dpi=600,right_margin = 2Plots.mm,xtickfontsize=8,ytickfontsize=8,tickdirection=:out,fontfamilys = "Times New Roman",tickfontfamily = "Times New Roman",axislabelfontfamily="Times New Roman",ylabelfontsize=8,xlabelfontsize=8,widen=false,ylab=L"t \ (\mu s)",xlab=L"ϵ_1/K",colorbartitle=L"P_\textrm{down}",ylim=(0,30000),title="old")
plot!(ϵ_1_array,fit_array_old,lw=1.5,c=:black,label=false)
plot!(ϵ_1_array,fit_array_old,lw=0.5,c=:white,ls=:dash,label=false)

plot(top, bot, layout=l,size=(width_px,width_px*1.2))


plot(t_array,X_data_array_new'[:,1])


chosen_n = 32
fit_2_plot = zeros(length(t_array))
for (n,t) in enumerate(t_array)
    fit_2_plot[n] = f_2_fit(t,fit_array_new[chosen_n,:])
end
plot(t_array,X_data_array_new[chosen_n,:])
plot!(t_array,fit_2_plot)

chosen_n = 32
fit_2_plot = zeros(length(t_array))
for (n,t) in enumerate(t_array)
    fit_2_plot[n] = f_2_fit(t,fit_array_old[chosen_n,:])
end
plot(t_array,X_data_array_old[chosen_n,:])
plot!(t_array,fit_2_plot)


chosen_n = 32
fit_2_plot = zeros(length(t_array))
for (n,t) in enumerate(t_array)
    fit_2_plot[n] = f_2_fit_alt(t,fit_array_new_alt[chosen_n,:])
end
plot(t_array,X_data_array_new[chosen_n,:])
plot!(t_array,fit_2_plot)

chosen_n = 32
fit_2_plot = zeros(length(t_array))
for (n,t) in enumerate(t_array)
    fit_2_plot[n] = f_2_fit_alt(t,fit_array_old_alt[chosen_n,:])
end
plot!(t_array,X_data_array_old[chosen_n,:])
plot!(t_array,fit_2_plot)