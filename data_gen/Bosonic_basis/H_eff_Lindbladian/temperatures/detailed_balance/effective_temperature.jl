# !/usr/bin/env julia

# @btime eigen(H_super_op( -H_eff(35,0,1,0,12) ) + C_ops_super) 2.177 s (54 allocations: 128.60 MiB)

using LinearAlgebra
using NPZ
using Roots
using Flux
using LsqFit
using StatsBase
using ProgressBars
using LaTeXStrings
using Plots

⦼(A,B) = kron(A,B)

global ħ = 1.054e-34  # Js
global k_b = 1.38e-23 # J/K

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

function get_fit_params(P_array, t_array, p0=[10.0,0.5])
    fit = curve_fit(f_2_fit, t_array, P_array, p0,lower=[0.0,0.0])
    return fit.param
end

function get_beta(ρ,H)
    p_n, ϕ_n_mat = eigen(ρ)
    temp_array = []
    for n=1:length(E_n)
        for m=1:length(E_n)
            if n>m
                E_n[m] =  ϕ_n_mat[:,m]' * H * ϕ_n_mat[:,m]
                E_n[n] =  ϕ_n_mat[:,n]' * H * ϕ_n_mat[:,n]
                push!(temp_array, log(p_n[n]/p_n[m])/(E_n[m]-E_n[n])   )
            end
        end
    end
    return mean(temp_array),std(temp_array),temp_array[1]
end


function get_beta_array(ρ,H,N_thresh)
    p_array, ϕ_n_mat = eigen(ρ)
    E_array, ψ_n_mat = eigen(H)
    p_indx = argmax.(eachcol(norm.(ϕ_n_mat'*ψ_n_mat))) # match ϕ_n's with ψ_n's
    beta_array = []
    for n=2:N_thresh
        m = n-1
        p_n, p_m = norm(p_array[p_indx[n]]), norm(p_array[p_indx[m]])
        if p_n>1e-6 && p_m>1e-6
            beta = log( real(p_n)/real(p_m) )/(E_array[m]-E_array[n])
            push!(beta_array, real(beta))
        end
    end
    return mean(beta_array), std(beta_array)
end

function tr_dist(A,B)
    λ_n = eigvals( A - B )
    return 0.5*sum(abs.(λ_n))
end

function plank_law(T,ω)
    β = 1/(k_b*T)
    return 1 / ( exp(β*ω*ħ) - 1  )  
end

function temp_nth(n_th,ω)
    beta =  log(1+ 1/n_th)/(ħ*ω)
    return 1/(k_b*beta)
end

ϵ_1_reso(ϵ_2,n,K_corr_f=1) = √(ϵ_2*K_corr_f)*n/K_corr_f


##############
# L_mat code #
##############

N = 60 # scales as N^2
N_thresh = 25

# Define H_eff parameter space, in units of K
K = 1
Δ = -0*K
ϵ_1_array = Vector(range(0,15,length=300))
ϵ_2 = 7.7 # Max's value

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

# start on the left.  well
# ψ_0 = coherent_state(N,-√(2*ϵ_2),0)
ψ_0 = coherent_state(N,0,0)
ρ_0 = ψ_0*ψ_0'
ρ_0_vec = reshape(ρ_0, N^2)

# Define data form
# beta_array_temp = zeros( length(ϵ_1_array),Int(0.5*(N_thresh^2-N_thresh)))
beta_array_mean = zeros( length(ϵ_1_array))
beta_array_std = zeros( length(ϵ_1_array))

trace_dist_array_12 = ones( length(ϵ_1_array))
trace_dist_array_mean = ones( length(ϵ_1_array))
P_X_data_array = zeros(length(ϵ_1_array))
for (j,ϵ_1) in ProgressBar(enumerate(ϵ_1_array))
    H_temp =  -H_eff(N,Δ,K,ϵ_1,ϵ_2) 
    H_temp_super = H_super_op( H_temp )
    L_temp = H_temp_super + C_ops_super
    λ_n, ϕ_mat = eigen(L_temp) 
    ϕ_mat_inv = inv(ϕ_mat)

    exp_infL = ϕ_mat * diagm( exp.(1e7*λ_n) ) * ϕ_mat_inv
    ρ_ss_vec = exp_infL * ρ_0_vec
    ρ_ss_mat = reshape(ρ_ss_vec,(N,N))

    # beta_temp = get_beta(ρ_ss_mat,H_temp)
    temp = get_beta_array(ρ_ss_mat,H_temp,N_thresh)
    beta_array_mean[j] = temp[1]
    beta_array_std[j] = temp[2]

    ρ_th = exp(-temp[1] * H_temp) / tr(exp(-temp[1] * H_temp))
    if !(any(x -> isnan(x) || isinf(x), ρ_th))
        trace_dist_array_mean[j] = tr_dist( ρ_ss_mat/norm(tr(ρ_ss_mat)), ρ_th)
    end

    P_temp = ( √(2*ϵ_2) + real( tr(X_op*ρ_ss_mat) ) )/ (2*√(2*ϵ_2))
    if P_temp > 1
        P_temp = 1
    elseif P_temp < 0 
        P_temp = 0 
    end          
    P_X_data_array[j] = P_temp

    # trace_dist_array_mean[j] = tr_dist( ρ_ss_mat/norm(tr(ρ_ss_mat)), exp(-beta_temp[3] * H_temp)/tr(exp(-beta_temp[1] * H_temp)) )
end
# npzwrite("data_gen/Bosonic_basis/H_eff_Lindbladian/temperatures/detailed_balance/beta_array_mean_N_$(N)_N_thresh_$N_thresh.npz", beta_array_mean)
# npzwrite("data_gen/Bosonic_basis/H_eff_Lindbladian/temperatures/detailed_balance/beta_array_std_N_$(N)_N_thresh_$N_thresh.npz", beta_array_std)
# npzwrite("data_gen/Bosonic_basis/H_eff_Lindbladian/temperatures/detailed_balance/trace_dist_N_$(N)_N_thresh_$N_thresh.npz", trace_dist_array_mean)
# npzwrite("data_gen/Bosonic_basis/H_eff_Lindbladian/temperatures/detailed_balance/ep_1_array_N_$(N)_N_thresh_$N_thresh.npz", ϵ_1_array)



beta_array_mean = npzread("data_gen/Bosonic_basis/H_eff_Lindbladian/temperatures/detailed_balance/beta_array_mean_N_$(N)_N_thresh_$N_thresh.npz")
beta_array_std = npzread("data_gen/Bosonic_basis/H_eff_Lindbladian/temperatures/detailed_balance/beta_array_std_N_$(N)_N_thresh_$N_thresh.npz")
trace_dist_array_mean = npzread("data_gen/Bosonic_basis/H_eff_Lindbladian/temperatures/detailed_balance/trace_dist_N_$(N)_N_thresh_$N_thresh.npz")
ϵ_1_array = npzread("data_gen/Bosonic_basis/H_eff_Lindbladian/temperatures/detailed_balance/ep_1_array_N_$(N)_N_thresh_$N_thresh.npz")

ϵ_1_array = ϵ_1_array[2:end]
beta_array_mean = beta_array_mean[2:end]
beta_array_std = beta_array_std[2:end]
trace_dist_array_mean = trace_dist_array_mean[2:end]

top = plot(ϵ_1_array,trace_dist_array_mean,ylabel=L"\mathrm{TrDist}\left(\rho_\mathrm{ss},\rho_\mathrm{th}\right)",lw=1.5,c=:black,ylim=(1e-3,1),xlabel=L"ϵ_1/K",yscale=:log,legend=false;plot_kwargs...)
vline!([ϵ_1_reso.(ϵ_2,[1,2,3],1)...],c=:orange,lw=2,ls=:dash,alpha=0.8)
vline!([ϵ_1_reso.(ϵ_2,[4],1)...],c=:purple,lw=2,ls=:dash,alpha=0.8)


mid = scatter(ϵ_1_array, beta_array_mean,yerr = beta_array_std,legend=false,ylim=(-0,0.5),c=:red,ms=1,ylabel=L"\beta_\mathrm{avg}", xlabel=L"ϵ_1/K",markerstrokewidth=1)
vline!([ϵ_1_reso.(ϵ_2,[1,2,3],1)...],c=:orange,lw=2,ls=:dash)
vline!([ϵ_1_reso.(ϵ_2,[4],1)...],c=:purple,lw=2,ls=:dash,alpha=0.8)
scatter!(ϵ_1_array, beta_array_mean,c=:red,ms=1.5,markerstrokewidth=0)


bot = plot(ϵ_1_array, beta_array_std,legend=false,ylim=(-0.0,1.1),c=:black,ms=2,ylabel=L"\sigma(\beta)", xlabel=L"ϵ_1/K",lw=1.5)
vline!([ϵ_1_reso.(ϵ_2,[1,2,3],1)...],c=:orange,lw=2,ls=:dash,alpha=0.8)
vline!([ϵ_1_reso.(ϵ_2,[4],1)...],c=:purple,lw=2,ls=:dash,alpha=0.8)


l = @layout [top; mid; bot]

width_px=340.39020340390204
plot(top,mid,bot,layout =l,size = (width_px,width_px*1.5),xtickfontsize=8,ytickfontsize=8,dpi=600,widen=false,ylabelfontsize=8,xlabelfontsize=8,tickdirection=:out,fontfamilys = "Times New Roman",tickfontfamily = "Times New Roman",axislabelfontfamily="Times New Roman",grid=false,plot_titlefontsize=8)

savefig("data_gen/Bosonic_basis/H_eff_Lindbladian/temperatures/detailed_balance/betas_detailed_b_and_resos_N_$(N)_N_thresh_$(N_thresh).png")
savefig("data_gen/Bosonic_basis/H_eff_Lindbladian/temperatures/detailed_balance/betas_detailed_b_and_resos_N_$(N)_N_thresh_$(N_thresh).pdf")