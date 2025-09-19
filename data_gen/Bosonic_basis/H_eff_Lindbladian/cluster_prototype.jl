# !/usr/bin/env julia

# @btime eigen(H_super_op( -H_eff(35,0,1,0,12) ) + C_ops_super) 2.177 s (54 allocations: 128.60 MiB)

using LinearAlgebra
using NPZ
using Roots
using Flux
using LsqFit

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

function get_fit_params(P_array, t_array, p0=[10.0,0.5])
    fit = curve_fit(f_2_fit, t_array, P_array, p0,lower=[0.0,0.0])
    return fit.param
end

##############
# L_mat code #
##############
job_index = ARGS[1] 
n = parse(Int, job_index) # To program!
display("Job correctly initalised, n=$n")

N = 35 # scales as N^2

# Define H_eff parameter space, in units of K
K = 1
Δ = -0*K
ϵ_1_array = Vector(range(0,15,length=200)).*K
ϵ_2_array = Vector(range(0,12,length=199)).*K
# n = 128
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
X_data_array = zeros( length(ϵ_1_array), length(t_array) ) 
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
        X_data_array[j,k] = P_temp
    end
    display("Loop: $j of 200")
end


fit_array = zeros( length(ϵ_1_array),2 )
for (j,ϵ_1) in enumerate(ϵ_1_array)
    fit_array[j,:] .= get_fit_params(X_data_array[j,:], t_array)
end 

display("Saving...")
npzwrite("/home/ucapacr/Scratch/Yale_project/data_gen/Lindbladian/dwell_data/X_data_ep_1_$ϵ_1.npz", X_data_array)
npzwrite("/home/ucapacr/Scratch/Yale_project/data_gen/Lindbladian/dwell_data/fit_data_ep_1_$ϵ_1.npz", fit_array)
display("Saved! Exiting...")