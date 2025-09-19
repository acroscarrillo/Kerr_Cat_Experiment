# !/usr/bin/env julia

# @btime eigen(H_super_op( -H_eff(35,0,1,0,12) ) + C_ops_super) 2.177 s (54 allocations: 128.60 MiB)

using LinearAlgebra
using NPZ
using Roots
using Flux
using LsqFit

⦼(A,B) = kron(A,B)

function a(N::Int)
    temp = zeros(Float64, (N, N))
    for n in 1:N-1
        temp[n,n+1] = sqrt(n)
    end
    return temp
end

function H_chem(N,k_1,k_2,k_4)
    A = a(N) # annahilation op. up to dim N
    x = (A + A')/√(2)
    p = -im*(A - A')/√(2)
    return 0.5*p^2 + k_1*x - k_2*x^2 + k_4*x^4
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
    return exp.(-t/τ) .+ O*(1 .- exp.(-t/τ))
end

function get_fit_params(P_array, t_array, p0=[100.0,0.5])
    fit = curve_fit(f_2_fit, t_array, P_array, p0,lower=[100.0,0.0],upper=[1000.0,1.0])
    return fit.param
end

function select_ψ_0(N,k_1,k_2,k_4)
    X_op = (a(N) + a(N)')/√(2)
    v_n = eigen(H_chem(N,k_1,k_2,k_4)).vectors
    for n=1:N
        v = v_n[:,n]
        if real(v'*X_op*v) > 0
            return v
        end
    end
end


function find_upper_well_x(k_1,k_2)
    H_p0_cut(x) = k_1*x - k_2*x^2 + 1*x^4
    H_p0_cut_derivative(x) = gradient(H_p0_cut, x)[1]
    all_zeros = find_zeros(H_p0_cut_derivative,-20,20)
    if length(all_zeros)==3
        return sort(all_zeros)[3]
    else 
        return √(k_2/2)
    end
end

##############
# L_mat code #
##############
job_index = ARGS[1] 
n = parse(Int, job_index) # To program!
display("Job correctly initalised, n=$n")

# diagonilising 2500x2500, N=50, takes 9.466 s (32 allocations: 291.22 MiB)
N = 30 # scales as N^2

# Define H_eff parameter space, in units of K
k_4 = 1
Δ = -0
k_1_array = Vector(range(0,10,length=50))
k_2_array = Vector(range(8,13,length=49))
k_2 = k_2_array[n]

# Define collapse ops 
κ_1 = 0.025 #in units of K
n_th = 0.05
κ_p, κ_m = κ_1*n_th, κ_1*(1 + n_th) 

# EXPONENTIAL time array
t_array = exp.( Vector(range(0,log(500),length=100)))
pushfirst!(t_array,0)


# Define data form
X_data_array = zeros( length(k_1_array), length(t_array) ) 
for (j,k_1) in enumerate(k_1_array)
    # get eigenspace
    E_n, ϕ_n = eigen(H_chem(N,k_1 ,k_2,1))
    H_temp = diagm(E_n)

    # define initial state
    ψ_0 = ϕ_n' * select_ψ_0(N,k_1,k_2,1)
    ρ_0_vec = reshape(ψ_0*ψ_0', N^2)

    # Define X op
    X_op = ϕ_n' * (a(N) + a(N)')/√(2) * ϕ_n

    ψ_0 = ϕ_n' * select_ψ_0(N,12.6 ,k_2,1)
    ρ_0_vec = reshape(ψ_0*ψ_0', N^2)

    # super ops
    H_ops_super = H_super_op( H_temp )
    C_ops =  ([ √(κ_p) * ϕ_n'* a(N)'*ϕ_n, √(κ_m) * ϕ_n'*a(N)*ϕ_n ]) 
    C_ops_super = C_super_ops( C_ops )
    L_temp = H_ops_super + C_ops_super

    λ_n, ϕ_mat = eigen(L_temp) 
    ϕ_mat_inv = inv(ϕ_mat)  

    upper_well_pos = find_upper_well_x(k_1,k_2)
    for (k,t) in enumerate(t_array)
        exp_tL = ϕ_mat * diagm( exp.(t*λ_n) ) * ϕ_mat_inv
        ρ_t_vec = exp_tL * ρ_0_vec
        ρ_t_mat = reshape(ρ_t_vec,(N,N))

        P_temp = ( upper_well_pos + real( tr(X_op*ρ_t_mat) ) )/ (2*upper_well_pos)
        if P_temp > 1
            P_temp = 1
        elseif P_temp < 0 
            P_temp = 0 
        end          
        X_data_array[j,k] = P_temp
    end
    display("Loop: $j of 200")
end


fit_array = zeros( length(k_1_array),2 )
for (j,k_1) in enumerate(k_1_array)
    fit_array[j,:] .= get_fit_params(X_data_array[j,95:end], t_array[95:end])
end 

# display("Saving...")
# npzwrite("/home/ucapacr/Scratch/Yale_project/data_gen/Lindbladian_dynamics/chem_data/X_data_k_2_$k_2.npz", X_data_array)
# npzwrite("/home/ucapacr/Scratch/Yale_project/data_gen/Lindbladian_dynamics/chem_data/fit_data_k_2_$k_2.npz", fit_array)
# display("Saved! Exiting...")










N = 40
ϕ_n = eigen(H_chem(N,12.6 ,k_2,1)).vectors

ψ_0 = ϕ_n' * select_ψ_0(N,12.6 ,k_2,1)
ρ_0 = ψ_0*ψ_0'
ρ_0_vec = reshape(ρ_0, N^2)


H_ops_super = H_super_op( diagm( eigen(H_chem(N,12.6 ,k_2,1)).values) )
C_ops =  ([ √(κ_p) * ϕ_n'* a(N)'*ϕ_n, √(κ_m) * ϕ_n'*a(N)*ϕ_n ]) 
C_ops_super = C_super_ops( C_ops )

L_temp = H_ops_super + C_ops_super
λ_n, ϕ_mat = eigen(L_temp) 
ϕ_mat_inv = inv(ϕ_mat)

t_array = exp.( Vector(range(0,log(500),length=100)))
pushfirst!(t_array,0)


@gif for (k,t) in ProgressBar(enumerate(t_array))
    exp_tL = ϕ_mat * diagm( exp.(t*λ_n) ) * ϕ_mat_inv
    ρ_t_vec = exp_tL * ρ_0_vec
    ρ_t_mat = reshape(ρ_t_vec,(N,N))

    heatmap(wigner_func_ρ( ρ_t_mat,6,6 )  )
end