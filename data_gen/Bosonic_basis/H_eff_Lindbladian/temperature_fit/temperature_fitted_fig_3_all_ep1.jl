# !/usr/bin/env julia

# @btime eigen(H_super_op( -H_eff(35,0,1,0,12) ) + C_ops_super) 2.177 s (54 allocations: 128.60 MiB)

using LinearAlgebra
using NPZ
using Roots
using Flux
using LsqFit
using Optim
using Plots
using ProgressBars
using LaTeXStrings


global ħ = 1.054e-34  # Js
global k_b = 1.38e-23 # J/K

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
    τ, A, O = p[1], p[2], p[3]
    return A*exp.(-t/τ) .+ O
end

function get_fit_params(P_array, t_array, p0=[10.0,1,0.5])
    fit = curve_fit(f_2_fit, t_array, P_array, p0,lower=[1,-1,0.0])
    return fit.param
end

function evolve!(ϵ_2, X_op, t_array, ϕ_mat, ϕ_mat_inv, λ_n, ρ_0_vec, ρ_t_vec)
    # full operation is:
    # ρ_t_vec = (ϕ_mat * diagm( exp.(t*λ_n) ) * ϕ_mat_inv) * ρ_0_vec
    #           |_____|  |__________________|   |__________________|
    #             ϕ_mat        exp_λt                    v0           (code naming)
    #                        |______________________________|
    #                                       w                         (code naming)
    # and then 
    # X = tr(X_op * ρ_t_mat)
    #   = vec(Xᵀ) · vec(ρ) = xvecT · ρ_t_vec                          (code naming)

    # So, precompute (saves one matvec per time):
    # v0 = ϕ_mat_inv * ρ_0_vec once 
    v0 = similar(ρ_0_vec)
    mul!(v0, ϕ_mat_inv, ρ_0_vec)

    # Preallocate memory for the loop (O(d))
    w       = similar(ρ_0_vec)       # w = exp(tλ) .* v0
    exp_λt  = similar(ρ_0_vec)       # store exp(tλ)
    xvecT   = vec(transpose(X_op))   # vec(Xᵀ), shares storage (strided)

    c0 = sqrt(2*ϵ_2)
    c  = inv(2*sqrt(2*ϵ_2))

    X_vec = similar(t_array)
    @inbounds for (k, t) in enumerate(t_array)
        # exp_λt .= exp.(t .* λ_n)       # alloc-free
        @. exp_λt = exp(t * λ_n)         # same, fused

        # w = exp_λt .* v0               # elementwise scale, no matrix broadcast
        w .= v0
        w .*= exp_λt

        # ρ_t_vec = Φ * w                # ONE matvec per time step
        mul!(ρ_t_vec, ϕ_mat, w)

        # s = tr(X_op * ρ_t_mat) without forming the product:
        # tr(Xρ) = vec(Xᵀ)·vec(ρ); for Complex use unconjugated dot
        s = dot(xvecT, ρ_t_vec)

        P = (c0 + real(s)) * c
        X_vec[k] = ifelse(P < 0.0, 0.0, ifelse(P > 1.0, 1.0, P))
    end
    return X_vec
end

function temp_nth(n_th,ω)
    beta =  log(1+ 1/n_th)/(ħ*ω)
    return 1/(k_b*beta)
end

################################################
# Experimental timescale T1 vs ϵ_1 calculation #
################################################

data_exp = npzread("data_gen/Bosonic_basis/H_eff_Lindbladian/temperature_fit/eps1KvTime07.npz")

t_array_exp = data_exp["time"]
P_data_exp = data_exp["z_data"]
ϵ_1_array = data_exp["eps1K"]*1.05 # THIS IS KERR CORRECTED!!!

T1_exp_array = zeros(length(ϵ_1_array))
offset_exp_array = zeros(length(ϵ_1_array))
amplitude_exp_array = zeros(length(ϵ_1_array))
for n=1:length(ϵ_1_array)
    p_temp = P_data_exp[:,n] 
    try #handle NaN
        τ, A, O =  get_fit_params(p_temp, t_array_exp)
        T1_exp_array[n] = τ
        amplitude_exp_array[n] = A
        offset_exp_array[n] = O
    catch
        T1_exp_array[n] = T1_exp_array[n-1]
    end
end

# I want to convert t(\mu s) to unitless tK. So I convert t to seconds and then I multiply by kerr to cancel the units. Now here’s the tricky bit, K/2pi = 528 kHz. Does that mean we want to do t -> tK = t x(10^-6) x 528 x 10^3 x 2pi 

T1_exp_array_unitless = T1_exp_array .* (1e-6 * 528e3 * 2 * π) # unitless

##############
# L_mat code #
##############

N = 35 # scales as N^2

# Define H_eff parameter space, in units of K
K = 1
Δ = -0*K
ϵ_2 = 7.7

# EXPONENTIAL time array
t_array = exp.( Vector(range(0,log(2e5),length=100)))
pushfirst!(t_array,0)

# Define X op
X_op = (a(N) + a(N)')/√(2)

# Define bath params
κ_1 = 0.025 #in units of K
# n_th =  To FIT!


# start on the left  well
ψ_0 = coherent_state(N,-√(2*ϵ_2),0)
ρ_0 = ψ_0*ψ_0'
ρ_0_vec = reshape(ρ_0, N^2)

# Prealocate memory
ρ_t_vec = zeros(ComplexF64, N^2)

# -------------------------------
# Fit global n_th by minimizing SSE over all ϵ_1
# -------------------------------

n_th_min, n_th_max = 0.01, 1.0

# Precompute H part for every ϵ_1 once
H_list = [ H_super_op( -H_eff(N,Δ,K,ϵ_1,ϵ_2) ) for ϵ_1 in ϵ_1_array ]
A_op = a(N)  # reuse in C_ops

function sse_global(n_th)
    if (n_th < n_th_min) || (n_th > n_th_max)
        return 1e12
    end
    κ_p = κ_1*n_th
    κ_m = κ_1*(1 + n_th)
    C_ops_super = C_super_ops([ √(κ_p) * A_op', √(κ_m) * A_op ])

    sse = 0.0
    for (j, Hs) in enumerate(H_list)
        L_temp = Hs + C_ops_super
        λ_n, ϕ_mat = eigen(L_temp)
        ϕ_mat_inv = inv(ϕ_mat)
        X_vec = evolve!(ϵ_2, X_op, t_array, ϕ_mat, ϕ_mat_inv, λ_n, ρ_0_vec, ρ_t_vec)
        τ, A, O = get_fit_params(X_vec, t_array)
        sse += (τ - T1_exp_array_unitless[j])^2
    end
    return sse
end

# Cap to at most 10 objective evaluations (⇒ ≤10 eigen calls per ϵ₁)
n_eval   = Ref(0)
best_sse = Ref(Inf)
obj(n_th) = begin
    if n_eval[] >= 10
        return best_sse[]
    end
    n_eval[] += 1
    val = sse_global(n_th)
    if val < best_sse[]
        best_sse[] = val
    end
    return val
end

# 3364.384655 seconds (3.25 M allocations: 219.009 GiB, 0.11% gc time, 0.00% compilation time: 45% of which was recompilation)
res = @time optimize(obj, n_th_min, n_th_max, Brent(); show_trace=true, iterations=10)
n_th_star = Optim.minimizer(res)

# Evaluate τ_array at fitted n_th_star (for plotting)
κ_p = κ_1*n_th_star
κ_m = κ_1*(1 + n_th_star)
C_ops_super = C_super_ops([ √(κ_p) * A_op', √(κ_m) * A_op ])

τ_array  = zeros(length(ϵ_1_array))
for (j, Hs) in enumerate(H_list)
    L_temp = Hs + C_ops_super
    λ_n, ϕ_mat = eigen(L_temp)
    ϕ_mat_inv = inv(ϕ_mat)
    X_vec = evolve!(ϵ_2, X_op, t_array, ϕ_mat, ϕ_mat_inv, λ_n, ρ_0_vec, ρ_t_vec)
    τ, A, O = get_fit_params(X_vec, t_array)
    τ_array[j] = τ
end

plot(size = (width_px,width_px*0.6),ϵ_1_array, τ_array, ylab = L"\tau K", xlab= L"\epsilon_1/K", label="fitted τ(n_th)", grid=false,lw=3, title = L"n_{th}="*"$round(n_th_star), "*L"T="*"$round(1e3 * temp_nth(n_th_star,2*π*6e9)) (mK).")
plot!(ϵ_1_array, T1_exp_array_unitless, label="experimental τ", ls=:dash,lw=3)

savefig("data_gen/Bosonic_basis/H_eff_Lindbladian/temperature_fit/fitted_temperature_all_ep1.png")