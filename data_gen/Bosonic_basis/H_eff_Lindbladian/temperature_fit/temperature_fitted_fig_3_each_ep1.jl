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
ϵ_1_array = data_exp["eps1K"]*1.05

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

# Define collapse ops 
κ_1 = 0.025 #in units of K
n_th = 0.172
κ_p, κ_m = κ_1*n_th, κ_1*(1 + n_th) 

display("constructing C_super_op...")
C_ops = ([ √(κ_p) * a(N)', √(κ_m) * a(N) ])
C_ops_super = C_super_ops(C_ops)

# start on the left  well
ψ_0 = coherent_state(N,-√(2*ϵ_2),0)
ρ_0 = ψ_0*ψ_0'
ρ_0_vec = reshape(ρ_0, N^2)

# Generate H_eff data within parameter space
sim_time = round(length(ϵ_1_array) * 2.177 / 3600,sigdigits=2 )
display("For N=35, this simulation will take: $sim_time h.")

# Prealocate memory
ρ_t_vec = zeros(ComplexF64, N^2)

# Define data form
X_data_array = zeros( length(ϵ_1_array), length(t_array) ) 
@time for (j,ϵ_1) in ProgressBar(enumerate(ϵ_1_array))
    H_temp_super = H_super_op( -H_eff(N,Δ,K,ϵ_1,ϵ_2) ) #  1.376 s (101 allo.: 128.68 MiB)
    L_temp = H_temp_super + C_ops_super
    λ_n, ϕ_mat = eigen(L_temp) 
    ϕ_mat_inv = inv(ϕ_mat)

    X_data_array[j,:] = evolve!(ϵ_2, X_op, t_array, ϕ_mat, ϕ_mat_inv, λ_n, ρ_0_vec, ρ_t_vec)
end

fit_array = zeros( length(ϵ_1_array), 3) # τ, A, O
for (j,ϵ_1) in enumerate(ϵ_1_array)
    fit_array[j,:] = get_fit_params(X_data_array[j,:], t_array) # τ, A, O
end 

τ_array = fit_array[:,1]

plot(ϵ_1_array, τ_array / (1e-6 * 528e3 * 2 * π) , ylab = L"\tau K", xlab= L"\epsilon_1/K",legend=false, grid=false)
plot!(ϵ_1_array, T1_exp_array_unitless / (1e-6 * 528e3 * 2 * π) )

# -------------------------------
# Fit n_th(ϵ_1) by minimizing SSE
# -------------------------------

n_th_min, n_th_max = 0.001, 1
n_th_fit = zeros(length(ϵ_1_array))
τ_array  = zeros(length(ϵ_1_array))

@time for (j,ϵ_1) in ProgressBar(enumerate(ϵ_1_array))
    # Precompute H part (depends only on ϵ_1)
    H_temp_super = H_super_op( -H_eff(N,Δ,K,ϵ_1,ϵ_2) )

    # Per-ϵ_1 counters and best-so-far
    n_eval = 0
    best_sse = Inf
    best_τ = 0.0
    best_nth = 0.05

    function sse(n_th)
        if (n_th < n_th_min) || (n_th > n_th_max)
            return 1e12
        end
        if n_eval >= 50
            return best_sse
        end
        n_eval += 1

        κ_p = κ_1*n_th
        κ_m = κ_1*(1 + n_th)
        C_ops = ([ √(κ_p) * a(N)', √(κ_m) * a(N) ])
        C_ops_super = C_super_ops(C_ops)

        L_temp = H_temp_super + C_ops_super
        λ_n, ϕ_mat = eigen(L_temp)
        ϕ_mat_inv = inv(ϕ_mat)

        X_vec = evolve!(ϵ_2, X_op, t_array, ϕ_mat, ϕ_mat_inv, λ_n, ρ_0_vec, ρ_t_vec)
        τ, A, O = get_fit_params(X_vec, t_array)

        sse_val = (τ - T1_exp_array_unitless[j])^2
        if sse_val < best_sse
            best_sse = sse_val
            best_τ = τ
            best_nth = n_th
        end
        return sse_val
    end

    optimize(sse, n_th_min, n_th_max, Brent(); iterations=50, show_trace=false)
    n_th_fit[j] = best_nth
    τ_array[j]  = best_τ
end

l = @layout [a ; b]


width_px=340.39020340390204

top_plot = plot(size = (width_px,width_px*0.6),ϵ_1_array, τ_array, ylab = L"\tau K", xlab= L"\epsilon_1/K", label="fitted τ(n_th)", grid=false,lw=3)
plot!(ϵ_1_array, T1_exp_array_unitless, label="experimental τ", ls=:dash,lw=3)

bottom_plot = plot(ϵ_1_array, 1e3 * temp_nth.(n_th_fit,2*π*6e9), ylab = L"T \quad (mK)", xlab= L"\epsilon_1/K",lw=3, legend=false,color=theme_palette(:default)[1],yguidefontcolor=theme_palette(:default)[1])
plot!(twinx(), ϵ_1_array, n_th_fit, yaxis = L"n_{th}",c=theme_palette(:default)[2], ls=:dash, lw=3,yguidefontcolor=theme_palette(:default)[2],legend=false)

plot(top_plot, bottom_plot; layout = l, plot_kwargs...)

savefig("data_gen/Bosonic_basis/H_eff_Lindbladian/temperature_fit/fitted_temperature.png")
# scatter(x_data,y_data,c=:black,,xtickfontsize=8,ytickfontsize=8,dpi=600,widen=false,ylabelfontsize=8,xlabelfontsize=8,tickdirection=:out,fontfamilys = "Times New Roman",tickfontfamily = "Times New Roman",axislabelfontfamily="Times New Roman",grid=false,legend=:bottomright,ms=3,ylab="Readout signal (u.n.n)",xlab=L"\left( \omega_{\mathrm{pr}} - \omega_{\mathrm{ge}} \right)/2\pi \ (\textrm{kHz}) ",ylim=(-0.1,1.1),xlim = (-1100,1100),label="Experimental data",foreground_color_legend = nothing)
# plot!(x_data,y_data,c=:black,label=false,lw=1)