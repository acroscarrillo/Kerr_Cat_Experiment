include("../../../src/src.jl") # import src.jl which has creation/annahilation operators defined

# @btime eigen(H_super_op( -H_eff(35,0,1,0,12) ) + C_ops_super) 2.177 s (54 allocations: 128.60 MiB)
# @btime eigen(H_super_op( -H_eff(40,0,1,0,12) ) + C_ops_super)  4.159 s (54 allocations: 218.32 MiB)

using DataFrames # this is like pandas
using CSV 
using ProgressBars
using Plots
using LaTeXStrings

⦼(A,B) = kron(A,B)

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


##############
# L_mat code #
##############
N = 40 # scales as N^2

# Define H_eff parameter space, in units of K
K = 1
Δ = -0*K
ϵ_1 = 0
ϵ_2_array = Vector(range(0,12,length=100)).*K

# Define collapse ops 
κ_1 = 0.025 #in units of K
n_th = 0.05
κ_p, κ_m = κ_1*n_th, κ_1*(1 + n_th) 

C_ops = ([ √(κ_p) * a(N)', √(κ_m) * a(N) ])
C_ops_super = C_super_ops(C_ops)

T_1_staircase_array = zeros(length(ϵ_2_array)) 
for (k,ϵ_2) in ProgressBar(enumerate(ϵ_2_array))
    H_temp_super = H_super_op( -H_eff(N,Δ,K,ϵ_1,ϵ_2) )
    L_temp = H_temp_super + C_ops_super
    λ_n = eigen(L_temp).values 
    T_1_staircase_array[k] = 1/smallest_nonzero_real(λ_n)
end
plot(ϵ_2_array,T_1_staircase_array,ylab=L"T_X=-1/Re[λ_1]",xlab=L"ϵ_2/K",yscale=:log,xlim=(0,13))  
hline!([1/κ_p])