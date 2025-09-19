include("../../../src/src.jl") # import src.jl which has creation/annahilation operators defined

using DataFrames # this is like pandas
using CSV 
using ProgressBars

N = 100 #from the calculations in the eff model, N>=25 is sufficient 
ω_0 = 1
# g_n = [2*10^(-5),8*10^(-6)]
g_n = [0.00075,1.27*10^(-7)]
# g_n = [1,1]
# ω_d = 2*ω_0


K = (10*g_n[1]^2)/(3*ω_0) - 3*g_n[2]/2

Ω_d_array = Vector(0:0.0001:1/10) #graph above could be truncated at 10
# Ω_d_array = Vector(0:0.005:1) #graph above could be truncated at 10

# Define data form
data_array = zeros( N*length(Ω_d_array), 6 ) # data form: Δϵ_n | Ω_d | K | ϵ_2 | ϵ_2/K | N

# The following is used to find ϵ_0 as described in the paper for your eyes only
E_n , V_n = eigen(H_0(N,ω_0,g_n))
V_0 = V_n[:, 1]

counter = 1
for (_, Ω_d) in ProgressBar(enumerate(Ω_d_array))
    ω_d = 2*( ω_0 + 3*g_n[2] -20*g_n[1]^2/(3*ω_0) + (6*g_n[2]+9*g_n[1]^2/ω_0)*(2*Ω_d/(3*ω_0))^2)
    ϵ_n, ϕ_n = qen_qmodes(N, ω_0, g_n, Ω_d, ω_d)

    overlaps = zeros(N)
    for n=1:N
        overlaps[n] = real( (V_0'*ϕ_n[:,n])*(ϕ_n[:,n]'*V_0) )
    end

    ϵ_0 = ϵ_n[argmax(overlaps)]
    V_0 = ϕ_n[:,argmax(overlaps)]
    
    ϵ_2 = 2*Ω_d*g_n[1]/(3*ω_0)
    for n=1:N
        data_array[counter,:] .= mod(ϵ_n[n] - ϵ_0, ω_d/2), Ω_d, K, ϵ_2, ϵ_2/K, N
        counter += 1
    end
end

# Put data in convenient DataFrame object and save it
df_floquet = DataFrame(data_array, ["Δϵ_n","Ω_d","K","ϵ_2","ϵ_2/K","N"]) 
CSV.write("data/floquet_kissing.csv", df_floquet)