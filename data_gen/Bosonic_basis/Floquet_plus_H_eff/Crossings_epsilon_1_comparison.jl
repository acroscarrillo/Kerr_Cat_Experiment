include("../../../src/src.jl") # import src.jl which has creation/annahilation operators defined

using LinearAlgebra # to diagonalise stuff
using DataFrames # this is like pandas
using ProgressBars
using CSV 


N = 30

#########
# H_eff #
#########
# Define parameter space
Δ = 0
K = 1
ϵ_1_array = Vector( range(0, 15, length=1000) ).*K
ϵ_2 = 17*K

# Define crossing data form
h_eff_array = zeros(N*length(ϵ_1_array), 4 ) # data form: ΔE_n | ϵ_2 | ϵ_1 | floquet?

# Generate data within parameter space
counter = 1
for ϵ_1 in ϵ_1_array
    H_temp = Hermitian( -H_eff(N,Δ,K,ϵ_1,ϵ_2) )
    lamb, _ = eigen(H_temp)
    for i=1:N
        h_eff_array[counter,:] .= lamb[i]-minimum(lamb), ϵ_2, ϵ_1, 0
        counter += 1
    end
end


###########
# Floquet #
###########
# Define parameter space
ω_0 = 1
ω_1 = 1
# g_n = [0.00075, 1.27*10^(-7)].*ω_0
g_n = [-0.0034, -3.3333333333333335e-5]
K = (10*g_n[1]^2)/(3*ω_0) - 3*g_n[2]/2
Ω_1_array = Vector( range(0, (1*K)*(ϵ_1_array[end])/2, length=100) )
Ω_2_array = Vector( range(0, 3*ϵ_2*ω_0*(1*K)/(4*g_n[1]), length=100) )

# Sanity checks
Ω_1_max = Ω_1_array[end]
Ω_2_max = Ω_2_array[end]

ϵ_1_max = 2*Ω_1_max/K # check: in K units like in H_eff above 
ϵ_2_max = Ω_2_max*4*g_n[1]/(3*ω_0*K) # check: in K units like in H_eff above 

# define data array to store Floquet data
floquet_array = zeros(N*length(Ω_1_array), 4 )  # data form: ΔE_n | ϵ_2 | ϵ_1 | floquet?

# Generate data within parameter space
E_n , V_n = eigen(H_0(N,ω_0,g_n))
V_0 = V_n[:, 1]

pbar = ProgressBar(total=length(Ω_1_array)*length(Ω_2_array))
counter = 1
Ω_1, Ω_2 = 0, 0
for (i,_) in enumerate(Ω_2_array)
    for (j,_) in enumerate(Ω_1_array)
        # otherwise for loop is not adiabatic (makes jumps in Ω_1, Ω_2), this is like a snake in the Ω_1-Ω_2 plane. Failing to do this is an error.
        if isodd(i)
            Ω_2, Ω_1 = Ω_2_array[i], Ω_1_array[j]
        else
            Ω_2, Ω_1 = Ω_2_array[i], Ω_1_array[length(Ω_1_array)-j+1]
        end

        ω_1, ω_2 = ω_a(ω_0,g_n,Ω_2), 2*ω_a(ω_0,g_n,Ω_2)
        ϵ_1, ϵ_2 = Ω_1/2, g_n[1]*Π(ω_0,Ω_2,ω_2)

        # ω_2 = 2*ω_a(ω_0,g_n,Ω_2)
        # # ϵ_2 = g_n[1]*Π(ω_0,Ω_2,ω_2)
        # # ϵ_2 = g_n[1]*Π(ω_0,Ω_2)
        # ϵ_2 = Ω_2*4*g_n[1]/(3*ω_0)
        # ϵ_1 = 2*Ω_1

        ϵ_n, ϕ_n = qen_qmodes(N, ω_0, g_n, Ω_1, ω_1, Ω_2, ω_2)

        # adiabatic tracking of the ground state
        overlaps = zeros(N)
        for n=1:N
            overlaps[n] = real( (V_0'*ϕ_n[:,n])*(ϕ_n[:,n]'*V_0) )
        end

        ϵ_0 = ϵ_n[argmax(overlaps)]
        V_0 = ϕ_n[:,argmax(osverlaps)]
        ϵ_n = sort(mod.(ϵ_n .- ϵ_0, ω_2/2))
        # store data only when Ω_2 of interest is reached
        if Ω_2 == Ω_2_array[end]
            for n=1:N
                floquet_array[counter,:] .= ϵ_n[n]/(1*K), ϵ_2/(1*K), ϵ_1/(1*K), 1
                counter += 1
            end
        end
        update(pbar)
    end
end

# put data in convenient DataFrame object
df_floquet = DataFrame(floquet_array, ["ΔE_n","ϵ_2","ϵ_1","floquet"]) 
df_H_eff = DataFrame(h_eff_array, ["ΔE_n","ϵ_2","ϵ_1","floquet"]) 
df_combination = DataFrame(vcat(floquet_array,h_eff_array), ["ΔE_n","ϵ_2","ϵ_1","floquet"]) 
# CSV.write("data/comparison_crossings_epsilon_1.csv", df_floquet)