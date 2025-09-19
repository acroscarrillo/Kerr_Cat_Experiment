# include("../../src/src.jl") 

using DataFrames # this is like pandas
using CSV 
using ProgressBars
using Plots
using LaTeXStrings
using LsqFit

τ_x_model(t, p) = 1 * exp.(- t / p[1])

function get_τ(x,y)
    τ = curve_fit(τ_x_model, x, y, [100.0]).param[1]
    return τ
end

N = 100 # Maximum photon number
N_traj = 500
K = 1
Δ = 0*K
ϵ_1 = 0*K
ϵ_2_array = Vector(range(0.1,12,length=20)).*K

# κ1/K = 0.025 and nth = 0.01
κ_1 =  0.025 
n_th = 0.1
κ_p, κ_m = κ_1*n_th, κ_1*(1 + n_th) 


# KT_f = 2πK*T_x, numbers from nick frattini
KT_x = 600e-6 # in seconds
KT_f = 2*π*(320e3)*KT_x
N_δt = Int(  5e4  )
KT_array = Vector(range(0,4*KT_f, length=N_δt))
δt = KT_array[2] - KT_array[1]
N_saves = 200

C_ops = [√(κ_p)*a(N)', √(κ_m)*a(N)]
X = ( a(N) + a(N)' ) / √(2)


τ_x_array = zeros(length(ϵ_2_array))
pbar = ProgressBar(total=length(ϵ_2_array)*N_traj)
for (n,ϵ_2) in enumerate(ϵ_2_array)
    # define ψ_0 as a coherent state at right well
    x_well, p_well = dblwell_minimas(K,ϵ_1,ϵ_2)[1], 0
    ψ_0 = coherent_state(N,x_well,p_well)

    # load propagator in memory
    U_T_temp = exp(  -im * δt * H_eff_nh(N,Δ,K,ϵ_1,ϵ_2,κ_p,κ_m)  )

    # start Trajectories
    avg_x_τ_array = zeros(N_saves)
    KT_array = zeros(N_saves)
    for n=1:N_traj
        x_array, ψ_t_array, t_array = H_eff_obs_traj(δt,N_δt, ψ_0, U_T_temp, C_ops, X; return_N = N_saves)
        avg_x_τ_array += x_array./ N_traj
        KT_array = t_array
        update(pbar)
    end

    τ_x_array[n] = get_τ(KT_array, avg_x_τ_array./ √(2*ϵ_2/K))
end 

width_pts = 246.0  # or any other value
inches_per_points = 1.0/72.27
width_inches = width_pts *inches_per_points
width_px= width_inches*100  # or  width_inches*DPI
plot(ϵ_2_array,1e6.*τ_x_array./(2*π*320e3), ylabel=L"T_x (\mu s)", xlabel=L"ϵ_2/K",lw=2,title=L"N="*"$N,   "*L"N_{traj}="*"$N_traj,   "*L"\kappa_1="*"$κ_1,   "*L"n_{th}="*"$n_th",yscale=:log)
scatter!(ϵ_2_array,1e6.*τ_x_array./(2*π*320e3), ms=2,legend=false,dpi=600,size = (width_px,width_px*0.6),grid=false,xtickfontsize=8,ytickfontsize=8,guidefont=font(8),widen=false,tickdirection=:out,right_margin = 4Plots.mm,titlefontsize=8,legendfontsize=8,left_margin = 0Plots.mm,bottom_margin = 0Plots.mm,fontfamilys = "Times New Roman",tickfontfamily = "Times New Roman",background_color_subplot=:white,foreground_color_legend = nothing,label=false)