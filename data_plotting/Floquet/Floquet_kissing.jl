include("../src/src.jl")

using CSV
using StatsPlots
using LaTeXStrings
using DataFrames

df = DataFrame(CSV.File("data/floquet_kissing.csv"))

df_formatted = filter(row -> row.Δϵ_n > 0.998, df)  # match paper limits

@df df_formatted plot(:ϵ_2/K, :Δϵ_n, markersize = .5, xlab=L"\epsilon_2/K",ylab=L"\epsilon-\epsilon_0",seriestype=:scatter, legend=false)

@df df plot(:ϵ_2/K, :Δϵ_n, markersize = .5, xlab=L"\epsilon_2/K",ylab=L"\tilde{\epsilon} ",seriestype=:scatter, legend=false,title=L"N="*string(N)*L",\omega_0="*string(ω_0)*L",g_n="*string(g_n)*L",\omega_d="*string(ω_d),guidefontsize=15)
#

savefig("figs/floquet_kissing.png")