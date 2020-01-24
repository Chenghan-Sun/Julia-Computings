
using Pkg

Pkg.add("SpecialFunctions")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("LinearAlgebra")
Pkg.add("Statistics")
Pkg.add("Random")
Pkg.add("NLopt")
Pkg.add("PyCall")

Pkg.add("PyPlot")

# Define local methods module
module LocalMethods

using SpecialFunctions: besselk, gamma

export ℳ_ν, M_νρσ

tν𝒦t(t,ν) = t^ν * besselk(ν, t)

function ℳ_ν(t, ν)
    pt, pν, p0, p1 = promote(t, ν, 0, 1)
    return (pt==p0) ? p1 : tν𝒦t(√(2pν)*pt,pν) * 2^(1-pν) / gamma(pν)
end

function M_νρσ(t; ν, ρ, σ) #T = L-2 x-y
    return σ * σ * ℳ_ν(t/ρ, ν)
end

end

#M_νρσ.(nrm; )

# loading packages 
using .LocalMethods 
using CSV
using DataFrames
using PyPlot
using LinearAlgebra
using Statistics
using Random

# loading data
@show pwd()
data_file = "./data/UStmax.csv"
csv_data  = CSV.File(data_file)
df = DataFrame!(csv_data)  # puts it into a data DataFrame

# Extract the global mean temperature 
G_tmean = mean(df.UStmax)

# Subtract the global mean temperature from all the data points
df.UStmax = df.UStmax .- G_tmean

# Show current DataFrame
df

# Function for extracting 10% of data for testing purpose 
function subset_extractor(df, test_weight)
    p = shuffle(1:size(df, 1)) # random subset
    train_p = view(p, 1:round(Int,(1-test_weight)*size(df, 1)))
    train_set = df[train_p,:]
    test_p = view(p, round(Int,(1-test_weight)*size(df, 1))+1:size(df, 1))
    test_set = df[test_p,:]
    return train_set, test_set
end

# call the function for extracting 10% test set, the remaining part as train set 
train_set, test_set = subset_extractor(df, 0.1)

# construct cov matrix
function calc_nrm(train_set, test_set)
    nrm_train = sqrt.((train_set.lat .- train_set.lat').^2 .+ (train_set.lon .- train_set.lon').^2)
    nrm_train_test = sqrt.((train_set.lat .- test_set.lat').^2 .+ (train_set.lon .- test_set.lon').^2)
    nrm_test = sqrt.((test_set.lat .- test_set.lat').^2 .+ (test_set.lon .- test_set.lon').^2)
    return nrm_train, nrm_train_test, nrm_test
end

# nrm matrices 
nrm_train, nrm_train_test, nrm_test = calc_nrm(train_set, test_set)
#@show nrm_train
#@show nrm_train_test
#@show nrm_test

# function for the loglikelihood 
function MLE(nrm, z, ν, ρ, σ)
    Σ = M_νρσ.(nrm; ν=ν, ρ=ρ, σ=σ) |> Symmetric
    ch = cholesky(Σ)
    ch_lower = ch.L
    # suggested to define new variable for ch_lower\z 
    return -0.5 * dot(ch_lower\z, ch_lower\z) - sum(log.(diag(ch_lower))) #by using the expression in notes
end

using NLopt

# call the MLE function
llmax = function(x)
    return MLE(nrm_train, train_set.UStmax, 0.5, x[1], x[2])
end

opt1 = Opt(:LN_NELDERMEAD, 2) # using the optimizer picked from julia-3.jl
opt1.max_objective = (x, grad) -> llmax(x)
# Note: could use MLE point to optimizer to prevent global variables issue

opt1.lower_bounds = [0.1, 0.1]
opt1.upper_bounds = [50.0, 50.0]
optf1, optx1, ret1 = optimize(opt1, Float64[25,25]) #start from the middle

llmax2=function(x)
    return MLE(nrm_train, train_set.UStmax, 1.5, x[1], x[2])
end

opt2 = Opt(:LN_NELDERMEAD, 2)
opt2.max_objective = (x,grad) -> llmax2(x)

opt2.lower_bounds = [0.1, 0.1]
opt2.upper_bounds = [50.0, 50.0]
optf2, optx2, ret2 = optimize(opt2, Float64[25,25])

llmax3 = function(x)
    return MLE(nrm_train, train_set.UStmax, 2.5, x[1], x[2])
end

opt3 = Opt(:LN_NELDERMEAD, 2)
opt3.max_objective = (x,grad) -> llmax3(x)

opt3.lower_bounds = [0.1, 0.1]
opt3.upper_bounds = [50.0, 50.0]
optf3, optx3, ret3 = optimize(opt3, Float64[25,25])

llmax4 = function(x)
    return MLE(nrm_train, train_set.UStmax, 3.5, x[1], x[2])
end

opt4 = Opt(:LN_NELDERMEAD, 2)
opt4.max_objective = (x,grad) -> llmax4(x)

opt4.lower_bounds = [0.1, 0.1]
opt4.upper_bounds = [50.0, 50.0]
optf4, optx4, ret4 = optimize(opt4, Float64[1,1])

llmax5 = function(x)
    return MLE(nrm_train, train_set.UStmax, 0.5, x[1], x[2])
end

opt5 = Opt(:LN_BOBYQA, 2) 
opt5.max_objective = (x, grad) -> llmax5(x)

opt5.lower_bounds = [0.1, 0.1]
opt5.upper_bounds = [50.0, 50.0]
optf5, optx5, ret5 = optimize(opt5, Float64[25,25]) #start from the middle

llmax7 = function(x)
    return MLE(nrm_train, train_set.UStmax, 0.5, x[1], x[2])
end

opt7 = Opt(:LN_PRAXIS, 2) 
opt7.max_objective = (x, grad) -> llmax7(x)

opt7.lower_bounds = [0.1, 0.1]
opt7.upper_bounds = [50.0, 50.0]
optf7, optx7, ret7 = optimize(opt7, Float64[25,25]) #start from the middle

llmax8 = function(x)
    return MLE(nrm_train, train_set.UStmax, 0.5, x[1], x[2])
end

opt8 = Opt(:LN_SBPLX, 2) 
opt8.max_objective = (x, grad) -> llmax8(x)

opt8.lower_bounds = [0.1, 0.1]
opt8.upper_bounds = [50.0, 50.0]
optf8, optx8, ret8 = optimize(opt8, Float64[1,1]) #start from the middle

Σ_train = M_νρσ.(nrm_train; ν=0.5, ρ=1.1431971490358848, σ=3.8777562138773285)

Σ_train_test = M_νρσ.(nrm_train_test; ν=0.5, ρ=1.1431971490358848, σ=3.8777562138773285)

Σ_test = M_νρσ.(nrm_test; ν=0.5, ρ=1.1431971490358848, σ=3.8777562138773285)

# The predicted values given by
pred_val = Σ_train_test' * (Σ_train \ train_set.UStmax)

# The residuals given by histogram plot
using Plots
res = pred_val - test_set.UStmax
histogram(res, xlabel="Residuals", ylabel="Occurence Frequency", title="Histogram of Residuals")

# Report the RMSE
@show rmse = sqrt(sum(res.^2.) / length(res))

# using code from Julia-4.jl
using PyCall
scii = pyimport("scipy.interpolate")

px = train_set.lon
py = train_set.lat
pf = train_set.UStmax

# define the resolution
nxgrid = 600
nygrid = 600
lat_min, lat_max = extrema(df.lat)
lon_min, lon_max = extrema(df.lon)

X = range(lon_min, lon_max, length=nxgrid) .+ fill(0, nxgrid, nygrid) 
Y = range(lat_min, lat_max, length=nygrid)' .+ fill(0, nxgrid, nygrid) 

griddata = scii.griddata

fn = griddata((px, py), pf, (X, Y), method="nearest")
fl = griddata((px, py), pf, (X, Y), method="linear")
fc = griddata((px, py), pf, (X, Y), method="cubic")

# Make plots
fig, ax = subplots(nrows=2, ncols=2)

ax[1,1].scatter(px, py, c="k", alpha=0.2, marker=".")
ax[1,1].set_title("Sample points of lat/lon information")

for (method,finterp,rc) ∈ zip(("nearest","linear","cubic"), (fn, fl, fc),((1,2),(2,1),(2,2)))
    ax[rc[1],rc[2]].contourf(X, Y, finterp)
    ax[rc[1],rc[2]].set_title("method = $method")
end

tight_layout()

# predicted values of three linear methods 
pred_nearest = griddata((px, py), pf, (test_set.lon, test_set.lat), method="nearest")
pred_linear = griddata((px, py), pf, (test_set.lon, test_set.lat), method="linear")
pred_cubic = griddata((px, py), pf, (test_set.lon, test_set.lat), method="cubic")

# find residuals 
res_nearest = pred_nearest - test_set.UStmax
res_linear = pred_linear - test_set.UStmax
res_cubic = pred_cubic - test_set.UStmax

# drop out all the NaN values
using Missings
dropmissingNaN(x)= filter( x->(!isnan(x)), Missings.coalesce.( x , NaN ) )
res_nearest = dropmissingNaN(res_nearest)
res_linear = dropmissingNaN(res_linear)
res_cubic = dropmissingNaN(res_cubic)

# Nearest case given by histogram plot
histogram(res_nearest, xlabel="Residuals", ylabel="Occurence Frequency", title="Nearest Case", seriescolor = "yellow")

# Linear case given by histogram plot
histogram(res_linear, xlabel="Residuals", ylabel="Occurence Frequency", title="Linear Case", seriescolor = "green")

# Cubic case given by histogram plot
histogram(res_cubic, xlabel="Residuals", ylabel="Occurence Frequency", title="Cubic Case", seriescolor = "orange")

# report RMSEs
@show rmse_nearest = sqrt(sum(res_nearest.^2) / length(res_nearest))
@show rmse_linear = sqrt(sum(res_linear.^2) / length(res_linear))
@show rmse_cubic = sqrt(sum(res_cubic.^2) / length(res_cubic))

LinNDInterp = scii.LinearNDInterpolator

fL = LinNDInterp((px, py), pf) # note: fL is automatically vectorized

# Make plot
fig, ax = subplots(nrows=2, ncols=1)

ax[1].scatter(px, py, c="k", alpha=0.2, marker=".")
ax[1].set_title("Sample points of lat/lon information")

ax[2].contourf(X, Y, fL.(X, Y))
ax[2].set_title("LinearNDInterpolator")

tight_layout()

# predicted values using linearND methods 
pred_linearND = fL(test_set.lon, test_set.lat)

# find residuals and histogram plot
res_linearND = pred_linearND - test_set.UStmax
# remove the NaN values
res_linearND = dropmissingNaN(res_linearND)
histogram(res_linearND, xlabel="Residuals", ylabel="Occurence Frequency", title="LinearND Case", seriescolor = "purple")


# report RMSE
rmse_linearND = sqrt(sum(res_linearND.^2) / length(res_linearND))

# define the function of Gaussian autocovariance model
Gaussian_AutoCov = function(t)
    K_t = σ * σ * exp( - t^2/ρ^2)
    return K_t
end

# function for the loglikelihood 
function MLE_GAC(nrm, z, ρ, σ) #GAC = Gaussian AutoCovariance
    Σ = Gaussian_AutoCov.(nrm) |> Symmetric
    ch = cholesky(Σ)
    ch_lower = ch.L
    return  - 0.5 * dot(ch_lower\z, ch_lower\z) - sum(log.(diag(ch_lower)))
end

# Using the pre-determined optimal values of ρ, σ to construct cov matrices
ρ=1.1431971490358848
σ=3.8777562138773285
nrm_GAV_train = Gaussian_AutoCov.((train_set.lat .- train_set.lat').^2 .+ (train_set.lon .- train_set.lon').^2)

nrm_GAV_train_test = Gaussian_AutoCov.((train_set.lat .- test_set.lat').^2 .+ (train_set.lon .- test_set.lon').^2)

nrm_GAV_test = Gaussian_AutoCov.((test_set.lat .- test_set.lat').^2 .+ (test_set.lon .- test_set.lon').^2)

# predicted values using Gaussian AutoCovariance methods 
pred_GAC = nrm_GAV_train_test' * (nrm_GAV_train \ train_set.UStmax)

# Find the residuals and make histogram plot
res_GAC = pred_GAC - test_set.UStmax
histogram(res_GAC, xlabel="Residuals", ylabel="Occurence Frequency", title="Gaussian AutoCov Case", seriescolor = "red")

# report RMSE
rmse_GAC = sqrt(sum(res_GAC.^2.) / length(res_GAC))


