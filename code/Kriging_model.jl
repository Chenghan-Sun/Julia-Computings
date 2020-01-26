
# Import necessary packages 
using PyPlot
using LinearAlgebra
using Statistics
using Random
using StaticArrays
import JLD2
using SpecialFunctions: besselk, gamma
import DynamicPolynomials: @polyvar, monomials
using FixedPolynomials
using NLopt

# load data
elv_obs = JLD2.FileIO.load("../data/UStmax.jld2", "elv")
UStmax_obs = JLD2.FileIO.load("../data/UStmax.jld2", "UStmax")
lon_obs = JLD2.FileIO.load("../data/UStmax.jld2", "lon")
lat_obs = JLD2.FileIO.load("../data/UStmax.jld2", "lat")

# load a grid of lat and lon with corresponding elevation 
elv_grid      = JLD2.FileIO.load("../data/krig_at.jld2", "elv_grid")
lat_grid_side = JLD2.FileIO.load("../data/krig_at.jld2", "lat_grid_side")
lon_grid_side = JLD2.FileIO.load("../data/krig_at.jld2", "lon_grid_side")

# Construct lat-lon data matrix
data_mat = [lon_obs lat_obs]

figure(figsize=(10,5))
scatter(lon_obs, lat_obs, c=elv_obs, s=1)
title("elevation")
colorbar()

figure(figsize=(10,5))
scatter(lon_obs, lat_obs, c=UStmax_obs, s=1)
title("temperature")
colorbar()

figure(figsize=(10,5))
imshow(elv_grid, 
    extent=(extrema(lon_grid_side)..., extrema(lat_grid_side)...),
)

function get_distance(lon_data, lat_data) 
    # Input: longitude and latitude data
    # Output: distance t
    t = sqrt.((lon_obs .- lon_data').^2 .+ (lat_obs .- lat_data').^2)
    return t
end

# Generalized covariance function that matches the principle irregular term from our Matérn covariance model
function get_Gnu(t::A, ν::B) where {A<:Real, B<:Real}
    C = promote_type(A,B)
    if t==A(0)
        return C(0)
    end
    
    if floor(ν)==ν
        scν = (2 * (-1)^ν) * (- (ν/2)^ν / gamma(ν) / gamma(ν+1))
        return C(scν * t^(2ν) * log(t))
    else
        scν = (π / sin(ν*π)) * (- (ν/2)^ν / gamma(ν) / gamma(ν+1))
        return C(scν * t^(2ν))
    end
end

@polyvar x[1:2] #describe Polynomial in two vars x and y
monos = monomials(x,0:2)
n = 4408 #dimension of dataset 

beta_initial = [1,1,1,1,1,1] # initialize beta coeffs to be all ones
beta_diag_mat = Diagonal(beta_initial)
mat_F_initial = []  # initialize the F matrix

function get_mat_F(mat_F_initial, data_mat, elv_data)
    # Input: data_mat: matrix contains longitude and latitude data
    #        elv_data: elevation data 
    # Output: matrix F
    for i = 1:length(beta_initial)
        ploy_f = Polynomial(monos' * beta_diag_mat[i,:])
        f = mapslices(ploy_f, data_mat'; dims=[1]) #each row of F matrix 
        mat_F = append!(mat_F_initial, f)
    end 

    # Construct the matrix F
    mat_F = [mat_F[1:n] mat_F[n+1:2n] mat_F[2n+1:3n] mat_F[3n+1:4n] mat_F[4n+1:5n] mat_F[5n+1:end] elv_obs]
    mat_F = hcat(mat_F')
    mat_F = convert(Array{Float64,2}, mat_F) #convert data to Array{Float64,2} format
    return mat_F
end

# see the resulting matrix F
mat_F = get_mat_F(mat_F_initial, data_mat, elv_obs)
size(mat_F)

# find the nullspace of matrix F
function get_cramer(mat_F)
    mat_Mt = nullspace(mat_F) # mat_Mt = M'
    mat_M = mat_Mt'
    return mat_M
end 

# see the resulting matrix M
mat_M = get_cramer(mat_F)
@show size(mat_M)

function get_REML(d, mat_M, mat_G, σz, σe)
    # Input: d = UStmax_obs; mat_M: from get_cramer function; mat_G: from get_distance and get_Gnu function
    #        σz, σe: covariance parameters that need to be optimized 
    # Output: RLL = Restricted log likelihood; K: covariance matrix
    mat_cov = (σz^2).*mat_G .+ (σe^2).*Matrix(I, n, n)
    A = mat_M*mat_cov*mat_M' |> Symmetric
    ch = cholesky(A) 
    RLL = -1/2*dot(mat_M*d, ch.L'\(ch.L \ mat_M*d)) - sum(log.(diag(ch.L))) - n/2 * log(2*π) 
    return RLL, mat_cov
end

# find the Matrix G for 𝜈 = 1/2
t = get_distance(lon_obs, lat_obs)
mat_G_1 = get_Gnu.(t, 1/2)

# Call get_REML function for optimation of σz and σe
llmax1 = function(x)
    return get_REML(UStmax_obs, mat_M, mat_G_1, x[1], x[2])[1]
end

opt1 = Opt(:LN_NELDERMEAD, 2) # pick the optimizer
opt1.max_objective = (x, grad) -> llmax1(x)

opt1.lower_bounds = [0.01, 0.01]
opt1.upper_bounds = [10, 10]
#opt.maxtime = 1200 #in sec
optf1, optσ1, ret1 = optimize(opt1, Float64[0.011, 0.011]) #start from the lower bounds 
#(-6689.643115077346, [0.46285856537448855, 0.9216567680974481], :XTOL_REACHED)

# find the Matrix G for 𝜈 = 3/2
mat_G_2 = get_Gnu.(t, 3/2)

# Call get_REML function for optimation of σz and σe
llmax2 = function(x)
    return get_REML(UStmax_obs, mat_M, mat_G_2, x[1], x[2])[1]
end

opt2 = Opt(:LN_NELDERMEAD, 2) # pick the optimizer
opt2.max_objective = (x, grad) -> llmax2(x)

opt2.lower_bounds = [0.01, 0.01]
opt2.upper_bounds = [10, 10]
opt.maxtime = 1200 #in sec
optf2, optσ2, ret2 = optimize(opt2, Float64[0.011, 0.011]) #start from the lower bounds 

σz_opt = 1.061
σe_opt = 0.773

# Recall the REML function that returns optimized MLE and corresponding mat_cov
mat_G = mat_G_1
RLL_opt, mat_cov_opt = get_REML(UStmax_obs, mat_M, mat_G, σz_opt, σe_opt)

function krig_linear_solver(mat_cov_opt, mat_F, d)
    # Construct the linear system and solve for vector beta and Ck
    mat_LHS = [[mat_cov_opt mat_F']; [mat_F zeros(7,7)]]
    vec_RHS = [d; zeros(7,1)]
    vec_beta_ck = inv(mat_LHS)*vec_RHS
    vec_ck = vec_beta_ck[1:n]
    vec_beta = vec_beta_ck[n+1:end]
    return vec_beta, vec_ck
end

# Get the resulting coeffs as vectors 
vec_beta, vec_ck = krig_linear_solver(mat_cov_opt, mat_F, UStmax_obs)

mat_F_initial = []

# Make a function for the observed data prediction
function krig_pred_obs(lon_data, lat_data, elv_data, vec_beta, vec_ck, σz_opt, mat_cov_opt)
    data_mat = [lon_data lat_data]
    mat_F = get_mat_F(mat_F_initial, data_mat, elv_data)
    t = get_distance(lon_data, lat_data)
    K = (σz_opt^2).*get_Gnu.(t, 1/2)
    Y_hat = mat_F'*vec_beta + K'*vec_ck
    return Y_hat
end

est_UStmax_obs = krig_pred_obs(lon_obs, lat_obs, elv_obs, vec_beta, vec_ck, σz_opt, mat_cov_opt)


# plot the krig prediction at the observation points
figure(figsize=(10,5))
scatter(lon_obs, lat_obs, c=est_UStmax_obs, s=1)
title("Krig interpolation at the observed points")
colorbar()

figure(figsize=(10,5))
scatter(lon_obs, lat_obs, c=UStmax_obs, s=1)
title("Comparing with the observed temperature")
colorbar()

figure(figsize=(10,5))
scatter(lon_obs, lat_obs, c=est_UStmax_obs .- UStmax_obs, s=1)
title("Residual of temperature")
colorbar()

@polyvar x[1:2] #describe Polynomial in two vars x and y
monos = monomials(x,0:2)
fpx = Polynomial(monos' * vec_beta[1:end-1])

# Make a function for the grid points data prediction
function krig_pred_grid(lon_data, lat_data, elv_data; vec_beta, vec_ck, σz_opt, mat_cov_opt)
    data_mat = [lon_data, lat_data]
    Fc = fpx(data_mat) + vec_beta[end] * elv_data
    t = sqrt.((lon_obs .- lon_data).^2 .+ (lat_obs .- lat_data).^2)
    K = (σz_opt^2).*get_Gnu.(t, 1/2)
    Y_hat = Fc + K'*vec_ck
    return Y_hat
end

est_UStmax_grid = krig_pred_grid.(lon_grid_side, lat_grid_side, elv_grid; vec_beta = vec_beta, vec_ck = vec_ck, σz_opt = σz_opt, mat_cov_opt = mat_cov_opt)

# prediction for a grid of lon-lat data set
figure(figsize=(10,5))
imshow(est_UStmax_grid, 
    extent=(extrema(lon_grid_side)..., extrema(lat_grid_side)...),
    vmin=0, vmax=50
)
title("Prediction on grid temperature with elevation")
colorbar()

est_UStmax_grid_wo_elev = est_UStmax_grid - vec_beta[end].*elv_grid

figure(figsize=(10,5))
imshow(est_UStmax_grid_wo_elev, 
    extent=(extrema(lon_grid_side)..., extrema(lat_grid_side)...),
    vmin=0, vmax=50
)
title("Prediction on grid temperature without elevation")
colorbar()


