
module Mod

using XFields
using Statistics
using Random
import JLD2

const data_dir  = "/Users/furinkazan/Box/STA_250/data-4/" #change dir
const data_file = "data4.jld2" 
load(varname) = JLD2.FileIO.load(data_dir*data_file, varname) 


const cls = JLD2.FileIO.load(data_dir*"cls.jld2", "cls")
const cT =  cls[:cl_len_scalar][:,1]  ./ cls[:factor_on_cl_cmb]
const ℓT =  cls[:ell]

macro block(ex)
    :((() -> $(esc(ex)))())
end

function white_noise_sim(::Type{FT}) where FT<:FourierTransform
	gr = Grid(FT)
	w = randn(gr.nxi...) ./ sqrt(gr.Ωx)
	return Smap{FT}(w)
end

function power(f::Sfield{FT}; bin::Int=1, kmax=Inf, mult=1) where FT<:FourierTransform
    if bin == 0
        return pwr
    else
		k      = wavenumber(FT)
		Δk     = k[3,1] - k[2,1]
		pwr    = abs2.(mult .* f[:k])
		k_left = 0
		while k_left < min(kmax, maximum(k))
			k_right    = k_left + bin
			indx       = k_left .< k .<= k_right
			pwr[indx] .= mean( pwr[indx] )
			k_left     = k_right
        end
        return pwr
    end
end

# custom pcg with function composition (Minv * A \approx I)
function pcg(Minv::Function, A::Function, b, x=0*b; nsteps::Int=75, rel_tol::Float64 = 1e-8)
    r       = b - A(x)
    z       = Minv(r)
    p       = deepcopy(z)
    res     = dot(r,z)
    reshist = Vector{typeof(res)}()
    for i = 1:nsteps
        Ap        = A(p)
        α         = res / dot(p,Ap)
        x         = x + α * p
        r         = r - α * Ap
        z         = Minv(r)
        res′      = dot(r,z)
        p         = z + (res′ / res) * p
        rel_error = squash(sqrt(dot(r,r)/dot(b,b)))
        if rel_error < rel_tol
            return x, reshist
        end
        push!(reshist, rel_error)
        res = res′
    end
    return x, reshist
end


end # module

using XFields
using PyPlot
using PyCall
using LinearAlgebra
using Statistics
using Random
using ProgressMeter
using .Mod

const prop_cycle = plt.rcParams["axes.prop_cycle"]
const colors     = prop_cycle.by_key()["color"]

gr, FT, d, Ma, Bm, Tf, Wn, Cn, Cf, MaWn = Mod.@block begin
	
	ni   = Mod.load("ni")        
	Δxi  = Mod.load("Δxi")     # in rads  
	FT   = rFFT(nᵢ=ni, Δxᵢ=Δxi)  # define Fourier transform
	gr   = Grid(FT) 			   # get corresponding coordinate grid object
    k    = wavenumber(FT)


	Bm = Mod.load("bm") |> Sfourier{FT} |> DiagOp # beam 1.5 arcmins
    Tf = Mod.load("tf") |> tf->(tf[k .<= 350].=1e-4;tf) |> Sfourier{FT} |> DiagOp # transfer function w/beam
	Cn = Mod.load("cn") |> Sfourier{FT} |> DiagOp # noise spectral density
	Wg = Mod.load("wg") |> Smap{FT}     |> DiagOp # pixel weights
	Ma = Mod.load("ma") |> m->(m[m .< 1e-3].=0;m) |> Smap{FT} |> DiagOp # pixel mask

	Wn = (1e-10 .* (Wg)[:]) |> 
			m->(m[m .<= 1e-5].=0;m) |>
			Smap{FT} |> 
			DiagOp |>
			sqrt |>
			inv

    MaWn = Ma * Wn

    Cf = Mod.JLD2.FileIO.load(
        joinpath(Mod.data_dir, "cffk4.jld2"), 
        "cffk"
    )  |> Sfourier{FT} |> DiagOp # noise spectral density



	d  = Ma * Smap{FT}(Mod.load("d")) # data

	return gr, FT, d, Ma, Bm, Tf, Wn, Cn, Cf, MaWn

end

Wn[:] |> matshow

# Here is the processed data 
d |> x->(figure();imshow(x[:],vmin=-125,vmax=125, cmap = "RdBu_r")); colorbar()

# Compare this with a simulation of the CMB masked in the same way as the SPT data.
f_sim = sqrt(Cf) * Mod.white_noise_sim(FT)
Ma*f_sim |> x->(figure();imshow(x[:], cmap = "RdBu_r")); colorbar()

# The main reason these look so different is the transfer function (i.e. `Tf`)
# which encodes how the CMB enters the detector, the scanning of the sky and the 
# internal processing to get the observations to map form. Here is what the 
# CMB looks like when the transfer function is applied.
Ma*Tf*f_sim |> x->(figure();imshow(x[:],vmin=-125,vmax=125, cmap = "RdBu_r")); colorbar()

Mod.@block begin

    tmap_full = Mod.load("tmap_full")
    tmap_full |> x->(figure();imshow(x,vmin=-125,vmax=125));colorbar()

end

n_sim = sqrt(Cn) * Mod.white_noise_sim(FT)
f_sim = sqrt(Cf) * Mod.white_noise_sim(FT)
d_sim = Ma * Tf * f_sim + MaWn * n_sim
d_sim |> x->(figure();imshow(x[:],vmin=-150,vmax=150, cmap = "RdBu_r")); colorbar()
d	  |> x->(figure();imshow(x[:],vmin=-150,vmax=150, cmap = "RdBu_r")); colorbar()

# Probe the noise by subtracting a beam convolution of the data (and the sim)
Bm′ = 2 |>  σf->deg2rad(σf/60)^2/8/log(2) |> # gaussian 
            σb²->exp.((.-σb²./2).*wavenumber(FT).^2) |> 
            x->DiagOp(Sfourier{FT}(x)) # diagonal operator

# Spatial coordinates
(d_sim-(Bm′*d_sim)) |> x->(figure();imshow(x[:],vmin=-10,vmax=10, cmap = "RdBu_r")); colorbar()
(d-(Bm′*d))			|> x->(figure();imshow(x[:],vmin=-10,vmax=10, cmap = "RdBu_r")); colorbar()

# Frequency coordinates
(d_sim-(Bm′*d_sim)) |> x->(figure();imshow(log.(abs.(x[!][1:end,2:end])), vmin=-17,vmax=-8, cmap = "RdBu_r")); colorbar()
(d-(Bm′*d))         |> x->(figure();imshow(log.(abs.(x[!][1:end,2:end])), vmin=-17,vmax=-8, cmap = "RdBu_r")); colorbar()

# bandpowers
Mod.@block begin

	k        = wavenumber(FT)
    fsky     = Ma[:] |> mean
	pwrd     = Mod.power(d;     bin=5, kmax = gr.nyqi[1]*3/4, mult = wavenumber(FT))
    pwrd_sim = Mod.power(d_sim; bin=5, kmax = gr.nyqi[1]*3/4, mult = wavenumber(FT))
    theory = fsky .* k.^2 .* Cf[!] .* Tf[!].^2 ./ gr.Ωk 

    figure()    
    loglog(k[:,1], pwrd[:,1], label="data")
    loglog(k[:,1], pwrd_sim[:,1], label="simulated data")
    loglog(k[:,1], theory[1,1:gr.nki[1]], label="theory without masking")
    legend()

end

Cn_max = 1.1 .* maximum(real.(Cn[!]))
Cnᶜ    = DiagOp(Sfourier{FT}(Cn_max .- real.(Cn[!]))) 
Cnnᶜ   = DiagOp(Sfourier{FT}(fill(Cn_max,gr.nki...))) 

 
n_sim   = sqrt(Cn)  * Mod.white_noise_sim(FT)
nᶜ_sim  = sqrt(Cnᶜ) * Mod.white_noise_sim(FT)
nnᶜ_sim = n_sim + nᶜ_sim # this should be white
# n_sim 	|> x->matshow(log.(abs.(x[!][1:250,2:250]))); colorbar()
# nᶜ_sim 	|> x->matshow(log.(abs.(x[!][1:250,2:250]))); colorbar()
# nnᶜ_sim   |> x->matshow(log.(abs.(x[!][1:250,2:250]))); colorbar()

# this should have noise model given by Ma * white
d_simᶜ = d_sim + MaWn * nᶜ_sim
d_sim  |> x->(figure();imshow(x[:],vmin=-150,vmax=150, cmap = "RdBu_r"))
d_simᶜ |> x->(figure();imshow(x[:],vmin=-150,vmax=150, cmap = "RdBu_r"))

nᶜ_sim = sqrt(Cnᶜ) * Mod.white_noise_sim(FT)
d_simᶜ = d_sim + MaWn * nᶜ_sim
dᶜ     = d     + MaWn * nᶜ_sim;

# set up the operators 
# Fill this in 
invN = inv(Cn_max*MaWn^2)
invCf = inv(Cf) |> x->(y=x[!];y[1]=y[2];DiagOp(Sfourier{FT}(y)))
pre = inv((1/Cn_max) * Tf^2 + invCf) # Preconditioner --> 1st arg

A = Tf*Ma*invN*Ma*Tf
B = invCf;

# test on the simulation
@time f_sim_wf, hist_sim = Mod.pcg(
    f -> pre*f, #preconditioner
    f -> A*f + B*f, #linear operator
    Tf * Ma * invN * d_simᶜ, #field
    nsteps  = 350,
    rel_tol = 1e-15,
)

f_sim_wf |> x->(figure();imshow(x[:], cmap = "RdBu_r")); colorbar()

# Run on the data
@time f_data_wf, hist_data = Mod.pcg(
    f -> pre*f,
    f -> A*f + B*f,
    Tf * Ma * invN * dᶜ,
    nsteps  = 350,
    rel_tol = 1e-15,
)

f_data_wf |> x->(figure();imshow(x[:], cmap = "RdBu_r")); colorbar()

# residual magnitude
semilogy(hist_sim, label="pcg on the simulated data")
semilogy(hist_data, label="pcg on the data", "--")
legend()

# first simulate a new realization of f and n
n_sim′   = sqrt(Cn) * Mod.white_noise_sim(FT)
nᶜ_sim′  = sqrt(Cnᶜ) * Mod.white_noise_sim(FT)

f_sim′ = sqrt(Cf) * Mod.white_noise_sim(FT)
ε_sim′ = MaWn * (n_sim′ + nᶜ_sim′);

# set up the operators
# --> same as previous code block

# test on the simulation
@time f_sim_wfpsim, hist_sim = Mod.pcg(
    f -> pre*f,
    f -> A*f + B*f,
    Tf * Ma * invN * (d_simᶜ + ε_sim′) + invCf * f_sim′,
    nsteps  = 350,
    rel_tol = 1e-15,
)

figure(figsize=(9,9))
subplot(2,2,1)
imshow(d_sim[:], vmin=-125,vmax=125, cmap = "RdBu_r"); colorbar(); title("simulated data (before extra noise)")
subplot(2,2,2)
imshow(f_sim_wf[:], cmap = "RdBu_r"); colorbar(); title("E(f|d)")
subplot(2,2,3)
imshow(f_sim_wfpsim[:], cmap = "RdBu_r"); colorbar(); title("simulate from P(f|d)")
subplot(2,2,4)
imshow((Ma * Tf* f_sim_wfpsim)[:], vmin=-125,vmax=125, cmap = "RdBu_r"); colorbar(); title("filter the conditional simulation ")

# Run on the data
@time f_data_wfpsim, hist_data = Mod.pcg(
    f -> pre*f,
    f -> A*f + B*f,
    Tf * Ma * invN * (dᶜ + ε_sim′) + invCf * f_sim′,
    nsteps  = 350,
    rel_tol = 1e-15,
)

figure(figsize=(10,7))
subplot(2,2,1)
imshow(d[:], vmin=-125,vmax=125, cmap = "RdBu_r"); colorbar(); title("data (before extra noise)")
subplot(2,2,2)
imshow(f_data_wf[:], cmap = "RdBu_r"); colorbar(); title("E(f|d)")
subplot(2,2,3)
imshow(f_data_wfpsim[:], cmap = "RdBu_r"); colorbar(); title("simulate from P(f|d)")
subplot(2,2,4)
imshow((Ma * Tf* f_data_wfpsim)[:], vmin=-125,vmax=125, cmap = "RdBu_r"); colorbar(); title("filter the conditional simulation ")

# residual magnitude
semilogy(hist_sim, label="pcg on the simulated data")
semilogy(hist_data, label="pcg on the data", "--")
legend()


