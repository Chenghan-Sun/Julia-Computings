
module LocalMethods

using GSL
using LinearAlgebra
using Statistics
import JLD2

export cov_n1_dot_n2, cov_angle

const data_dir = "/Users/furinkazan/Box/STA_250/data/"
const cls = JLD2.FileIO.load(data_dir*"cls.jld2", "cls")
const w_on_Pl = cls[:cl_len_scalar][:,1] .* (2 .* cls[:ell] .+ 1) ./ (4π) ./ cls[:factor_on_cl_cmb]
const lmax = maximum(cls[:ell])

function cov_cosθ(cosθ,lmax::Int) 
	@assert lmax >= 2 "lmax too small, must be greater than 1"
	dot(sf_legendre_Pl_array(lmax, cosθ)[3:end], w_on_Pl[3:lmax+1])
end

cov_n1_dot_n2(n1_dot_n2) = cov_cosθ(n1_dot_n2,lmax) 
cov_angle(angle) = cov_cosθ(cos(angle),lmax) 

function bin_mean(fk; bin=0)
	fkrtn = copy(fk)
	if bin > 1
		Nm1    = length(fk)
		subNm1 = bin * (Nm1÷bin)
		fmk    = reshape(fk[1:subNm1], bin, Nm1÷bin) 
		fmk   .= mean(fmk, dims=1)
		fkrtn[1:subNm1] .= vec(fmk)
		fkrtn[subNm1+1:end] .= mean(fk[subNm1+1:end])
	end
	return fkrtn
end

end # module

using PyPlot
using PyCall
using LinearAlgebra
using Statistics
using Random
import JLD2
using FFTW
using SparseArrays
using ProgressMeter
using .LocalMethods

const prop_cycle = plt.rcParams["axes.prop_cycle"]
const colors     = prop_cycle.by_key()["color"]
const mplot3d    = pyimport("mpl_toolkits.mplot3d")

load_it = (jld2file, varname) -> JLD2.FileIO.load(LocalMethods.data_dir*jld2file, varname) 
θ_northp = load_it("data_slices.jld2", "θ_northp")
θ_southp = load_it("data_slices.jld2", "θ_southp")
cmb_northp = load_it("data_slices.jld2", "cmb_northp")
cmb_southp = load_it("data_slices.jld2", "cmb_southp")
φ_ref = load_it("data_slices.jld2", "φ_ref")
N = length(φ_ref)
Nside = 2048

# check dimensions
println(N)
N_θ_northp = length(θ_northp)
println(N_θ_northp)
N_cmb_northp = length(cmb_northp)
println(N_cmb_northp)

I_slice_1 = 1
I_slice_2 = 4
θ1 = θ_southp[I_slice_1]
θ2 = θ_southp[I_slice_2]
cmb1 = cmb_southp[I_slice_1]
cmb2 = cmb_southp[I_slice_2]

@show θ1, θ2
@show summary(cmb1)
@show summary(cmb2)
@show summary(φ_ref);

ofset_cmb = 9000 .* vcat(θ_northp, θ_southp) ./ π .- 0.5 |> reverse
figure(figsize=(10,5))
for i = 1:length(θ_northp)
    plot(φ_ref, cmb_northp[i] .+ ofset_cmb[i], "k", alpha = 0.1)
end

for i = 1:length(θ_southp)
    if i in [I_slice_1, I_slice_2]
        plot(φ_ref, cmb_southp[i] .+ ofset_cmb[i+length(θ_southp)])
    else
        plot(φ_ref, cmb_southp[i] .+ ofset_cmb[i+length(θ_southp)], "k", alpha = 0.1)
    end
end
# Note: The cmb is in units mK ≡ MicroKelvin (1 Kelvin = 1e+6 MicroKelvin)

function n̂(θ,φ)
    nx =  sin.(θ) .* cos.(φ)
    ny =  sin.(θ) .* sin.(φ)
    nz =  cos.(θ)
    return (nx, ny, nz)
end

@show n̂(θ1, φ_ref[1]);
@show n̂(θ2, φ_ref[1]);

# we use `Ref` here since `n̂(θ1, φ_ref[1])` is a single 3-tuple that we want
# broadcasted to the vector of 3-tuples `n̂.(θ1, φ_ref)`

dot.(n̂.(θ1, φ_ref), Ref(n̂(θ2, φ_ref[1])));

# slice 1 within group covariance to first pixel
Σ11 = dot.(n̂.(θ1, φ_ref), Ref(n̂(θ1, φ_ref[1]))) .|> cov_n1_dot_n2
# slice 2 within group covariance to first pixel
Σ22 = dot.(n̂.(θ2, φ_ref), Ref(n̂(θ2, φ_ref[1]))) .|> cov_n1_dot_n2
# slice 2 / slice 1 across group covariance to first pixel
Σ12 = dot.(n̂.(θ1, φ_ref), Ref(n̂(θ2, φ_ref[1]))) .|> cov_n1_dot_n2
# slice 1 / slice 2 across group covariance to first pixel
Σ21 = dot.(n̂.(θ2, φ_ref), Ref(n̂(θ1, φ_ref[1]))) .|> cov_n1_dot_n2;

figure(figsize=(10,4))
subplot(1,2,1)
plot(φ_ref[1:200], Σ11[1:200], label="slice 1")
plot(φ_ref[1:200], Σ22[1:200], "--", label="slice 2")
xlabel("azmuth lag")
title("within slice covariance")
legend()
subplot(1,2,2)
plot(φ_ref[1:200], Σ21[1:200], label="slice 1")
plot(φ_ref[1:200], Σ12[1:200], "--", label="slice 1")
xlabel("azmuth")
title("across slice cross-covariance")
legend()

# Here are the pixel and grid parameters
period = 2π
Δφ  = period / N  # pixel spacing
Δτ  = 2π / period # Fourier spacing
nyq = π / Δφ
τ_ref = Δτ .* (0:N-1) |> collect
τ_nyq = @. ifelse(τ_ref <= nyq, τ_ref, τ_ref - 2nyq)

cmb1;

# Here are the fft operators
𝒲   = plan_fft(similar(cmb1)) # unscaled fft
𝒰   = (1/sqrt(N)) * 𝒲         # unitary version
ℱ   = (Δφ / √(2π)) * 𝒲        # scaled fft which approximates the integral
scℱ = (√(2π)/Δτ) * ℱ          # scaling used for the eigenvalues from the first col of the circulantmatrix

# These model ℱ*cmb1 and ℱ*cmb2
scℱΣ11 = scℱ * Σ11 .|> real
scℱΣ22 = scℱ * Σ22 .|> real
scℱΣ21 = scℱ * Σ21
scℱΣ12 = scℱ * Σ12

ℱcmb1 = ℱ * cmb1
ℱcmb2 = ℱ * cmb2 #model, use for pred

let plt_range = 10:N÷2+1, 
	k = τ_ref[plt_range],
	km = k.^2,
	abs2ℱcmb1 = abs2.(ℱcmb1),
	abs2ℱcmb2 = abs2.(ℱcmb2),
	scℱΣ11 = scℱΣ11,
	scℱΣ22 = scℱΣ22


	for myplot ∈ [semilogx, loglog]

		figure(figsize=(10,4))
		sub = subplot(1,2,1)
			myplot(k, km .* abs2ℱcmb1[plt_range], label="abs2ℱcmb1", color=colors[1], ".", alpha=0.15)
			myplot(k, km .* scℱΣ11[plt_range], label="scℱΣ11", color=colors[2])
			xlabel("wavenumber")
			sub.set_ylim((km .* scℱΣ11[plt_range])[1], 2*maximum(km .* scℱΣ11[plt_range])) 
			sub.set_xlim(k[1], maximum(k)) 
			legend()
		sub = subplot(1,2,2)
			myplot(k, km .* abs2ℱcmb2[plt_range], label="abs2ℱcmb2", color=colors[1], ".", alpha=0.15)
			myplot(k, km .* scℱΣ22[plt_range], label="scℱΣ22", color=colors[2])
			xlabel("wavenumber")
			sub.set_ylim((km .* scℱΣ22[plt_range])[1], 2*maximum(km .* scℱΣ22[plt_range])) 
			sub.set_xlim(k[1], maximum(k)) 
			legend()

	end # for loop over myplot

end

# add more here
# Compute the standard deviation of white noise in each Healpix pixel. 
# Note: the Healpix pixel area ≈ 1.7² [arcmin²] with Nside = 2048.

# #############
#
#
# For the homework ... assign the variable `sd_of_noise_per_1p7_arcmin_pixel` the correct value.
# Note: the Healpix pixel area ≈ 1.7² [arcmin²] with Nside = 2048.
#
#
# ##############
pixel_area = 1.7^2
N_pixel = 3600 / pixel_area
Temp_sensitivity = 0.55
@show sd_of_noise_per_1p7_arcmin_pixel = Temp_sensitivity*(N_pixel)^0.5
bFWHM_arcmin = 7.3 # arcmin

ℱ𝗡ℱᴴ, ℱ𝗕 = let σn = sd_of_noise_per_1p7_arcmin_pixel, bfwhm = bFWHM_arcmin
 	𝗡 = fill(σn^2, size(τ_ref))
 	ℱ𝗡ℱᴴ = (Δφ / Δτ) .* 𝗡 ### note the scaling
 	bFWHM_rad = deg2rad(bfwhm/60)
 	ℱ𝗕 = @. exp( - (bFWHM_rad^2) * abs2(τ_nyq) / (16*log(2)) )
 	ℱ𝗡ℱᴴ, ℱ𝗕 
end

let bin_powers_width = 12,
	plt_range = 10:N÷2+1, 
	k = τ_ref[plt_range],
	km = k.^2, 
	abs2ℱcmb1 = abs2.(ℱcmb1),
	abs2ℱcmb2 = abs2.(ℱcmb2),
	scℱΣ11 = scℱΣ11,
	scℱΣ22 = scℱΣ22


	beam_scℱΣ11 = scℱΣ11 .* abs2.(ℱ𝗕)
	beam_scℱΣ22 = scℱΣ22 .* abs2.(ℱ𝗕)
	
	tot_scℱΣ11 = scℱΣ11 .* abs2.(ℱ𝗕) .+ ℱ𝗡ℱᴴ
	tot_scℱΣ22 = scℱΣ22 .* abs2.(ℱ𝗕) .+ ℱ𝗡ℱᴴ

	for myplot ∈ [semilogx, loglog]

		figure(figsize=(10,4))
		sub = subplot(1,2,1)
			myplot(k, 
				km .* abs2ℱcmb1[plt_range] |> x->LocalMethods.bin_mean(x; bin = bin_powers_width), 
				label="binned abs2ℱcmb1", color=colors[1], ".", alpha=0.15
			)
			myplot(k, km .* beam_scℱΣ11[plt_range], label="beam_scℱΣ11", color=colors[2])
			myplot(k, km .* ℱ𝗡ℱᴴ[plt_range], label="ℱNℱᴴ", "k", alpha = 0.8)
			myplot(k, km .* tot_scℱΣ11[plt_range], "--", label="tot_scℱΣ11", color=colors[7])
			xlabel("wavenumber")
			sub.set_ylim((km .* beam_scℱΣ11[plt_range])[1], 2*maximum(km .* beam_scℱΣ11[plt_range])) 
			sub.set_xlim(k[1], maximum(k)) 
			legend()
		sub = subplot(1,2,2)
			myplot(k, 
				km .* abs2ℱcmb2[plt_range] |> x->LocalMethods.bin_mean(x; bin = bin_powers_width), 
				label="binned abs2ℱcmb2", color=colors[1], ".", alpha=0.15
			)
			myplot(k, km .* beam_scℱΣ22[plt_range], label="beam_scℱΣ22", color=colors[2])
			myplot(k, km .* ℱ𝗡ℱᴴ[plt_range], label="ℱNℱᴴ", "k", alpha = 0.8)
			myplot(k, km .* tot_scℱΣ22[plt_range], "--", label="tot_scℱΣ22", color=colors[7])
			xlabel("wavenumber")
			sub.set_ylim((km .* beam_scℱΣ22[plt_range])[1], 2*maximum(km .* beam_scℱΣ22[plt_range])) 
			sub.set_xlim(k[1], maximum(k)) 
			legend()

	end # for loop over myplot
		
end

let bin_powers_width = 0,
	plt_range = 2:N÷2+1, 
	k = τ_ref[plt_range],
	km = k.^0, 
	crossℱcmb12 = ℱcmb1 .* conj(ℱcmb2),
	crossℱcmb21 = ℱcmb2 .* conj(ℱcmb1),
	scℱΣ12 = scℱΣ12,
	scℱΣ21 = scℱΣ21


	r_crossℱcmb =  (crossℱcmb12 .+ crossℱcmb21) ./ 2 .|> real # this must be real 
	i_crossℱcmb =  (crossℱcmb12 .- crossℱcmb21) ./ 2 .|> imag # this must be purely imaginary
	
	r_beam_cross_scℱΣ = real.((scℱΣ12 .+ scℱΣ21) ./ 2) .* abs2.(ℱ𝗕)
	i_beam_cross_scℱΣ = imag.((scℱΣ12 .- scℱΣ21) ./ 2) .* abs2.(ℱ𝗕)

	myplot = semilogx #plot
	figure(figsize=(10,4))
	sub = subplot(1,2,1)
		myplot(k, 
			km .* r_crossℱcmb[plt_range] |> x->LocalMethods.bin_mean(x; bin = bin_powers_width), 
			label="binned r_crossℱcmb", color=colors[1], ".", alpha=0.15
		)
		myplot(k, km .* r_beam_cross_scℱΣ[plt_range], label="r_beam_cross_scℱΣ", color=colors[2])
		xlabel("wavenumber")
		#sub.set_ylim((km .* r_beam_cross_scℱΣ[plt_range])[1], 2*maximum(km .* r_beam_cross_scℱΣ[plt_range])) 
		sub.set_xlim(k[2], maximum(k)) 
		legend()
	sub = subplot(1,2,2)
		myplot(k, 
			km .* i_crossℱcmb[plt_range] |> x->LocalMethods.bin_mean(x; bin = bin_powers_width), 
			label="binned i_crossℱcmb", color=colors[1], ".", alpha=0.15
		)
		myplot(k, km .* i_beam_cross_scℱΣ[plt_range], label="i_beam_cross_scℱΣ", color=colors[2])
		xlabel("wavenumber")
		#sub.set_ylim((km .* i_beam_cross_scℱΣ[plt_range])[1], 2*maximum(km .* i_beam_cross_scℱΣ[plt_range])) 
		sub.set_xlim(k[2], maximum(k)) 
		legend()

		
end

tot_scℱΣ11 = scℱΣ11 .* abs2.(ℱ𝗕) .+ ℱ𝗡ℱᴴ
tot_scℱΣ22 = scℱΣ22 .* abs2.(ℱ𝗕) .+ ℱ𝗡ℱᴴ
tot_scℱΣ21 = scℱΣ21 .* abs2.(ℱ𝗕) .|> real

λ = tot_scℱΣ21 ./ sqrt.( tot_scℱΣ11 .* tot_scℱΣ22)
nrm_ℱcmb1 = ℱcmb1 ./ sqrt.(tot_scℱΣ11)
nrm_ℱcmb2 = ℱcmb2 ./ sqrt.(tot_scℱΣ22)

z1 = nrm_ℱcmb1 .+ exp.(im.*angle.(λ)) .* nrm_ℱcmb2
z2 = nrm_ℱcmb1 .- exp.(im.*angle.(λ)) .* nrm_ℱcmb2
λ1 = 1 .- abs.(λ)
λ2 = 1 .+ abs.(λ)

let bin_powers_width = 4,
	plt_range = 10:N÷2+1, 
	k = τ_ref[plt_range],
	abs2_z1 = abs2.(z1),
	abs2_z2 = abs2.(z2),
	cross_z2_z1 = z2 .* conj.(z1),
	λ1 = λ1,
	λ2 = λ2

	figure(figsize=(10,4))
		sub = subplot(1,2,1)
		plot(
			k, 
			abs2_z1[plt_range] |> x->LocalMethods.bin_mean(x; bin = bin_powers_width), 
			label="binned abs2(z1)", color=colors[1], ".", alpha=0.15
		)
		plot(k, λ1[plt_range], label="λ1", color=colors[2])

		sub = subplot(1,2,2)
		plot(
			k, 
			abs2_z2[plt_range] |> x->LocalMethods.bin_mean(x; bin = bin_powers_width), 
			label="binned abs2(z2)", color=colors[1], ".", alpha=0.15
		)
		plot(k, λ2[plt_range], label="λ2", color=colors[2])
		legend()


	figure(figsize=(10,4))
		plot(
			k, 
			real.(cross_z2_z1)[plt_range] |> x->LocalMethods.bin_mean(x; bin = bin_powers_width), 
			label="real(z1 * conj(z2))", color=colors[1], ".", alpha=0.15
		)
		plot(
			k, 
			imag.(cross_z2_z1)[plt_range] |> x->LocalMethods.bin_mean(x; bin = bin_powers_width), 
			label="imag(z1 * conj(z2))", color=colors[2], ".", alpha=0.15
		)
		plot(k, fill(0, size(k)), "k", alpha = 0.8)
		legend()


end

# #############
#
#
# For the homework ... 
# Create a function `wf_cmb_fun(θ)` that takes in a polar angle `θ` and
# returns the conditional expected value of `cmb` at the azmuth grid values `φ_ref`.
#
#
# ##############

# From Prof: @show sp_tot_scℱΣ11 = spdiagm(0=>tot_scℱΣ11) 

# Construct the four elements of the sparse diagonal block matrix
sp_tot_scℱΣ11 = spdiagm(0=>tot_scℱΣ11)
sp_tot_scℱΣ12 = spdiagm(0=>tot_scℱΣ21) # by the symmetric covariance matrix
sp_tot_scℱΣ21 = spdiagm(0=>tot_scℱΣ21)
sp_tot_scℱΣ22 = spdiagm(0=>tot_scℱΣ22);

function wf_cmb_fun(θ)
    stk_cmb_obs = [ℱcmb1; ℱcmb2] # CMB data stacking
    tot_scℱΣ_inv = [sp_tot_scℱΣ11 sp_tot_scℱΣ12; sp_tot_scℱΣ21 sp_tot_scℱΣ22] \ stk_cmb_obs #invert * CMB data
    
    Σ31 = dot.(n̂.(θ, φ_ref), Ref(n̂(θ1, φ_ref[1]))) .|> cov_n1_dot_n2 # index 3 refers to CMB new observations
    Σ32 = dot.(n̂.(θ, φ_ref), Ref(n̂(θ2, φ_ref[1]))) .|> cov_n1_dot_n2
    scℱΣ31 = scℱ * Σ31 # apply FFT
    scℱΣ32 = scℱ * Σ32
    
    sp_scℱΣ31 = spdiagm(0=>scℱΣ31) # construct sparse diagnoal matrix
    sp_scℱΣ32 = spdiagm(0=>scℱΣ32)
    cmb_new_expect = ℱ \ ([sp_scℱΣ31 sp_scℱΣ32] * tot_scℱΣ_inv) # calculate expectation of ℱcmb new observations
    return cmb_new_expect.|>real
end

# if at north pole
#θs = range(0.001, max(θ2, θ1) .+ 0.1,   length=250) |> transpose

# if at south pole
θs = range(min(θ2, θ1) .- 0.1, π,  length=150) |> transpose

θgrid =      θs .+ 0 .* φ_ref
φgrid = 0 .* θs .+      φ_ref


function make_wf_cmb_mat(θs, φ_ref)
	wf_cmb_mat_local = 0 .* θs .+ 0 .* φ_ref
	@showprogress for icol = 1:length(θs)
		#print("$icol,")
		wf_cmb_mat_local[:,icol] = wf_cmb_fun(θs[icol])
	end
	wf_cmb_mat_local
end

wf_cmb_mat = make_wf_cmb_mat(θs, φ_ref)

# reproduce this figure 

fig = figure()
ax  = mplot3d.Axes3D(fig)
subplot(projection="polar")

# if at north pole
# pcolormesh(φgrid, θgrid, wf_cmb_mat, vmin=-125, vmax=225)

# if at south pole
pcolormesh(φgrid, π .- θgrid, wf_cmb_mat, vmin=-135, vmax=235, cmap = "RdBu_r") # use the same plotting color 
# grid()
