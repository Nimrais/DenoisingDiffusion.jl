using Plots
using Flux
using Dates
using BSON, JSON
using Printf

using DenoisingDiffusion
using DenoisingDiffusion: train!, batched_loss
using Random

Random.seed!(42)

# Extend conditioning helper locally for matrix labels (C x B)
import DenoisingDiffusion: randomly_set_unconditioned
function randomly_set_unconditioned(labels::AbstractMatrix{<:Real}; prob_uncond::Float64=0.20)
    labels = copy(labels)
    B = size(labels, 2)
    mask = rand(B) .<= prob_uncond
    if any(mask)
        labels[:, mask] .= 0
    end
    labels
end
include("datasets.jl")
include("utilities.jl")

### settings
directory = joinpath("outputs", "2d_cond_" * Dates.format(now(), "yyyymmdd_HHMM"))
num_timesteps = 100
n_batch = 9_000
to_device = cpu
d_hid = 64
num_epochs = 100
num_classes = 3
prob_uncond = 0.2


### data
nsamples_per_class = round(Int, n_batch / num_classes)
X1 = normalize_neg_one_to_one(make_spiral(nsamples_per_class));
X2 = normalize_neg_one_to_one(make_s_curve(nsamples_per_class));
X3 = normalize_neg_one_to_one(make_moons(nsamples_per_class));

X = hcat(X1, X2, X3)
# label = 1 is for any/unguided diffusion
labels = 1 .+ vcat(fill(1, nsamples_per_class), fill(2, nsamples_per_class), fill(3, nsamples_per_class))

# map integer labels to float condition vectors (C x B), where
# label==1 -> unconditional (all zeros), label>1 -> one-hot at (label-1)
function labels_to_cond_vectors(labels::AbstractVector{<:Integer}, num_classes::Int)
    B = length(labels)
    C = num_classes
    Y = zeros(Float32, C, B)
    @inbounds for (j, lbl) in enumerate(labels)
        if lbl > 1
            idx = lbl - 1
            @assert 1 <= idx <= C
            Y[idx, j] = 1.0f0
        end
    end
    Y
end

n_val = floor(Int, 0.1 * nsamples_per_class)
X1_val = normalize_neg_one_to_one(make_spiral(n_val));
X2_val = normalize_neg_one_to_one(make_s_curve(n_val));
X3_val = normalize_neg_one_to_one(make_moons(n_val));

X_val = hcat(X1_val, X2_val, X3_val)
labels_val = 1 .+ vcat(fill(1, n_val), fill(2, n_val), fill(3, n_val))

# build conditioning matrices
Y = labels_to_cond_vectors(labels, num_classes)
Y_val = labels_to_cond_vectors(labels_val, num_classes)

### model
model = ConditionalChain(
    Parallel(
        .+,
        Dense(2, d_hid),
        Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid),
            Dense(d_hid, d_hid)),
        Dense(num_classes, d_hid)
        #Embedding(1 + num_classes => d_hid)
    ),
    swish,
    Parallel(
        .+,
        Dense(d_hid, d_hid),
        Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid), Dense(d_hid, d_hid)),
        Dense(num_classes, d_hid)
        #Embedding(1 + num_classes => d_hid)
    ),
    swish,
    Parallel(
        .+,
        Dense(d_hid, d_hid),
        Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid),
            Dense(d_hid, d_hid)),
        Dense(num_classes, d_hid)
        #Embedding(1 + num_classes => d_hid)
    ),
    swish,
    Parallel(
        .+,
        Dense(d_hid, d_hid),
        Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid),
            Dense(d_hid, d_hid)),
        Dense(num_classes, d_hid)
        #Embedding(1 + num_classes => d_hid)
    ),
    swish,
    Dense(d_hid, 2),
)
display(model)

βs = linear_beta_schedule(num_timesteps, 8e-6, 9e-5)
diffusion = GaussianDiffusion(Vector{Float32}, βs, (2,), model)

#### train
diffusion = diffusion |> to_device

train_data = Flux.DataLoader((X, Y) |> to_device; batchsize=32, shuffle=true);
val_data = Flux.DataLoader((X_val, Y_val) |> to_device; batchsize=32, shuffle=false);
loss_type = Flux.mse;
loss(diffusion, x::AbstractArray, y::AbstractArray) = p_losses(diffusion, loss_type, x, y; to_device=to_device)
opt = Adam(0.001);

println("Calculating initial loss")
val_loss = batched_loss(loss, diffusion, val_data; prob_uncond=prob_uncond)
@printf("\nval loss: %.5f\n", val_loss)

mkpath(directory)
output_path = joinpath(directory, "diffusion.bson")
history_path = joinpath(directory, "history.json")
hyperparameters_path = joinpath(directory, "hyperparameters.json")

hyperparameters = Dict(
    "num_timesteps" => num_timesteps,
    "data_shape" => "$(diffusion.data_shape)",
    "denoise_fn" => "$(typeof(model).name.wrapper)",
    "parameters" => sum(length, Flux.params(model)),
    "loss_type" => "$loss_type",
    "d_hid" => d_hid,
    "num_classes" => num_classes,
    "prob_uncond" => prob_uncond,
)
open(hyperparameters_path, "w") do f
    JSON.print(f, hyperparameters)
end
println("saved hyperparameters to $hyperparameters_path")

println("training")
start_time = time_ns()
opt_state = Flux.setup(opt, diffusion)
history = train!(
    loss, diffusion, train_data, opt_state, val_data;
    num_epochs=num_epochs, prob_uncond=prob_uncond
)
end_time = time_ns() - start_time
println("\ndone training")
@printf "time taken: %.2fs\n" end_time / 1e9

### save results
diffusion = diffusion |> cpu
BSON.bson(output_path, Dict(:diffusion => diffusion))
println("saved model to $output_path")

open(history_path, "w") do f
    JSON.print(f, history)
end
println("saved history to $history_path")

### plot results
diffusion = diffusion |> cpu

canvas_train = plot(
    1:length(history["mean_batch_loss"]), history["mean_batch_loss"], label="mean batch loss",
    xlabel="epoch",
    ylabel="loss",
    legend=:right, # :best, :right
    ylims=(0, Inf),
)
plot!(canvas_train, 1:length(history["val_loss"]), history["val_loss"], label="validation loss")
savefig(canvas_train, joinpath(directory, "history.png"))
display(canvas_train)

canvases = []
for label in 1:4
    # build batch conditioning matrix for sampling
    B = 1000
    cond = zeros(Float32, num_classes, B)
    if label > 1
        #cond[label-1, :] .= 0.3
        cond[1, :] .= 1.0
        cond[2, :] .= 1.0
        cond[3, :] .= 1.0
        @show size(cond)
        #cond = [1.0, 0.0, 0.0]
    end
    X0 = p_sample_loop(diffusion, cond; guidance_scale=1.0f0)
    p0 = scatter(X0[1, :], X0[2, :], alpha=0.5, label="",
        aspectratio=:equal,
        xlims=(-2, 2), ylims=(-2, 2),
        title="label=$label"
    )
    push!(canvases, p0)
end
canvas_samples = plot(canvases...)
savefig(canvas_samples, joinpath(directory, "samples_111.png"))
display(canvas_samples)

println("press enter to finish")
readline()
