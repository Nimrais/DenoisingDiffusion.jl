using Plots
using Flux
using Dates
using BSON, JSON
using Printf

using DenoisingDiffusion
using DenoisingDiffusion: train!, batched_loss
using Random

function randomly_set_unconditioned(
    labels::AbstractMatrix{Float32}; prob_uncond::Float64=0.20
)
    # Zero full label columns with probability prob_uncond (device-safe)
    L = copy(labels)
    b = size(L, 2)
    r = similar(L, 1, b)
    rand!(r)
    keep = r .> prob_uncond
    L .= L .* Float32.(keep)
    L
end


Random.seed!(42)

include("datasets.jl")
include("utilities.jl")

### settings
directory = joinpath("outputs", "2d_cond_" * Dates.format(now(), "yyyymmdd_HHMM"))
num_timesteps = 40
n_batch = 9_000
to_device = cpu
d_hid = 32
num_epochs = 100
num_classes = 3
prob_uncond = 0.2

### data
nsamples_per_class = round(Int, n_batch / num_classes)
X1 = normalize_neg_one_to_one(make_spiral(nsamples_per_class));
X2 = normalize_neg_one_to_one(make_s_curve(nsamples_per_class));
X3 = normalize_neg_one_to_one(make_moons(nsamples_per_class));

X = Float32.(hcat(X1, X2, X3))
# one-hot Float32 class labels (num_classes, batch); zero vector = unguided
labels_ids = vcat(fill(1, nsamples_per_class), fill(2, nsamples_per_class), fill(3, nsamples_per_class))
labels = Float32.(Matrix(Flux.onehotbatch(labels_ids, 1:num_classes)))

n_val = floor(Int, 0.1 * nsamples_per_class)
X1_val = normalize_neg_one_to_one(make_spiral(n_val));
X2_val = normalize_neg_one_to_one(make_s_curve(n_val));
X3_val = normalize_neg_one_to_one(make_moons(n_val));

X_val = Float32.(hcat(X1_val, X2_val, X3_val))
labels_val_ids = vcat(fill(1, n_val), fill(2, n_val), fill(3, n_val))
labels_val = Float32.(Matrix(Flux.onehotbatch(labels_val_ids, 1:num_classes)))

### model
model = ConditionalChain(
    Parallel(
        .+,
        Dense(2, d_hid),
        Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid),
            Dense(d_hid, d_hid)),
        Dense(num_classes, d_hid)
    ),
    swish,
    Parallel(
        .+,
        Dense(d_hid, d_hid),
        Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid), Dense(d_hid, d_hid)),
        Dense(num_classes, d_hid)
    ),
    swish,
    Parallel(
        .+,
        Dense(d_hid, d_hid),
        Chain(SinusoidalPositionEmbedding(num_timesteps, d_hid),
            Dense(d_hid, d_hid)),
        Dense(num_classes, d_hid)
    ),
    swish,
    Dense(d_hid, 2),
)
display(model)

βs = linear_beta_schedule(num_timesteps, 8e-6, 9e-5)
diffusion = GaussianDiffusion(Vector{Float32}, βs, (2,), model)

#### train
diffusion = diffusion |> to_device

train_data = Flux.DataLoader((X, labels) |> to_device; batchsize=32, shuffle=true);
val_data = Flux.DataLoader((X_val, labels_val) |> to_device; batchsize=32, shuffle=false);
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
labels_unguided = zeros(Float32, num_classes, 1000)
X0 = p_sample_loop(diffusion, labels_unguided; guidance_scale=1.0f0)
p0 = scatter(X0[1, :], X0[2, :], alpha=0.5, label="",
    aspectratio=:equal,
    xlims=(-2, 2), ylims=(-2, 2),
    title="unguided"
)
push!(canvases, p0)

for k in 1:num_classes
    labels_k = zeros(Float32, num_classes, 1000)
    labels_k[k, :] .= 0.5
    X0 = p_sample_loop(diffusion, labels_k; guidance_scale=1.0f0)
    p0 = scatter(X0[1, :], X0[2, :], alpha=0.5, label="",
        aspectratio=:equal,
        xlims=(-2, 2), ylims=(-2, 2),
        title="label=$k"
    )
    push!(canvases, p0)
end
canvas_samples = plot(canvases...)
savefig(canvas_samples, joinpath(directory, "samples.png"))
display(canvas_samples)

println("press enter to finish")
readline()
