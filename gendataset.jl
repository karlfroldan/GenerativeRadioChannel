using Plots 
using Printf: @printf, @sprintf 
using ProgressMeter 

using Serialization: serialize

using HDF5
using H5Zblosc

include("map.jl")
include("signals.jl")

const DEFAULT_FREQ = 2.4f0 # GHz
const PIXELS_PER_METER = 3.0f0

num_threads = Threads.nthreads()
println("Number of threads: $(num_threads)")

train_samples = 10000 
val_samples = 500 
test_samples = 500
room = loadroom("FloorPlan.svg")

# Replace your current generate_dataset calls with:
generate_to_h5("dataset/train.h5", room, train_samples; desc="Generating training set")
generate_to_h5("dataset/val.h5", room, val_samples; desc="Generating validation set")
generate_to_h5("dataset/test.h5", room, test_samples; desc="Generating test set")
