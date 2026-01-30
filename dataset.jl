### AI_GENERATED

struct LazyHDF5Dataset
    file::HDF5.File
    X::HDF5.Dataset
    y::HDF5.Dataset
    n_samples::Int
end

function LazyHDF5Dataset(filepath::String)
    # Open in read-only mode
    fid = h5open(filepath, "r")
    X = fid["X"]
    y = fid["Y"]
    n_samples = size(X, 4)
    return LazyHDF5Dataset(fid, X, y, n_samples)
end

# Implement the interface for DataLoader
Base.length(d::LazyHDF5Dataset) = d.n_samples
Base.getindex(d::LazyHDF5Dataset, i::Int) = (d.X[:, :, :, i], d.y[:, :, :, i])
Base.getindex(d::LazyHDF5Dataset, i::AbstractVector{Int}) = (d.X[:, :, :, i], d.y[:, :, :, i])

function Base.getindex(d::LazyHDF5Dataset, idxs::AbstractVector{Int})
    # HDF5 can't read [1, 5, 2] directly.
    # We must read them one by one and stack them.
    
    # Read individual samples (returns a list of 3D arrays)
    x_batch_list = [d.X[:, :, :, i] for i in idxs]
    y_batch_list = [d.y[:, :, :, i] for i in idxs]
    
    # Stack them along the 4th dimension (Batch dimension)
    # Result is (256, 256, 3, BatchSize)
    X_batch = cat(x_batch_list..., dims=4)
    y_batch = cat(y_batch_list..., dims=4)
    
    return (X_batch, y_batch)
end

# 3. Close method (good practice)
Base.close(d::LazyHDF5Dataset) = close(d.fid)
### END AI GENERATED 

# Function to calculate the statistics of a dataset to be used for normalization.
function calculate_stats(data::LazyHDF5Dataset; step=64)
    x_mins = fill(Inf32, 3)
    x_maxs = fill(-Inf32, 3)
    
    y_min = Inf32
    y_max = -Inf32

    @showprogress for i in 1:step:length(data)
        x_sample, y_sample = data[i] 

        for c in 1:3
            c_min = minimum(x_sample[:, :, c])
            c_max = maximum(x_sample[:, :, c])
            
            x_mins[c] = min(x_mins[c], c_min)
            x_maxs[c] = max(x_maxs[c], c_max)
        end

        y_min = min(y_min, minimum(y_sample))
        y_max = max(y_max, maximum(y_sample))
    end
    
    return x_mins, x_maxs, y_min, y_max
end

### AI GENERATED 
to_device(x, stats) = x isa AbstractGPUArray ? gpu(stats) : stats

function normalize_data(x, y, x_min, x_max, y_min, y_max)
    # Reshape stats for broadcasting: (1, 1, Channels, 1)
    min_x_b = reshape(x_min, 1, 1, 3, 1)
    max_x_b = reshape(x_max, 1, 1, 3, 1)

    # CRITICAL FIX: Move stats to GPU if x is on GPU
    min_x_b = to_device(x, min_x_b)
    max_x_b = to_device(x, max_x_b)
    
    # Epsilon to avoid division by zero
    ϵ = 1f-6
    
    # Now this broadcast happens entirely on the GPU (fast!)
    x_n = (x .- min_x_b) ./ (max_x_b .- min_x_b .+ ϵ)
    y_n = (y .- y_min) ./ (y_max - y_min + ϵ)

    return x_n, y_n
end
### END AI GENERATED 

function denormalize_y(y_pred, y_min, y_max)
    return y_pred .* (y_max - y_min) .+ y_min
end