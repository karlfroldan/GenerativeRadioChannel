function fspl(tx :: Point{2, Float32}, rx :: Point{2, Float32}; freq = DEFAULT_FREQ)
    # Calculate the FSPL given the tx and the rx
    d = sqrt(sum((rx .- tx).^2))
    d_meters = d / PIXELS_PER_METER
    20 * log(10, freq) + 20 * log10(max(d_meters, 0.0001f0)) + 32.45 # GHz
end

function fspl(room :: Room{Float32}, rx :: Point{2, Float32}; freq = DEFAULT_FREQ)
    [fspl(tx, rx; freq=freq) for tx ∈ room.transmitters]
end


# Function to generate an FSPL map. 
# Essentially, it will iterate through all the discrete points 
# in the room and calculate the FSPL.
function fspl_map(room :: Room{Float32})
    x, y = Int.(floor.(room.dims))

    fmap = zeros(Float32, (x, y, length(room.transmitters)))
    for i ∈ 1:x
        for j ∈ 1:y
            fmap[i, j, :] = Float32.(fspl(room, Point2f(Float32(i), Float32(j))))
        end
    end

    fmap
end

# LINE INTERSECTION
function intersects(p1, p2, p3, p4)
    # Segment-Segment Intersection Test.
    # Ray: p1 -> p2
    # Wall (segment) : p3 -> p4

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    # parallel if denom is 0
    if denom == 0
        return false
    end

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    # 0 <= ua <= 1 intersects ray
    # 0 <= ub <= 1 intersects wall
    return (0 ≤ ua ≤ 1) && (0 ≤ ub ≤ 1)
end

function intersects(tx, rx, wall)
    p3, p4 = wall
    intersects(tx, rx, p3, p4)
end

# Count the number of walls that intersect the signal's path
function count_walls(room, tx, rx)
    sum([intersects(tx, rx, w) for w ∈ room.walls])
end

function count_walls(room, rx)
    [
        count_walls(room, tx, rx)
        for tx ∈ room.transmitters
    ]
end

function multiwall(
        room :: Room{Float32}, 
        tx :: Point{2, Float32}, 
        rx :: Point{2, Float32}; 
        freq=DEFAULT_FREQ,
        σ=0.0f0, # introduce shadowing
)
    X_σ = randn() * σ # log normal dist
    l_fspl = fspl(tx, rx; freq=freq)
    k = count_walls(room, tx, rx)
    l_fspl + 12k + X_σ
end

function multiwall(
        room :: Room{Float32}, 
        rx :: Point{2, Float32}; 
        freq=DEFAULT_FREQ,
        σ=0.0f0,
)
    [multiwall(room, tx, rx; freq=freq, σ=σ) for tx ∈ room.transmitters]
end

function multiwall_heatmap(
        room :: Room{Float32}; 
        freq=DEFAULT_FREQ,
        σ=0.0f0,
)
    x, y = Int.(floor.(room.dims))

    fmap = zeros(Float32, (x, y, length(room.transmitters)))
    for i ∈ 1:x
        for j ∈ 1:y
            fmap[i, j, :] = Float32.(multiwall(room, Point2f(Float32(i), Float32(j)); freq=DEFAULT_FREQ, σ=σ))
        end
    end

    fmap
end


### AI GENERATED
function generate_to_h5(filename, room, num_samples; batch_size=64, grid_size=(256, 256), σ=8.0f0, desc="Generate dataset")
    w_grid, h_grid = grid_size
    
    h5open(filename, "w") do fid
        dset_x = create_dataset(fid, "X", datatype(Float32), dataspace(w_grid, h_grid, 3, num_samples), 
                                chunk=(w_grid, h_grid, 3, 1), blosc=5)
        dset_y = create_dataset(fid, "Y", datatype(Float32), dataspace(w_grid, h_grid, 1, num_samples), 
                                chunk=(w_grid, h_grid, 1, 1), blosc=5)

        num_batches = ceil(Int, num_samples / batch_size)
        prog = Progress(num_samples, desc=desc)

        for b in 1:num_batches
            start_idx = (b - 1) * batch_size + 1
            end_idx = min(b * batch_size, num_samples)
            current_batch_size = end_idx - start_idx + 1

            # Temp buffers for batch
            batch_X = zeros(Float32, w_grid, h_grid, 3, current_batch_size)
            batch_Y = zeros(Float32, w_grid, h_grid, 1, current_batch_size)

            Threads.@threads for i in 1:current_batch_size
                # Place random transmitters
                random_txs = [Point2f(rand() * room.dims[1], rand() * room.dims[2]) for _ in 1:3]
                current_room = Room(room.dims[1], room.dims[2], room.walls, random_txs)
                batch_X[:, :, :, i] = rasterize_3channel(current_room, grid_size)
                hms = multiwall_heatmap(current_room; σ=σ)
                
                batch_Y[:, :, 1, i] = dropdims(minimum(hms, dims=3), dims=3)
            end

            # Write batch to disk
            dset_x[:, :, :, start_idx:end_idx] = batch_X
            dset_y[:, :, :, start_idx:end_idx] = batch_Y
            
            update!(prog, end_idx)
        end
    end
end
### END AI GENERATED BLOCK

function rasterize_3channel(r::Room, grid_size=(256, 256))
    w_grid, h_grid = grid_size
    wall_map = zeros(Float32, w_grid, h_grid)
    tx_dist_map = fill(Inf32, w_grid, h_grid)
    fspl_map = zeros(Float32, w_grid, h_grid)
    
    scale_x = w_grid / r.dims[1]
    scale_y = h_grid / r.dims[2]

    # CHANNEL 1: WALLS
    for wall in r.walls
        p1, p2 = wall
        dist = norm(p2 - p1)
        steps = Int(ceil(dist * max(scale_x, scale_y) * 2))
        for i in 0:steps
            t = i / steps
            p = p1 + t * (p2 - p1)
            idx_x = Int(clamp(floor(p[1] * scale_x) + 1, 1, w_grid))
            idx_y = Int(clamp(floor(p[2] * scale_y) + 1, 1, h_grid))
            wall_map[idx_x, idx_y] = 1.0 
        end
    end

    for x in 1:w_grid, y in 1:h_grid
        px, py = (x - 0.5f0) / scale_x, (y - 0.5f0) / scale_y
        pixel_point = Point2f(px, py)
        
        # CHANNEL 2: Distance
        min_d_pix = minimum([norm(pixel_point - tx) for tx in r.transmitters])
        d_meters = min_d_pix / PIXELS_PER_METER
        
        # CHANNEL 3: FSPL
        tx_dist_map[x, y] = d_meters
        fspl_map[x, y] = 20 * log10(DEFAULT_FREQ) + 20 * log10(max(d_meters, 0.1f0)) + 32.45f0
    end

    return cat(wall_map, tx_dist_map, fspl_map, dims=3)
end

function normalize_radio_data(obs)
    x, y = obs
    x_norm = copy(x)
    x_norm[:, :, 2, :] ./= 100.0f0 
    x_norm[:, :, 3, :] .= (x_norm[:, :, 3, :] .- 30.0f0) ./ 120.0f0
    
    y_norm = (y .- 30.0f0) ./ 120.0f0
    return x_norm, y_norm
end