DEFAULT_FREQ = 2.4f0 # GHz

function fspl(tx :: Point{2, Float32}, rx :: Point{2, Float32}; freq = DEFAULT_FREQ)
    # Calculate the FSPL given the tx and the rx
    freq = 5.9f0
    d = sqrt(sum((rx .- tx).^2))
    loss = 20 * log(10, freq) + 20 * log(10, d) + 32.45 # GHz
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