@group(0) @binding(0)
var<storage,read> time: f32;

@group(0) @binding(1)
var<storage,read> dx: f32;

@group(0) @binding(2)
var<storage,read> time_sample: f32;

@group(0) @binding(3)
var<storage,read> acoustic_speed: f32;

@group(0) @binding(4)
var<storage,read_write> delays: array<i32>;

@group(0) @binding(5)
var<storage,read_write> image: array<f32>;

@group(0) @binding(6)
var<storage,read> fmc: array<f32>;

@group(0) @binding(7)
var<storage,read> depth_length: i32;

@group(0) @binding(8)
var<storage,read> transducer_length: i32;

@group(0) @binding(9)
var<storage,read> gate_start_frames: f32;

// 2D index to 1D index
fn zx(z: i32, x: i32) -> i32 {
    let index = z + x * depth_length;

    return select(-1, index, x >= 0 && x < transducer_length && z >= 0 && z < depth_length);
}

// 3D index to 1D index
fn zxy(z: i32, x: i32, y: i32) -> i32 {
    let index = z + x * depth_length + y * depth_length * transducer_length;

    return select(-1, index, z >= 0 && z < depth_length && x >= 0 && x < transducer_length && y >= 0 && y < transducer_length);
}

fn zxy_fmc(z: i32, x: i32, y: i32) -> i32 {
    let index = z * transducer_length * transducer_length + x * transducer_length + y;

    return select(-1, index, z >= 0 && z < depth_length && x >= 0 && x < transducer_length && y >= 0 && y < transducer_length);
}

@compute
@workgroup_size(wsReceptor, wsDepth, wsLength)
fn create_delays(@builtin(global_invocation_id) index: vec3<u32>) {
    let receptor: i32 = i32(index.x);
    let depth: i32 = i32(index.y);
    let length: i32 = i32(index.z);

    var distance: f32 = 0.;

    distance = pow(f32(depth) * dx, 2.) + pow(f32(length - receptor) * dx, 2.);
    distance = pow(distance, 0.5);

    delays[zxy(depth, length, receptor)] = i32(round((distance / acoustic_speed / time_sample) - (gate_start_frames / 2.)));
}

@compute
@workgroup_size(wsDepth, wsLength)
fn sim_tfm(@builtin(global_invocation_id) index: vec3<u32>) {
    let depth: i32 = i32(index.x);
    let length: i32 = i32(index.y);

    var delay: i32 = 0;

    for (var source: i32 = 0; source < transducer_length; source += 1)
    {
        for (var receptor: i32 = 0; receptor < transducer_length; receptor += 1)
        {
            delay = delays[zxy(depth, length, receptor)] + delays[zxy(depth, length, source)];
            if (delay < depth_length)
            {
                image[zx(depth, length)] += fmc[zxy_fmc(delay, source, receptor)];
            }
        }
    }
}
