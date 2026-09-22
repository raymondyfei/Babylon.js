// Gaussian Point Splatting — workload scan, level 2. A single workgroup exclusive-scans the block
// sums in place, carrying a running offset across 512-wide chunks so it handles any number of
// blocks. Also writes the grand total point count and the splat kernel's indirect dispatch args.
#include<gaussianPointSplattingScan>

@group(0) @binding(0) var<storage, read_write> blockSums : array<u32>;
@group(0) @binding(1) var<storage, read_write> pointCount : array<u32>;
@group(0) @binding(2) var<storage, read_write> indirectArgs : array<u32>;

var<workgroup> wgCarry : u32;
var<workgroup> wgChunkTotal : u32;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid : vec3u) {
    let t = lid.x;
    let n = arrayLength(&blockSums);

    if (t == 0u) {
        wgCarry = 0u;
    }
    workgroupBarrier();

    var chunk = 0u;
    loop {
        if (chunk >= n) {
            break;
        }
        let i0 = chunk + 2u * t;
        let i1 = chunk + 2u * t + 1u;

        gpsScanTemp[2u * t] = select(0u, blockSums[i0], i0 < n);
        gpsScanTemp[2u * t + 1u] = select(0u, blockSums[i1], i1 < n);

        let total = gpsScanExclusive512(t);
        if (t == 0u) {
            wgChunkTotal = total;
        }
        workgroupBarrier();

        if (i0 < n) {
            blockSums[i0] = gpsScanTemp[2u * t] + wgCarry;
        }
        if (i1 < n) {
            blockSums[i1] = gpsScanTemp[2u * t + 1u] + wgCarry;
        }
        workgroupBarrier();

        if (t == 0u) {
            wgCarry = wgCarry + wgChunkTotal;
        }
        workgroupBarrier();

        chunk = chunk + 512u;
    }

    if (t == 0u) {
        // Tile the point dispatch across a 2D workgroup grid so the total can exceed WebGPU's
        // 65535-workgroups-per-dimension limit. A single row (groups > 65535) would either fail the
        // dispatch or, if clamped, drop the CDF tail and make whole splats vanish when zoomed in
        // (huge per-splat footprints -> enormous total point count). Y-tiling dispatches every point.
        let totalGroups = (wgCarry + 255u) / 256u;
        let gx = min(totalGroups, 65535u);
        let gy = (totalGroups + 65534u) / 65535u;
        pointCount[0] = wgCarry; // actual total; the splat kernel bounds-checks its linear index against it
        pointCount[1] = totalGroups; // CPU-readable diagnostics (indirectArgs is GPU-only)
        indirectArgs[0] = gx;
        indirectArgs[1] = gy;
        indirectArgs[2] = 1u;
    }
}
