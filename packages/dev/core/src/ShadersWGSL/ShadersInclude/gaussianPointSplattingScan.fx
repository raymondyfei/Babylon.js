// Work-efficient (Blelloch) exclusive prefix scan of a 512-element block held in workgroup memory.
// Shared by the block scan and the block-sums scan. 256 threads, 2 elements per thread.
//
// Callers fill gpsScanTemp[0..511] before calling, then read the exclusive results back from it. The
// return value (the block's total sum) is valid on thread 0 only.

var<workgroup> gpsScanTemp : array<u32, 512>;

fn gpsScanExclusive512(t : u32) -> u32 {
    var offset = 1u;
    // Up-sweep (reduce): build partial sums up the tree.
    for (var d = 256u; d > 0u; d = d >> 1u) {
        workgroupBarrier();
        if (t < d) {
            let ai = offset * (2u * t + 1u) - 1u;
            let bi = offset * (2u * t + 2u) - 1u;
            gpsScanTemp[bi] = gpsScanTemp[bi] + gpsScanTemp[ai];
        }
        offset = offset << 1u;
    }

    var total = 0u;
    if (t == 0u) {
        total = gpsScanTemp[511];
        gpsScanTemp[511] = 0u; // clear the last element for the exclusive down-sweep
    }

    // Down-sweep: distribute the partial sums back down the tree.
    for (var d = 1u; d < 512u; d = d << 1u) {
        offset = offset >> 1u;
        workgroupBarrier();
        if (t < d) {
            let ai = offset * (2u * t + 1u) - 1u;
            let bi = offset * (2u * t + 2u) - 1u;
            let s = gpsScanTemp[ai];
            gpsScanTemp[ai] = gpsScanTemp[bi];
            gpsScanTemp[bi] = gpsScanTemp[bi] + s;
        }
    }
    workgroupBarrier();

    return total;
}
