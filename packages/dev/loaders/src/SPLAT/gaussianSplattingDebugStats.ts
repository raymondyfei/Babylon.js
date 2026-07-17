/* eslint-disable no-console */
/**
 * TEMPORARY debug instrumentation for the Windows/D3D12 "streamed GS renders 1 FPS on a static camera while
 * the non-streamed path renders 60 FPS" investigation. Every log line is tagged `[GS-DEBUG]` and stamped with
 * a monotonically increasing frame index plus the wall-clock delta since the previous `_onLodFrame` call, so
 * lines from different files/call sites can be correlated back to the same frame.
 *
 * Delete this file and every `gsDebugLog`/`gsDebugFrameTick` call site once the root cause is confirmed.
 */

let FrameIndex = 0;
let LastFrameTime = 0;

/**
 * Call once per `_onLodFrame` invocation. Returns the current frame index and the time (ms) since the
 * previous call, so callers can log both alongside their own event data.
 * @returns frame index and delta-time in milliseconds since the previous tick
 */
export function GsDebugFrameTick(): { frame: number; dt: number } {
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    const dt = LastFrameTime ? now - LastFrameTime : 0;
    LastFrameTime = now;
    FrameIndex++;
    return { frame: FrameIndex, dt };
}

/**
 * Logs a tagged, frame-stamped debug line to the console.
 * @param tag short event name (e.g. "onLodFrame", "workBuffer.relayoutSync")
 * @param data arbitrary structured payload to log alongside the tag
 */
export function GsDebugLog(tag: string, data: Record<string, unknown> = {}): void {
    console.log(`[GS-DEBUG][frame=${FrameIndex}][${tag}]`, JSON.stringify(data));
}
