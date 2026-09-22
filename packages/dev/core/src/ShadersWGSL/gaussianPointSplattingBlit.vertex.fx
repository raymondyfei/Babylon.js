// Fullscreen triangle that composites the Gaussian Point Splatting accumulation buffer into the
// scene. Positions arrive already in clip space, so the camera's view/projection is ignored here.
// A screen-space UV is passed to the fragment so it can index the accumulation buffer independently
// of the render target's pixel resolution (which may differ from the compute resolution, e.g. on a
// high-DPI display or when the viewer renders through a scaled target).
attribute position: vec3f;

varying vScreenUv: vec2f;

#define CUSTOM_VERTEX_DEFINITIONS

@vertex
fn main(input : VertexInputs) -> FragmentInputs {

#define CUSTOM_VERTEX_MAIN_BEGIN

    vertexOutputs.position = vec4f(vertexInputs.position.xy, 0.0, 1.0);
    vertexOutputs.vScreenUv = vertexInputs.position.xy * 0.5 + vec2f(0.5, 0.5);

#define CUSTOM_VERTEX_MAIN_END
}
