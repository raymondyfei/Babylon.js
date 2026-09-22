/** This file must only contain pure code and pure imports */

import { type Scene } from "core/scene.pure";
import { type Nullable } from "core/types";
import { type Observer } from "core/Misc/observable";
import { type Camera } from "core/Cameras/camera.pure";
import { Matrix, Quaternion, Vector3 } from "core/Maths/math.vector.pure";
import { Mesh } from "../mesh.pure";
import { VertexData } from "../mesh.vertexData";
import { Logger } from "core/Misc/logger";
import { GaussianPointSplattingRenderer } from "./gaussianPointSplattingRenderer.pure";
import { GaussianPointSplattingBlitMaterial } from "core/Materials/GaussianSplatting/gaussianPointSplattingBlitMaterial.pure";

const SplatRowStride = 32; // bytes: 3f position + 3f scale + 4 u8 color + 4 u8 quaternion

/** Options controlling how raw splat data is interpreted. */
export interface IGaussianPointSplattingUpdateOptions {
    /** Flip the Y axis of positions (right-handed source data). Defaults to false. */
    flipY?: boolean;
    /** World transform to bake into the splats (means + covariance), matching the source mesh's node matrix. */
    worldMatrix?: Matrix;
}

/**
 * Renders a Gaussian splat scene with the stochastic, sort-free "point splatting" technique on
 * WebGPU. Unlike the classic {@link GaussianSplattingMesh} (sorted alpha-blended quads), this drives
 * a compute pipeline that stochastically rasterizes each Gaussian into pixel-sized point samples,
 * resolves visibility with a per-pixel atomic depth-min, and converges over frames.
 *
 * The mesh itself is only a fullscreen triangle carrying a blit material; the heavy lifting lives in
 * {@link GaussianPointSplattingRenderer}, dispatched before each camera's render pass (compute cannot
 * run inside an active render pass).
 */
export class GaussianPointSplattingMesh extends Mesh {
    private _renderer: GaussianPointSplattingRenderer;
    private _blitMaterial: GaussianPointSplattingBlitMaterial;
    private _beforeRenderObserver: Nullable<Observer<Scene>> = null;
    private _splatCount = 0;
    private readonly _vpMatrix = new Matrix();
    // World-space AABB of the decoded means, used to normalize the depth-sort key to the model's own
    // NDC-z span each frame (see _runCompute / the renderer's depthNorm uniform).
    private readonly _boundsMin = new Vector3(0, 0, 0);
    private readonly _boundsMax = new Vector3(0, 0, 0);
    // Rows of inverse(world 3x3); transforms the SH view dir to splat-local space. Identity by default.
    private readonly _invWorldRot = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

    /**
     * Creates a new Gaussian Point Splatting mesh.
     * @param name mesh name
     * @param scene hosting scene (its engine must be WebGPU)
     */
    constructor(name: string, scene: Scene) {
        super(name, scene);

        const engine = scene.getEngine();
        if (!engine.isWebGPU) {
            Logger.Warn("GaussianPointSplattingMesh requires a WebGPU engine; nothing will be rendered.");
        }

        // Fullscreen triangle in clip space. The blit vertex shader passes these straight through,
        // so the mesh's world/view/projection is irrelevant and it must not be frustum-culled.
        const vertexData = new VertexData();
        vertexData.positions = [-1, -1, 0, 3, -1, 0, -1, 3, 0];
        vertexData.indices = [0, 1, 2];
        vertexData.applyToMesh(this);
        this.alwaysSelectAsActiveMesh = true;
        this.isPickable = false;
        this.doNotSyncBoundingInfo = true;

        this._renderer = new GaussianPointSplattingRenderer(engine);
        this._blitMaterial = new GaussianPointSplattingBlitMaterial(`${name}_blit`, scene);
        this.material = this._blitMaterial;

        // Run the compute before the scene's render pass each frame. onBeforeRenderObservable is fired
        // by the standard render loop and by the babylon-viewer host, unlike onBeforeCameraRender.
        this._beforeRenderObserver = scene.onBeforeRenderObservable.add(() => {
            const camera = this.getScene().activeCamera;
            if (camera) {
                this._runCompute(camera);
            }
        });
    }

    /** Number of splats currently loaded. */
    public get splatCount(): number {
        return this._splatCount;
    }

    /** Global density multiplier: higher spends more points per Gaussian (denser, slower). */
    public get pointScale(): number {
        return this._renderer.pointScale;
    }
    public set pointScale(value: number) {
        this._renderer.pointScale = value;
        this._renderer.resetAccumulation();
    }

    /** Maximum number of frames blended into the progressive accumulation. */
    public get maxAccumFrames(): number {
        return this._renderer.maxAccumFrames;
    }
    public set maxAccumFrames(value: number) {
        this._renderer.maxAccumFrames = value;
    }

    /** Restarts progressive accumulation (called automatically on camera move / resize / new data). */
    public resetAccumulation(): void {
        this._renderer.resetAccumulation();
    }

    /**
     * Loads splat data in the standard `.splat` layout (per splat: 3 float position, 3 float scale,
     * 4 byte RGBA color, 4 byte quaternion) into the point-splatting pipeline.
     * @param splatsData raw splat bytes
     * @param options interpretation options
     * @param shData packed higher-order SH textures (`Uint8Array[]`, 16 bytes/splat/texture) from the loader, or undefined
     * @param shDegree spherical-harmonics degree of shData (1..3)
     */
    public updateData(splatsData: ArrayBuffer, options?: IGaussianPointSplattingUpdateOptions, shData?: Uint8Array[], shDegree = 0): void {
        const bytes = new Uint8Array(splatsData);
        const floats = new Float32Array(splatsData);
        const count = (bytes.length / SplatRowStride) | 0;
        const flipY = options?.flipY ? -1 : 1;

        // Optional world transform (column-major .m) baked into means and covariance so the compute
        // renderer matches how the classic path renders the same data under a node world matrix.
        const wm = options?.worldMatrix ?? null;

        const means = new Float32Array(count * 4);
        const cov3d = new Float32Array(count * 8);
        const colorOpacity = new Uint32Array(count);
        const sh = this._dequantizeSh(shData, shDegree, count);

        const quaternion = new Quaternion();
        const rotation = new Matrix();
        const scale = new Matrix();
        const rs = new Matrix();

        const wmm = wm ? wm.m : null;
        this._computeInverseWorldRotation(wmm);
        let bMinX = Infinity,
            bMinY = Infinity,
            bMinZ = Infinity;
        let bMaxX = -Infinity,
            bMaxY = -Infinity,
            bMaxZ = -Infinity;
        for (let i = 0; i < count; i++) {
            let mx = floats[8 * i + 0];
            let my = floats[8 * i + 1] * flipY;
            let mz = floats[8 * i + 2];

            const qb = SplatRowStride * i + 28;
            quaternion.set((bytes[qb + 1] - 127.5) / 127.5, (bytes[qb + 2] - 127.5) / 127.5, (bytes[qb + 3] - 127.5) / 127.5, -(bytes[qb + 0] - 127.5) / 127.5);
            quaternion.normalize();
            quaternion.toRotationMatrix(rotation);
            // Source 3D covariance Sigma = M M^T with M = R * S. The classic mesh doubles the scale
            // (rawScale * 2) and cancels it at render: its quad's invViewport = 1/renderWidth draws at
            // half the projected pixel size (one pixel spans 2/renderWidth in NDC). This compute path
            // has no quad, so it bakes the net scale directly (no * 2) to render at the same size.
            Matrix.ScalingToRef(floats[8 * i + 3], floats[8 * i + 4], floats[8 * i + 5], scale);
            rotation.multiplyToRef(scale, rs);
            const m = rs.m;
            let s00 = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
            let s01 = m[0] * m[4] + m[1] * m[5] + m[2] * m[6];
            let s02 = m[0] * m[8] + m[1] * m[9] + m[2] * m[10];
            let s11 = m[4] * m[4] + m[5] * m[5] + m[6] * m[6];
            let s12 = m[4] * m[8] + m[5] * m[9] + m[6] * m[10];
            let s22 = m[8] * m[8] + m[9] * m[9] + m[10] * m[10];

            if (wmm) {
                // World transform of the mean (column-major m: v' = M * v).
                const wx = wmm[0] * mx + wmm[4] * my + wmm[8] * mz + wmm[12];
                const wy = wmm[1] * mx + wmm[5] * my + wmm[9] * mz + wmm[13];
                const wz = wmm[2] * mx + wmm[6] * my + wmm[10] * mz + wmm[14];
                mx = wx;
                my = wy;
                mz = wz;

                // Transform the covariance by the same linear map A used on the mean (A[i][j] = wmm[i + 4j]):
                // Sigma' = A Sigma A^T. A must match the mean's column-major convention exactly, because the
                // Sigma formula above reads M row-major (Sigma = M M^T); mixing the two shears the splats.
                const a00 = wmm[0],
                    a01 = wmm[4],
                    a02 = wmm[8];
                const a10 = wmm[1],
                    a11 = wmm[5],
                    a12 = wmm[9];
                const a20 = wmm[2],
                    a21 = wmm[6],
                    a22 = wmm[10];
                const b00 = a00 * s00 + a01 * s01 + a02 * s02,
                    b01 = a00 * s01 + a01 * s11 + a02 * s12,
                    b02 = a00 * s02 + a01 * s12 + a02 * s22;
                const b10 = a10 * s00 + a11 * s01 + a12 * s02,
                    b11 = a10 * s01 + a11 * s11 + a12 * s12,
                    b12 = a10 * s02 + a11 * s12 + a12 * s22;
                const b20 = a20 * s00 + a21 * s01 + a22 * s02,
                    b21 = a20 * s01 + a21 * s11 + a22 * s12,
                    b22 = a20 * s02 + a21 * s12 + a22 * s22;
                s00 = b00 * a00 + b01 * a01 + b02 * a02;
                s01 = b00 * a10 + b01 * a11 + b02 * a12;
                s02 = b00 * a20 + b01 * a21 + b02 * a22;
                s11 = b10 * a10 + b11 * a11 + b12 * a12;
                s12 = b10 * a20 + b11 * a21 + b12 * a22;
                s22 = b20 * a20 + b21 * a21 + b22 * a22;
            }

            means[4 * i + 0] = mx;
            means[4 * i + 1] = my;
            means[4 * i + 2] = mz;
            means[4 * i + 3] = 0;
            bMinX = Math.min(bMinX, mx);
            bMinY = Math.min(bMinY, my);
            bMinZ = Math.min(bMinZ, mz);
            bMaxX = Math.max(bMaxX, mx);
            bMaxY = Math.max(bMaxY, my);
            bMaxZ = Math.max(bMaxZ, mz);
            cov3d[8 * i + 0] = s00;
            cov3d[8 * i + 1] = s01;
            cov3d[8 * i + 2] = s02;
            cov3d[8 * i + 3] = s11;
            cov3d[8 * i + 4] = s12;
            cov3d[8 * i + 5] = s22;

            const cb = SplatRowStride * i + 24;
            colorOpacity[i] = bytes[cb] | (bytes[cb + 1] << 8) | (bytes[cb + 2] << 16) | (bytes[cb + 3] << 24);
        }

        this._splatCount = count;
        if (count > 0) {
            this._boundsMin.copyFromFloats(bMinX, bMinY, bMinZ);
            this._boundsMax.copyFromFloats(bMaxX, bMaxY, bMaxZ);
        }
        this._renderer.updateSplats(means, cov3d, colorOpacity, sh, shDegree, count);
    }

    /**
     * Stores the rows of inverse(world 3x3) so the SH view direction can be brought into splat-local
     * space (where SH coefficients live). Column-major input `wmm` (Matrix.m); identity when null.
     * @param wmm column-major world matrix components (Matrix.m), or null for identity
     */
    private _computeInverseWorldRotation(wmm: Nullable<ArrayLike<number>>): void {
        const r = this._invWorldRot;
        if (!wmm) {
            r.set([1, 0, 0, 0, 1, 0, 0, 0, 1]);
            return;
        }
        // World 3x3 (row i, col j) = wmm[i + 4j]; invert it via the adjugate / determinant.
        const a = wmm[0],
            b = wmm[4],
            c = wmm[8];
        const d = wmm[1],
            e = wmm[5],
            f = wmm[9];
        const g = wmm[2],
            h = wmm[6],
            i = wmm[10];
        const cofA = e * i - f * h,
            cofB = f * g - d * i,
            cofC = d * h - e * g;
        const det = a * cofA + b * cofB + c * cofC;
        const inv = Math.abs(det) > 1e-12 ? 1 / det : 0;
        // Rows of the inverse (adjugate transposed), scaled by 1/det.
        r[0] = cofA * inv;
        r[1] = (c * h - b * i) * inv;
        r[2] = (b * f - c * e) * inv;
        r[3] = cofB * inv;
        r[4] = (a * i - c * g) * inv;
        r[5] = (c * d - a * f) * inv;
        r[6] = cofC * inv;
        r[7] = (b * g - a * h) * inv;
        r[8] = (a * e - b * d) * inv;
    }

    /**
     * Dequantizes the loader's packed SH textures into a tight per-splat float array (interleaved RGB,
     * shDim*3 floats per splat). The packed form is `Uint8Array[]` — one 16-byte-per-splat texture per
     * 16 scalar components — with each byte b encoding a coefficient as b*2/255 - 1.
     * @param shData packed per-splat SH textures, one per 16 scalar components, or undefined
     * @param shDegree spherical harmonics degree (0-3)
     * @param count number of splats
     * @returns dequantized interleaved RGB SH coefficients, or null if shData is empty or shDegree < 1
     */
    private _dequantizeSh(shData: Uint8Array[] | undefined, shDegree: number, count: number): Nullable<Float32Array> {
        if (!shData || shData.length === 0 || shDegree < 1) {
            return null;
        }
        const shDim = shDegree === 1 ? 3 : shDegree === 2 ? 8 : 15;
        const scalars = shDim * 3;
        const out = new Float32Array(count * scalars);
        for (let i = 0; i < count; i++) {
            for (let k = 0; k < scalars; k++) {
                const textureIndex = (k / 16) | 0;
                const byteInSplat = k % 16;
                const tex = shData[textureIndex];
                out[i * scalars + k] = (tex[i * 16 + byteInSplat] * 2) / 255 - 1;
            }
        }
        return out;
    }

    private _runCompute(camera: Camera): void {
        // Skip all compute when disabled (e.g. toggled off in favor of the classic renderer) so we
        // neither waste GPU work nor composite over the active renderer.
        if (!this.isEnabled()) {
            return;
        }
        const engine = this.getScene().getEngine();
        const width = engine.getRenderWidth();
        const height = engine.getRenderHeight();

        const view = camera.getViewMatrix();
        const projection = camera.getProjectionMatrix();
        view.multiplyToRef(projection, this._vpMatrix);
        // Focal length in pixels, matching the classic material: (w * proj_00 / 2, h * proj_11 / 2).
        const focalX = (width * projection.m[0]) / 2;
        const focalY = (height * projection.m[5]) / 2;
        const camPos = camera.globalPosition;
        this._renderer.reverseDepth = engine.useReverseDepthBuffer;

        // Project the model AABB to find its NDC-z span this frame, so the depth-sort key can use the
        // full 16-bit range over just the model instead of the whole (possibly enormous) scene depth.
        const vp = this._vpMatrix.m;
        const bmin = this._boundsMin;
        const bmax = this._boundsMax;
        let ndczMin = Infinity;
        let ndczMax = -Infinity;
        for (let c = 0; c < 8; c++) {
            const x = c & 1 ? bmax.x : bmin.x;
            const y = c & 2 ? bmax.y : bmin.y;
            const z = c & 4 ? bmax.z : bmin.z;
            const cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
            if (cw > 1e-6) {
                const ndcz = (vp[2] * x + vp[6] * y + vp[10] * z + vp[14]) / cw;
                ndczMin = Math.min(ndczMin, ndcz);
                ndczMax = Math.max(ndczMax, ndcz);
            }
        }
        if (!(ndczMax > ndczMin)) {
            ndczMin = 0;
            ndczMax = 1;
        }
        this._renderer.setCamera(view, this._vpMatrix, camera.minZ, camera.maxZ, focalX, focalY, camPos.x, camPos.y, camPos.z, ndczMin, ndczMax);
        this._renderer.setInverseWorldRotation(this._invWorldRot);

        // renderToBuffer allocates/resizes buffers even when the compute pipeline is still compiling,
        // and dispatches once ready. The blit binds the accumulation buffer every frame regardless:
        // the mesh is drawn each frame and an unbound storage buffer is a validation error.
        this._renderer.renderToBuffer(width, height);

        const accum = this._renderer.accumBuffer;
        const accumDepth = this._renderer.accumDepthBuffer;
        if (accum && accumDepth) {
            this._blitMaterial.setAccumBuffer(accum);
            this._blitMaterial.setAccumDepthBuffer(accumDepth);
            this._blitMaterial.setResolution(this._renderer.width, this._renderer.height);
        }
    }

    /**
     * Releases the mesh and all GPU resources owned by its renderer and blit material.
     * @param doNotRecurse passed through to {@link Mesh.dispose}
     * @param disposeMaterialAndTextures passed through to {@link Mesh.dispose}
     */
    public override dispose(doNotRecurse?: boolean, disposeMaterialAndTextures?: boolean): void {
        if (this._beforeRenderObserver) {
            this.getScene().onBeforeRenderObservable.remove(this._beforeRenderObserver);
            this._beforeRenderObserver = null;
        }
        this._renderer.dispose();
        this._blitMaterial.dispose();
        super.dispose(doNotRecurse, disposeMaterialAndTextures);
    }

    /**
     * Returns the class name.
     * @returns "GaussianPointSplattingMesh"
     */
    public override getClassName(): string {
        return "GaussianPointSplattingMesh";
    }
}
