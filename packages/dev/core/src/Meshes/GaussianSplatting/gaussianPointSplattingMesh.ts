/**
 * Re-exports pure implementation and applies runtime side effects.
 * Import gaussianPointSplattingMesh.pure for tree-shakeable, side-effect-free usage.
 */
export * from "./gaussianPointSplattingMesh.pure";

import "./gaussianPointSplattingRenderer";
import "../../Materials/GaussianSplatting/gaussianPointSplattingBlitMaterial";
