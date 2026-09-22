/**
 * Re-exports pure implementation and applies runtime side effects.
 * Import gaussianPointSplattingBlitMaterial.pure for tree-shakeable, side-effect-free usage.
 */
export * from "./gaussianPointSplattingBlitMaterial.pure";

import "../../ShadersWGSL/gaussianPointSplattingBlit.vertex";
import "../../ShadersWGSL/gaussianPointSplattingBlit.fragment";
