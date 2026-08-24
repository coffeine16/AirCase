/**
 * Shared deck.gl parameters for layers that must stay VISIBLE over the 3D
 * fusion field.
 *
 * The fusion choropleth extrudes each hexagon to its PM2.5 value
 * (elevationScale 12, so a Delhi column can stand ~2,500 units tall). Every
 * other layer — stations, fires, hotspot zones, blind spots, dispatch routes —
 * is drawn flat at ground level, so the depth buffer correctly concludes they
 * are BEHIND the columns and discards them. The analyst flips to 3D and the
 * evidence layers vanish.
 *
 * Turning off the depth test is the right fix rather than a hack: these are
 * map overlays, not objects in the scene. A fire detection is an annotation
 * about a place, and an annotation that hides when a nearby column is tall is
 * simply broken — the same reason a map pin is never occluded by terrain.
 *
 * The trade-off is that overlays no longer sort against each other by depth.
 * That costs nothing here: they are 2-D marks at ground level, so painter's
 * order (the order layers are listed) is what decides them, which is what we
 * want anyway.
 *
 * `depthCompare: "always"` and NOT `depthTest: false` — deck.gl v9 moved to
 * luma.gl v9's WebGPU-style parameters, where the old boolean no longer exists.
 * It is not a silent no-op if you get it wrong: TypeScript rejects the v8
 * spelling outright ("no properties in common with type DepthStencilParameters").
 */
export const OVERLAY_PARAMETERS = { depthCompare: "always" } as const;
