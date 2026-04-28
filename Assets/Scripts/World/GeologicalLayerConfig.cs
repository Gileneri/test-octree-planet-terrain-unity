using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Defines the ordered stack of geological layers for a world.
///
/// SETUP
/// ─────
/// Assets > Create > Voxel Void > Geological Layer Config
///
/// Add layers in order from TOP (shallowest) to BOTTOM (deepest).
/// The last layer acts as bedrock — it fills everything below.
///
/// LAYER DEPTH RULES
/// ─────────────────
/// baseDepth   — units below the local surface where this layer STARTS.
///               Layer 0 starts at depth 0 (right at the surface).
/// thickness   — how many units this layer spans before the next begins.
///               The last layer's thickness is ignored (fills to infinity).
///
/// BORDER NOISE
/// ────────────
/// Each layer's top border is displaced by a 2D noise (XZ only) to simulate
/// geological undulation. This is cheap — one GetNoise(x,z) call per voxel
/// per layer, no cos/sin toroidal embedding needed for subsurface layers.
///
/// EXAMPLE — Earths_Moon
/// ──────────────────────
///   0  Dirt     depth=0    thick=8    freq=0.02  amp=4
///   1  Stone    depth=8    thick=200  freq=0.01  amp=20
///   2  Bedrock  depth=208  thick=∞    freq=0.005 amp=5
///
/// HOW IT REACHES THE JOB
/// ──────────────────────
/// WorldConfigLoader pushes a NativeArray<GeologicalLayerBlob> into the Octree,
/// which forwards it to Node, which writes it into NodeJob before scheduling.
/// GeologicalLayerBlob is a Burst-compatible blittable struct (no strings).
/// </summary>
[CreateAssetMenu(menuName = "Voxel Void/Geological Layer Config", fileName = "GeologicalLayers_New")]
public class GeologicalLayerConfig : ScriptableObject
{
    [Tooltip("Ordered list of layers, shallowest first. Last layer = bedrock (fills to floor).")]
    public List<GeologicalLayer> layers = new List<GeologicalLayer>
    {
        new GeologicalLayer { layerName = "Dirt",    blockId = 2, baseDepth = 0f,   thickness = 8f,   borderNoiseFrequency = 0.02f, borderNoiseAmplitude = 4f,  borderNoiseSeed = 100 },
        new GeologicalLayer { layerName = "Stone",   blockId = 3, baseDepth = 8f,   thickness = 200f, borderNoiseFrequency = 0.01f, borderNoiseAmplitude = 20f, borderNoiseSeed = 200 },
        new GeologicalLayer { layerName = "Bedrock", blockId = 4, baseDepth = 208f, thickness = 9999f,borderNoiseFrequency = 0.005f,borderNoiseAmplitude = 5f,  borderNoiseSeed = 300 },
    };

    // ------------------------------------------------------------------
    //  Conversion to Burst-compatible blob array
    // ------------------------------------------------------------------

    /// <summary>
    /// Converts the managed layer list to a plain array of blittable structs
    /// ready to be wrapped in a NativeArray and passed to NodeJob.
    /// Call this on the main thread before scheduling jobs.
    /// </summary>
    public GeologicalLayerBlob[] ToBlobs()
    {
        if (layers == null) return Array.Empty<GeologicalLayerBlob>();
        var result = new GeologicalLayerBlob[layers.Count];
        for (int i = 0; i < layers.Count; i++)
            result[i] = layers[i].ToBlob();
        return result;
    }
}

// ──────────────────────────────────────────────────────────────────────────
//  Managed layer definition  (Inspector-friendly)
// ──────────────────────────────────────────────────────────────────────────

[Serializable]
public class GeologicalLayer
{
    [Tooltip("Display name only — not used at runtime.")]
    public string layerName = "Layer";

    [Tooltip("Block id for this layer. Must match a BlockDefinition with the same id.")]
    [Range(1, 254)]
    public byte blockId = 1;

    [Tooltip("Depth below the local surface where this layer starts (units).")]
    public float baseDepth = 0f;

    [Tooltip("Vertical thickness of this layer (units). Ignored for the last (bedrock) layer.")]
    public float thickness = 10f;

    [Header("Border noise — top edge undulation")]
    [Tooltip("XZ frequency of the border displacement noise. Higher = more jagged transitions.")]
    public float borderNoiseFrequency = 0.01f;

    [Tooltip("Max vertical displacement of the layer border (units). 0 = flat horizontal border.")]
    public float borderNoiseAmplitude = 10f;

    [Tooltip("Seed offset for this layer's border noise. Each layer should use a different value.")]
    public int borderNoiseSeed = 0;

    public GeologicalLayerBlob ToBlob() => new GeologicalLayerBlob
    {
        blockId              = blockId,
        baseDepth            = baseDepth,
        thickness            = thickness,
        borderNoiseFrequency = borderNoiseFrequency,
        borderNoiseAmplitude = borderNoiseAmplitude,
        borderNoiseSeed      = borderNoiseSeed,
    };
}

// ──────────────────────────────────────────────────────────────────────────
//  Burst-compatible blittable struct  (passed inside NativeArray to NodeJob)
// ──────────────────────────────────────────────────────────────────────────

/// <summary>
/// Plain blittable version of GeologicalLayer for use inside Burst jobs.
/// No strings, no managed references.
/// </summary>
public struct GeologicalLayerBlob
{
    public byte  blockId;
    public float baseDepth;
    public float thickness;
    public float borderNoiseFrequency;
    public float borderNoiseAmplitude;
    public int   borderNoiseSeed;
}
