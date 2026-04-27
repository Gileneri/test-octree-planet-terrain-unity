using UnityEngine;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Burst;

/// <summary>
/// Generates voxel mesh data for a single octree node using a layered
/// density pipeline.
///
/// ═══════════════════════════════════════════════════════════════════
///  DENSITY PIPELINE  (SampleDensity — top = highest priority)
/// ═══════════════════════════════════════════════════════════════════
///
///  Layer 0 — FLOOR
///    Voxels below minSubsurfaceHeight are unconditionally Solid.
///    Nothing can override this — not player digs, not caves.
///
///  Layer 1 — PLAYER MODIFICATIONS
///    Explicit Air / Solid overrides from BlockInteraction.
///    Stored in modKeys / modValues (snapshot prepared before scheduling).
///
///  Layer 2 — PROCEDURAL SURFACE
///    Toroidal noise surface that shapes the terrain.
///    Voxels above surfaceY → Air.
///    Voxels below surfaceY → Solid (falls through to Layer 3).
///
///  Layer 3 — PROCEDURAL CAVES  [reserved — enable via cavesEnabled]
///    3-D worm noise carved from the solid underground region.
///    Fades out near the surface so caves never punch through.
///
/// ═══════════════════════════════════════════════════════════════════
///  ENABLING CAVES
/// ═══════════════════════════════════════════════════════════════════
///  In WorldConfig (and Earths_Moon.json):
///    "cavesEnabled": true
///    "caveNoiseFrequency": 0.008
///    "caveNoiseThreshold": 0.55
///    "caveNoiseAmplitudeY": 0.4
///    "caveSurfaceFadeRange": 20.0
///
///  No code changes required — the slot is already wired end-to-end.
/// </summary>
[BurstCompile]
public struct NodeJob : IJob
{
    // ── Mesh output ───────────────────────────────────────────────────────
    public Bounds bounds;
    public Mesh.MeshDataArray meshDataArray;

    // ── World ─────────────────────────────────────────────────────────────
    public float worldSizeX;
    public float worldSizeZ;

    // ── Layer 0 : Floor ───────────────────────────────────────────────────
    public float minSubsurfaceHeight;

    // ── Layer 1 : Player modifications ────────────────────────────────────
    [ReadOnly] public NativeArray<int3> modKeys;
    [ReadOnly] public NativeArray<byte> modValues;

    // ── Layer 2 : Procedural surface ──────────────────────────────────────
    public float surfaceBaseHeight;
    public float surfaceNoiseAmplitude;
    public int noiseSeed;
    public int noiseTypeId;   // 0=OpenSimplex2  1=OpenSimplex2S  2=Perlin

    // ── Layer 3 : Caves ───────────────────────────────────────────────────
    public bool cavesEnabled;
    public float caveNoiseFrequency;    // typical 0.005–0.02
    public float caveNoiseThreshold;    // typical 0.4–0.7 (higher = fewer caves)
    public float caveNoiseAmplitudeY;   // vertical squish, typical 0.3–0.6
    public float caveSurfaceFadeRange;  // Y units below surface where caves fade in

    // ── Chunk ─────────────────────────────────────────────────────────────
    public int chunkResolution;
    public float nodeScale;
    public float3 worldNodePosition;

    // ── Private ───────────────────────────────────────────────────────────
    private float3 centerOffset;
    private float normalizedVoxelScale;

    // ═══════════════════════════════════════════════════════════════════════
    //  Main-thread helper
    // ═══════════════════════════════════════════════════════════════════════

    public static int NoiseTypeToId(string noiseType)
    {
        switch (noiseType)
        {
            case "OpenSimplex2S": return 1;
            case "Perlin": return 2;
            default: return 0;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Execute
    // ═══════════════════════════════════════════════════════════════════════

    public void Execute()
    {
        int maxCubes = (int)math.ceil(chunkResolution * chunkResolution * chunkResolution / 2f);
        int maxIdx = maxCubes * 6 * 6;
        int maxVerts = maxCubes * 4 * 6;

        var indices = new NativeArray<uint>(maxIdx, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        var vertices = new NativeArray<float3>(maxVerts, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        var normals = new NativeArray<float3>(maxVerts, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

        centerOffset = new float3(1f, 1f, 1f) * (nodeScale * 0.5f);
        normalizedVoxelScale = nodeScale / chunkResolution;

        // Layer 2 — surface noise
        var surfaceNoise = new FastNoiseLite();
        surfaceNoise.SetSeed(noiseSeed);
        surfaceNoise.SetFrequency(0.003f);
        switch (noiseTypeId)
        {
            case 1: surfaceNoise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2S); break;
            case 2: surfaceNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin); break;
            default: surfaceNoise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2); break;
        }

        // Layer 3 — cave noise (always constructed; only sampled when cavesEnabled)
        var caveNoise = new FastNoiseLite();
        caveNoise.SetSeed(noiseSeed + 9371);    // decorrelated from surface
        caveNoise.SetFrequency(caveNoiseFrequency > 0f ? caveNoiseFrequency : 0.008f);
        caveNoise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2);

        int vi = 0;
        int ii = 0;

        for (int x = 0; x < chunkResolution; x++)
            for (int y = 0; y < chunkResolution; y++)
                for (int z = 0; z < chunkResolution; z++)
                {
                    if (IsAir(x, y, z, ref surfaceNoise, ref caveNoise)) continue;

                    float3 localPos = new float3(x, y, z) * normalizedVoxelScale - centerOffset;

                    for (int side = 0; side < 6; side++)
                    {
                        int3 nb = Tables.NeighborOffset[side];
                        if (!IsAir(x + nb.x, y + nb.y, z + nb.z, ref surfaceNoise, ref caveNoise)) continue;

                        vertices[vi + 0] = Tables.Vertices[Tables.BuildOrder[side][0]] * normalizedVoxelScale + localPos;
                        vertices[vi + 1] = Tables.Vertices[Tables.BuildOrder[side][1]] * normalizedVoxelScale + localPos;
                        vertices[vi + 2] = Tables.Vertices[Tables.BuildOrder[side][2]] * normalizedVoxelScale + localPos;
                        vertices[vi + 3] = Tables.Vertices[Tables.BuildOrder[side][3]] * normalizedVoxelScale + localPos;

                        normals[vi + 0] = normals[vi + 1] = normals[vi + 2] = normals[vi + 3] = Tables.Normals[side];

                        indices[ii + 0] = (uint)(vi + 0);
                        indices[ii + 1] = (uint)(vi + 1);
                        indices[ii + 2] = (uint)(vi + 2);
                        indices[ii + 3] = (uint)(vi + 2);
                        indices[ii + 4] = (uint)(vi + 1);
                        indices[ii + 5] = (uint)(vi + 3);

                        vi += 4;
                        ii += 6;
                    }
                }

        var idxSlice = new NativeArray<uint>(ii, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        var vtxSlice = new NativeArray<float3>(vi, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        var nrmSlice = new NativeArray<float3>(vi, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        indices.Slice(0, ii).CopyTo(idxSlice);
        vertices.Slice(0, vi).CopyTo(vtxSlice);
        normals.Slice(0, vi).CopyTo(nrmSlice);

        MeshingUtility.ApplyMesh(ref meshDataArray, ref idxSlice, ref vtxSlice, ref nrmSlice, ref bounds);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Density pipeline
    // ═══════════════════════════════════════════════════════════════════════

    private bool IsAir(int lx, int ly, int lz,
        ref FastNoiseLite surfaceNoise, ref FastNoiseLite caveNoise)
    {
        float3 wp = new float3(lx, ly, lz) * normalizedVoxelScale
                    + worldNodePosition - centerOffset;
        return SampleDensity(wp, ref surfaceNoise, ref caveNoise);
    }

    /// <summary>
    /// Returns true = Air, false = Solid.
    /// Layers are evaluated top to bottom; the first definitive answer wins.
    /// </summary>
    private bool SampleDensity(float3 wp,
        ref FastNoiseLite surfaceNoise, ref FastNoiseLite caveNoise)
    {
        // ── Layer 0 : Floor ────────────────────────────────────────────────
        if (wp.y < minSubsurfaceHeight)
            return false; // Solid — unconditional, nothing overrides

        // ── Layer 1 : Player modifications ────────────────────────────────
        int3 key = new int3(
            (int)math.floor(PosMod(wp.x, worldSizeX)),
            (int)math.floor(wp.y),
            (int)math.floor(PosMod(wp.z, worldSizeZ))
        );

        for (int i = 0; i < modKeys.Length; i++)
        {
            if (modKeys[i].x == key.x &&
                modKeys[i].y == key.y &&
                modKeys[i].z == key.z)
            {
                return modValues[i] == 1; // 1 = Air, 0 = Solid
            }
        }

        // ── Layer 2 : Procedural surface ──────────────────────────────────
        float wx = PosMod(wp.x, worldSizeX);
        float wz = PosMod(wp.z, worldSizeZ);

        float aX = wx / worldSizeX * 2f * math.PI;
        float aZ = wz / worldSizeZ * 2f * math.PI;
        float R = worldSizeX / (2f * math.PI);
        float S = worldSizeZ / (2f * math.PI);

        float cx = math.cos(aX) * R;
        float sx = math.sin(aX) * R;
        float cz = math.cos(aZ) * S;
        float sz = math.sin(aZ) * S;

        float n1 = surfaceNoise.GetNoise(cx, sx, cz);
        float n2 = surfaceNoise.GetNoise(cz + 100f, sz + 100f, cx * 0.5f);
        float n3 = surfaceNoise.GetNoise(sx * 0.7f, sz * 0.7f + 200f, cz * 0.3f);

        float surfaceY = surfaceBaseHeight + (n1 * 0.6f + n2 * 0.3f + n3 * 0.1f) * surfaceNoiseAmplitude;

        if (wp.y > surfaceY)
            return true; // Air — above surface, caves can't carve air

        // ── Layer 3 : Procedural caves ────────────────────────────────────
        if (!cavesEnabled)
            return false; // Solid — underground, caves disabled

        // Fade caves out near the surface so they never punch a hole through
        float depthBelowSurface = surfaceY - wp.y;
        float fade = caveSurfaceFadeRange > 0f
            ? math.saturate(depthBelowSurface / caveSurfaceFadeRange)
            : 1f;

        // Y squished so caves are tunnel-shaped rather than spherical
        float cn = caveNoise.GetNoise(wp.x, wp.y * caveNoiseAmplitudeY, wp.z);

        // At fade=0 (surface) the threshold is 1.0 (impossible to be air → solid)
        // At fade=1 (deep)    the threshold is caveNoiseThreshold
        float effectiveThreshold = math.lerp(1f, caveNoiseThreshold, fade);

        return cn > effectiveThreshold; // Air = inside a cave tunnel
    }

    // ═══════════════════════════════════════════════════════════════════════

    private static float PosMod(float x, float m)
    {
        float r = x % m;
        return r < 0f ? r + m : r;
    }
}