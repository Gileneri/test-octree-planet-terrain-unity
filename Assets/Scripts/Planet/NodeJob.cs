using UnityEngine;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Burst;

/// <summary>
/// Generates voxel mesh data for a single octree node.
///
/// MODIFICATIONS:
///   Before scheduling, the main thread copies all WorldModifications that
///   overlap this node into two parallel NativeArrays (modKeys / modValues).
///   IsAirWorldPos checks those arrays first; only if no override is found
///   does it fall back to the procedural noise.
///
/// SEAMLESS WRAP:
///   XZ positions are wrapped via PosMod before noise sampling using the
///   4-D toroidal embedding (cos/sin) so the terrain tiles with no seam.
///
/// WORLD CONFIG INTEGRATION (added):
///   noiseSeed, noiseTypeId and minSubsurfaceHeight are now job fields
///   filled by Node.TryCollectJob from Octree, which receives them from
///   OctreeGrid, which receives them from WorldConfigLoader.
///   No hard-coded values remain for these settings.
/// </summary>
[BurstCompile]
public struct NodeJob : IJob
{
    public Bounds bounds;
    public Mesh.MeshDataArray meshDataArray;

    public float worldSizeX;
    public float worldSizeZ;
    public float surfaceBaseHeight;
    public float surfaceNoiseAmplitude;

    // ── WorldConfig-driven fields (were hard-coded before) ────────────────
    /// <summary>
    /// FastNoiseLite seed. Maps directly to noise.SetSeed().
    /// </summary>
    public int noiseSeed;

    /// <summary>
    /// Noise type as an integer so Burst can handle it without managed enums.
    /// Use NodeJob.NoiseTypeToId() on the main thread to convert the string
    /// from WorldConfig (e.g. "OpenSimplex2") to the right integer.
    /// 0=OpenSimplex2 (default), 1=OpenSimplex2S, 2=Cellular,
    /// 3=Perlin, 4=ValueCubic, 5=Value.
    /// </summary>
    public int noiseTypeId;

    /// <summary>
    /// Hard floor for solid terrain.  Any voxel whose world-Y is below this
    /// value is always solid, regardless of the noise surface.
    /// Set to a very negative number (e.g. -9999) to disable.
    /// </summary>
    public float minSubsurfaceHeight;
    // ─────────────────────────────────────────────────────────────────────

    public int chunkResolution;
    public float nodeScale;
    public float3 worldNodePosition;

    // Modification overrides for this node (prepared on main thread before scheduling)
    // Each entry: modKeys[i] is a wrapped voxel position, modValues[i] is 0=Solid 1=Air
    [ReadOnly] public NativeArray<int3> modKeys;
    [ReadOnly] public NativeArray<byte> modValues;

    private float3 centerOffset;
    private float normalizedVoxelScale;

    // -----------------------------------------------------------------------
    //  Noise type id helper  (call on main thread — not inside Burst)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Converts the noiseType string from WorldConfig to the integer id that
    /// NodeJob.noiseTypeId expects.  Falls back to 0 (OpenSimplex2) for
    /// unrecognised values.
    /// Available in this FastNoiseLite build:
    ///   0 = OpenSimplex2 (default)
    ///   1 = OpenSimplex2S
    ///   2 = Perlin
    /// </summary>
    public static int NoiseTypeToId(string noiseType)
    {
        switch (noiseType)
        {
            case "OpenSimplex2S": return 1;
            case "Perlin": return 2;
            default: return 0; // OpenSimplex2
        }
    }

    // -----------------------------------------------------------------------
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

        var noise = new FastNoiseLite();
        noise.SetSeed(noiseSeed);
        noise.SetFrequency(0.003f);

        // Apply noise type from the id field.
        // Only types enabled in this FastNoiseLite build:
        // 0=OpenSimplex2, 1=OpenSimplex2S, 2=Perlin
        switch (noiseTypeId)
        {
            case 1: noise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2S); break;
            case 2: noise.SetNoiseType(FastNoiseLite.NoiseType.Perlin); break;
            default: noise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2); break;
        }

        int vi = 0;
        int ii = 0;

        for (int x = 0; x < chunkResolution; x++)
            for (int y = 0; y < chunkResolution; y++)
                for (int z = 0; z < chunkResolution; z++)
                {
                    if (IsAir(x, y, z, ref noise)) continue;

                    float3 localPos = new float3(x, y, z) * normalizedVoxelScale - centerOffset;

                    for (int side = 0; side < 6; side++)
                    {
                        int3 nb = Tables.NeighborOffset[side];
                        if (!IsAir(x + nb.x, y + nb.y, z + nb.z, ref noise)) continue;

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

    // -----------------------------------------------------------------------
    //  Density
    // -----------------------------------------------------------------------

    private bool IsAir(int lx, int ly, int lz, ref FastNoiseLite noise)
    {
        float3 wp = new float3(lx, ly, lz) * normalizedVoxelScale
                    + worldNodePosition
                    - centerOffset;
        return IsAirWorldPos(wp, ref noise);
    }

    private bool IsAirWorldPos(float3 wp, ref FastNoiseLite noise)
    {
        // Hard floor: always solid below minSubsurfaceHeight
        if (wp.y < minSubsurfaceHeight) return false;

        // --- Check modification overrides first ---
        // The key is the wrapped integer voxel coordinate, matching WorldModifications.WrapKey
        int3 key = new int3(
            (int)math.floor(PosMod(wp.x, worldSizeX)),
            (int)math.floor(wp.y),
            (int)math.floor(PosMod(wp.z, worldSizeZ))
        );

        for (int i = 0; i < modKeys.Length; i++)
        {
            if (modKeys[i].x == key.x && modKeys[i].y == key.y && modKeys[i].z == key.z)
                return modValues[i] == 1; // 1 = Air, 0 = Solid
        }

        // --- Procedural noise fallback ---
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

        float n1 = noise.GetNoise(cx, sx, cz);
        float n2 = noise.GetNoise(cz + 100f, sz + 100f, cx * 0.5f);
        float n3 = noise.GetNoise(sx * 0.7f, sz * 0.7f + 200f, cz * 0.3f);

        float h = n1 * 0.6f + n2 * 0.3f + n3 * 0.1f;
        float surfaceY = surfaceBaseHeight + h * surfaceNoiseAmplitude;

        return wp.y > surfaceY;
    }

    // -----------------------------------------------------------------------
    private static float PosMod(float x, float m)
    {
        float r = x % m;
        return r < 0f ? r + m : r;
    }
}