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
        noise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2);
        noise.SetFrequency(0.003f);
        noise.SetSeed(2376);

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