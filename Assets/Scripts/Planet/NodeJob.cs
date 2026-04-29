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
    public bool cullEnclosedUndergroundFaces;
    public float enclosedUndergroundFaceMargin;
    public bool useSurfaceShellOnCoarseLod;
    public int surfaceShellMinDivisions;

    // ── Layer 4 : Geological layers ──────────────────────────────────────
    /// <summary>
    /// Ordered array of geological layers, shallowest first.
    /// Prepared on the main thread from GeologicalLayerConfig.ToBlobs().
    /// Empty array = uniform Stone (blockId 1) underground.
    /// </summary>
    [ReadOnly] public NativeArray<GeologicalLayerBlob> geoLayers;

    // ── Chunk ─────────────────────────────────────────────────────────────
    public int nodeDivisions;
    public int chunkResolution;
    public float nodeScale;
    public float3 worldNodePosition;
    public bool cullUndergroundFacesOnCoarseLod;
    public int coarseLodFaceCullMinDivisions;
    public float coarseLodFaceCullMargin;
    public int coarseLodMinSurfaceVoxels;

    // ── Private ───────────────────────────────────────────────────────────
    private float3 centerOffset;
    private float normalizedVoxelScale;

    // Pre-computed node AABB Y extents (set at start of Execute)
    // Used for the two node-level early exits.
    private float nodeMinY;   // world Y of the node's bottom face
    private float nodeMaxY;   // world Y of the node's top face

    // ═══════════════════════════════════════════════════════════════════════
    //  UV layout per face side
    //  Matches BuildOrder winding: BL, BR, TL, TR
    // ═══════════════════════════════════════════════════════════════════════
    // UV corners in the same winding order as Tables.BuildOrder quads
    static readonly float2 UV0 = new float2(0f, 0f); // bottom-left
    static readonly float2 UV1 = new float2(1f, 0f); // bottom-right
    static readonly float2 UV2 = new float2(0f, 1f); // top-left
    static readonly float2 UV3 = new float2(1f, 1f); // top-right

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
        var uvs = new NativeArray<float2>(maxVerts, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        var blockData = new NativeArray<float2>(maxVerts, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        // blockIds per voxel — used for face culling and (later) greedy meshing
        var blockIds = new NativeArray<byte>(chunkResolution * chunkResolution * chunkResolution,
            Allocator.Temp, NativeArrayOptions.ClearMemory); // 0 = Air by default

        // Zero-length arrays reused by node-level early exits
        var emptyUint = new NativeArray<uint>(0, Allocator.Temp);
        var emptyFloat3 = new NativeArray<float3>(0, Allocator.Temp);
        var emptyFloat2 = new NativeArray<float2>(0, Allocator.Temp);

        centerOffset = new float3(1f, 1f, 1f) * (nodeScale * 0.5f);
        normalizedVoxelScale = nodeScale / chunkResolution;

        // Node world-space Y extents
        nodeMinY = worldNodePosition.y - nodeScale * 0.5f;
        nodeMaxY = worldNodePosition.y + nodeScale * 0.5f;

        // ── Node-level early exits ────────────────────────────────────────
        // These check whether the entire node can be resolved without
        // iterating a single voxel. Both cases produce an empty mesh.

        // Exit A: node is entirely below the hard floor → all solid, no faces.
        if (nodeMaxY < minSubsurfaceHeight)
        {
            MeshingUtility.ApplyMesh(ref meshDataArray,
                ref emptyUint, ref emptyFloat3, ref emptyFloat3,
                ref emptyFloat2, ref emptyFloat2, ref bounds);
            return;
        }

        // Exit B: node is entirely above the maximum possible surface Y.
        // surfaceBaseHeight + surfaceNoiseAmplitude is the highest the terrain
        // can ever reach. If the node's bottom is above that, every voxel is Air
        // → no faces to emit (the mesh is empty).
        if (nodeMinY > surfaceBaseHeight + surfaceNoiseAmplitude)
        {
            MeshingUtility.ApplyMesh(ref meshDataArray,
                ref emptyUint, ref emptyFloat3, ref emptyFloat3,
                ref emptyFloat2, ref emptyFloat2, ref bounds);
            return;
        }

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

        // Layer 4 — geological border noises, one instance per layer created once here
        // (not inside SampleDensity, which would re-allocate per voxel)
        var borderNoises = new NativeArray<FastNoiseLite>(geoLayers.Length, Allocator.Temp);
        for (int li = 0; li < geoLayers.Length; li++)
        {
            var gl = geoLayers[li];
            var bn = new FastNoiseLite();
            bn.SetSeed(gl.borderNoiseSeed);
            bn.SetFrequency(gl.borderNoiseFrequency);
            bn.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2);
            borderNoises[li] = bn;
        }

        int vi = 0;
        int ii = 0;
        bool applySurfaceShell = useSurfaceShellOnCoarseLod
            && nodeDivisions >= math.max(1, surfaceShellMinDivisions);

        // Pass 1 — evaluate density for every voxel and cache blockIds.
        // This separates density evaluation from meshing so each voxel's
        // density is computed exactly once (not once per face check).
        for (int x = 0; x < chunkResolution; x++)
            for (int y = 0; y < chunkResolution; y++)
                for (int z = 0; z < chunkResolution; z++)
                {
                    float3 wp = new float3(x, y, z) * normalizedVoxelScale
                                + worldNodePosition - centerOffset;
                    blockIds[x * chunkResolution * chunkResolution + y * chunkResolution + z]
                        = SampleDensity(wp, ref surfaceNoise, ref caveNoise, borderNoises, applySurfaceShell);
                }

        // Pass 2 — emit faces where a solid voxel borders an Air voxel.
        // Optional: detect "externally reachable" air to avoid emitting deep
        // underground faces that border fully enclosed air pockets.
        int voxCount = chunkResolution * chunkResolution * chunkResolution;
        int res2 = chunkResolution * chunkResolution;
        bool applyCoarseUndergroundFaceCull = cullUndergroundFacesOnCoarseLod
            && nodeDivisions >= math.max(1, coarseLodFaceCullMinDivisions);
        int minSurfaceVoxelsToKeep = math.max(0, coarseLodMinSurfaceVoxels);
        var reachableAir = new NativeArray<byte>(voxCount, Allocator.Temp, NativeArrayOptions.ClearMemory);
        var floodQueue = new NativeArray<int>(voxCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        var topSurfaceYByXZ = new NativeArray<int>(res2, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        int qHead = 0;
        int qTail = 0;

        for (int i = 0; i < res2; i++)
            topSurfaceYByXZ[i] = -1;

        if (cullEnclosedUndergroundFaces)
        {
            for (int x = 0; x < chunkResolution; x++)
                for (int y = 0; y < chunkResolution; y++)
                    for (int z = 0; z < chunkResolution; z++)
                    {
                        if (x != 0 && x != chunkResolution - 1 &&
                            y != 0 && y != chunkResolution - 1 &&
                            z != 0 && z != chunkResolution - 1)
                            continue;

                        int seedIdx = x * res2 + y * chunkResolution + z;
                        if (blockIds[seedIdx] != 0 || reachableAir[seedIdx] != 0) continue;
                        reachableAir[seedIdx] = 1;
                        floodQueue[qTail++] = seedIdx;
                    }

            while (qHead < qTail)
            {
                int idx = floodQueue[qHead++];
                int x = idx / res2;
                int rem = idx - x * res2;
                int y = rem / chunkResolution;
                int z = rem - y * chunkResolution;

                for (int side = 0; side < 6; side++)
                {
                    int3 nb = Tables.NeighborOffset[side];
                    int nx = x + nb.x;
                    int ny = y + nb.y;
                    int nz = z + nb.z;
                    if (nx < 0 || nx >= chunkResolution ||
                        ny < 0 || ny >= chunkResolution ||
                        nz < 0 || nz >= chunkResolution)
                        continue;

                    int nIdx = nx * res2 + ny * chunkResolution + nz;
                    if (blockIds[nIdx] != 0 || reachableAir[nIdx] != 0) continue;
                    reachableAir[nIdx] = 1;
                    floodQueue[qTail++] = nIdx;
                }
            }
        }

        if (applyCoarseUndergroundFaceCull && minSurfaceVoxelsToKeep > 0)
        {
            for (int x = 0; x < chunkResolution; x++)
            {
                int xBase = x * res2;
                for (int z = 0; z < chunkResolution; z++)
                {
                    for (int y = chunkResolution - 1; y >= 0; y--)
                    {
                        int idx = xBase + y * chunkResolution + z;
                        if (blockIds[idx] == 0) continue;

                        bool hasAirAbove = y == chunkResolution - 1 ||
                            blockIds[xBase + (y + 1) * chunkResolution + z] == 0;
                        if (hasAirAbove)
                        {
                            topSurfaceYByXZ[x * chunkResolution + z] = y;
                            break;
                        }
                    }
                }
            }
        }

        for (int x = 0; x < chunkResolution; x++)
            for (int y = 0; y < chunkResolution; y++)
                for (int z = 0; z < chunkResolution; z++)
                {
                    byte bid = blockIds[x * res2 + y * chunkResolution + z];
                    if (bid == 0) continue; // Air

                    float3 localPos = new float3(x, y, z) * normalizedVoxelScale - centerOffset;
                    float3 voxelWp = localPos + worldNodePosition;

                    for (int side = 0; side < 6; side++)
                    {
                        int3 nb = Tables.NeighborOffset[side];
                        int nx = x + nb.x, ny = y + nb.y, nz = z + nb.z;

                        // Neighbour blockId — sample cache if in bounds, else query density
                        byte nbId;
                        if (nx >= 0 && nx < chunkResolution &&
                            ny >= 0 && ny < chunkResolution &&
                            nz >= 0 && nz < chunkResolution)
                        {
                            nbId = blockIds[nx * chunkResolution * chunkResolution + ny * chunkResolution + nz];
                        }
                        else
                        {
                            float3 wp = new float3(nx, ny, nz) * normalizedVoxelScale
                                        + worldNodePosition - centerOffset;
                            nbId = SampleDensity(wp, ref surfaceNoise, ref caveNoise, borderNoises, applySurfaceShell);
                        }

                        if (nbId != 0) continue; // neighbour is solid — face hidden

                        if (applyCoarseUndergroundFaceCull)
                        {
                            int topY = topSurfaceYByXZ[x * chunkResolution + z];
                            int keepFromY = topY >= 0
                                ? math.max(0, topY - (minSurfaceVoxelsToKeep - 1))
                                : int.MaxValue;
                            bool keepMinimumSurfaceBand = topY >= 0 && y >= keepFromY;

                            if (!keepMinimumSurfaceBand)
                            {
                                float3 nwp = new float3(nx, ny, nz) * normalizedVoxelScale
                                           + worldNodePosition - centerOffset;
                                float3 faceWp = (voxelWp + nwp) * 0.5f;
                                float surfaceY = SampleSurfaceHeight(faceWp.x, faceWp.z, ref surfaceNoise);
                                if (faceWp.y < surfaceY - math.max(0f, coarseLodFaceCullMargin))
                                    continue;
                            }
                        }

                        if (cullEnclosedUndergroundFaces &&
                            nx >= 0 && nx < chunkResolution &&
                            ny >= 0 && ny < chunkResolution &&
                            nz >= 0 && nz < chunkResolution)
                        {
                            int nIdx = nx * res2 + ny * chunkResolution + nz;
                            if (reachableAir[nIdx] == 0)
                            {
                                float3 nwp = new float3(nx, ny, nz) * normalizedVoxelScale
                                           + worldNodePosition - centerOffset;
                                float surfaceY = SampleSurfaceHeight(nwp.x, nwp.z, ref surfaceNoise);
                                if (nwp.y < surfaceY - math.max(0f, enclosedUndergroundFaceMargin))
                                    continue;
                            }
                        }

                        vertices[vi + 0] = Tables.Vertices[Tables.BuildOrder[side][0]] * normalizedVoxelScale + localPos;
                        vertices[vi + 1] = Tables.Vertices[Tables.BuildOrder[side][1]] * normalizedVoxelScale + localPos;
                        vertices[vi + 2] = Tables.Vertices[Tables.BuildOrder[side][2]] * normalizedVoxelScale + localPos;
                        vertices[vi + 3] = Tables.Vertices[Tables.BuildOrder[side][3]] * normalizedVoxelScale + localPos;

                        normals[vi + 0] = normals[vi + 1] = normals[vi + 2] = normals[vi + 3] = Tables.Normals[side];

                        // UVs — standard quad winding BL BR TL TR
                        uvs[vi + 0] = UV0;
                        uvs[vi + 1] = UV1;
                        uvs[vi + 2] = UV2;
                        uvs[vi + 3] = UV3;

                        // blockData: x = blockId (float), y = tex array layer for this face side
                        // layer = blockId * 3 + faceSlot (0=top, 1=bottom, 2=side)
                        // Tables.Normals: 0=right 1=left 2=up 3=down 4=front 5=back
                        int faceSlot = side == 2 ? 0 : (side == 3 ? 1 : 2);
                        float texLayer = bid * 3 + faceSlot;
                        var bd = new float2((float)bid, texLayer);
                        blockData[vi + 0] = bd;
                        blockData[vi + 1] = bd;
                        blockData[vi + 2] = bd;
                        blockData[vi + 3] = bd;

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
        var uvSlice = new NativeArray<float2>(vi, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        var bdSlice = new NativeArray<float2>(vi, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
        indices.Slice(0, ii).CopyTo(idxSlice);
        vertices.Slice(0, vi).CopyTo(vtxSlice);
        normals.Slice(0, vi).CopyTo(nrmSlice);
        uvs.Slice(0, vi).CopyTo(uvSlice);
        blockData.Slice(0, vi).CopyTo(bdSlice);

        reachableAir.Dispose();
        floodQueue.Dispose();
        topSurfaceYByXZ.Dispose();
        borderNoises.Dispose();

        MeshingUtility.ApplyMesh(ref meshDataArray,
            ref idxSlice, ref vtxSlice, ref nrmSlice,
            ref uvSlice, ref bdSlice, ref bounds);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Density pipeline
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Returns blockId for a voxel at local chunk coords.
    /// 0 = Air. Reads from the blockIds cache (Pass 1 must have run first).
    /// Used only for out-of-bounds neighbour checks in Pass 2.
    /// </summary>
    private byte SampleDensityLocal(int lx, int ly, int lz,
        ref FastNoiseLite surfaceNoise, ref FastNoiseLite caveNoise,
        NativeArray<FastNoiseLite> borderNoises,
        bool applySurfaceShell)
    {
        float3 wp = new float3(lx, ly, lz) * normalizedVoxelScale
                    + worldNodePosition - centerOffset;
        return SampleDensity(wp, ref surfaceNoise, ref caveNoise, borderNoises, applySurfaceShell);
    }

    /// <summary>
    /// Core density function. Returns blockId (byte):
    ///   0          = Air
    ///   1          = generic Solid (fallback when no layers defined)
    ///   2, 3, 4… = geological layer block ids
    ///
    /// Pipeline (top = highest priority):
    ///   Layer 0  Floor            — below minSubsurfaceHeight → Solid (id 1)
    ///   Layer 1  Player mods      — explicit Air/Solid overrides
    ///   Layer 2  Procedural surface — above surfaceY → Air
    ///   Layer 3  Caves            — underground noise carving → Air
    ///   Layer 4  Geological layers — assigns the correct block id
    /// </summary>
    private byte SampleDensity(float3 wp,
        ref FastNoiseLite surfaceNoise, ref FastNoiseLite caveNoise,
        NativeArray<FastNoiseLite> borderNoises,
        bool applySurfaceShell)
    {
        // ── Layer 0 : Floor ────────────────────────────────────────────────
        if (wp.y < minSubsurfaceHeight)
            return SolidId(wp); // unconditional solid

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
                if (modValues[i] == 1)
                    return applySurfaceShell ? SolidId(wp) : (byte)0;
                return SolidId(wp);
            }
        }

        // ── Layer 2 : Procedural surface ──────────────────────────────────
        float maxSurface = surfaceBaseHeight + surfaceNoiseAmplitude;
        float minSurface = surfaceBaseHeight - surfaceNoiseAmplitude;

        if (wp.y > maxSurface)
            return 0; // Air — definitely above terrain

        float surfaceY = wp.y < minSurface
            ? maxSurface // definitely below — skip noise
            : SampleSurfaceHeight(wp.x, wp.z, ref surfaceNoise);

        if (wp.y > surfaceY)
            return 0; // Air — above surface

        // ── Layer 3 : Caves ────────────────────────────────────────────────
        if (cavesEnabled && !applySurfaceShell)
        {
            float depthBelowSurface = surfaceY - wp.y;
            float fade = caveSurfaceFadeRange > 0f
                ? math.saturate(depthBelowSurface / caveSurfaceFadeRange)
                : 1f;

            float cn = caveNoise.GetNoise(wp.x, wp.y * caveNoiseAmplitudeY, wp.z);
            float effectiveThreshold = math.lerp(1f, caveNoiseThreshold, fade);

            if (cn > effectiveThreshold)
                return 0; // Air — inside a cave
        }

        // ── Layer 4 : Geological layers ────────────────────────────────────
        // Walk layers top-to-bottom and find which one owns this voxel.
        if (geoLayers.Length > 0)
        {
            float depth = surfaceY - wp.y; // depth below local surface

            for (int i = 0; i < geoLayers.Length; i++)
            {
                var layer = geoLayers[i];

                // Use the pre-created border noise instance for this layer
                var bn = borderNoises[i];
                float borderDisplace = bn.GetNoise(wp.x, wp.z)
                                       * layer.borderNoiseAmplitude;
                float layerTopDepth = layer.baseDepth + borderDisplace;

                if (depth < layerTopDepth)
                {
                    // Above this layer's (displaced) top — belongs to previous layer
                    // (or layer 0 if this is the first layer)
                    if (i == 0) return 1; // fallback solid before first layer
                    return geoLayers[i - 1].blockId;
                }

                // Last layer — everything below goes here (bedrock)
                if (i == geoLayers.Length - 1)
                    return layer.blockId;
            }
        }

        // Fallback: no layers defined → generic solid (id 1)
        return 1;
    }

    /// <summary>
    /// Returns the solid block id at wp using geological layers,
    /// without re-evaluating Air/cave conditions.
    /// Used by Layer 0 (floor) and Layer 1 (solid mod).
    /// </summary>
    private byte SolidId(float3 wp)
    {
        if (geoLayers.Length == 0) return 1;

        // Use the last (deepest / bedrock) layer as the guaranteed solid
        // for anything below the surface or forced solid by mods.
        // A more accurate version would re-run the layer walk — but for
        // floor/mod cases the exact layer matters less than returning non-zero.
        float surfaceApprox = surfaceBaseHeight;
        float depth = surfaceApprox - wp.y;

        for (int i = 0; i < geoLayers.Length; i++)
        {
            var layer = geoLayers[i];
            if (depth < layer.baseDepth + layer.borderNoiseAmplitude) // rough check
            {
                if (i == 0) return 1;
                return geoLayers[i - 1].blockId;
            }
            if (i == geoLayers.Length - 1) return layer.blockId;
        }
        return 1;
    }

    // ═══════════════════════════════════════════════════════════════════════

    private static float PosMod(float x, float m)
    {
        float r = x % m;
        return r < 0f ? r + m : r;
    }

    private float SampleSurfaceHeight(float wxWorld, float wzWorld, ref FastNoiseLite surfaceNoise)
    {
        float wx = PosMod(wxWorld, worldSizeX);
        float wz = PosMod(wzWorld, worldSizeZ);

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

        return surfaceBaseHeight + (n1 * 0.6f + n2 * 0.3f + n3 * 0.1f) * surfaceNoiseAmplitude;
    }
}