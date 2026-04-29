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

        // ═══════════════════════════════════════════════════════════════════
        //  Greedy meshing — per face direction, per slice
        //
        //  For each of the 6 face directions, sweep slice by slice along the
        //  perpendicular axis. For each slice, build a 2D mask where each
        //  cell holds the source voxel's blockId if a face should be emitted
        //  there (and 0 otherwise — air voxel, hidden by neighbour, or culled
        //  by underground rules). Then greedy-merge contiguous same-blockId
        //  cells into rectangular quads with per-voxel UV tiling so the
        //  texture never stretches across the merged area.
        //
        //  WHY THIS IS FASTER
        //  ──────────────────
        //  The naïve per-voxel approach emits up to 1 quad per voxel per
        //  face direction (6 × res³ quads worst case). Greedy meshing on
        //  flat or large-flat regions collapses entire planes into a single
        //  quad — typical ground/sky terrain shrinks vertex/index counts by
        //  5–20× depending on the noise profile, which is the difference
        //  between meshing-bound and rendering-bound on the GPU.
        //
        //  PER-BLOCK-TYPE MERGING
        //  ──────────────────────
        //  Cells in the mask only merge when they share the same blockId
        //  AND the same face direction's cull/visibility decision, so a
        //  grass block never absorbs a neighbouring stone block's face.
        //  Texture array layer is recomputed from blockId after merging.
        //
        //  PER-VOXEL UV TILING
        //  ───────────────────
        //  A merged quad of size w × h voxels gets UV corners (0,0),
        //  (w,0), (0,h), (w,h) instead of (0,0)…(1,1). With the texture
        //  set to Repeat (or a tiling-aware shader sampling the texture
        //  array), the surface looks identical to per-voxel quads — no
        //  stretching when the player breaks/places a single block.
        // ═══════════════════════════════════════════════════════════════════
        var mask = new NativeArray<byte>(chunkResolution * chunkResolution,
            Allocator.Temp, NativeArrayOptions.UninitializedMemory);

        for (int side = 0; side < 6; side++)
        {
            for (int slice = 0; slice < chunkResolution; slice++)
            {
                BuildSliceMask(side, slice, mask, blockIds, reachableAir, topSurfaceYByXZ,
                    applyCoarseUndergroundFaceCull, minSurfaceVoxelsToKeep,
                    ref surfaceNoise, ref caveNoise, borderNoises, applySurfaceShell);

                GreedyEmitSlice(side, slice, mask,
                    vertices, normals, uvs, blockData, indices,
                    ref vi, ref ii);
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
    //  Greedy meshing helpers
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Maps a (slice, mask_u, mask_v) triplet for a given face direction
    /// to the source voxel coordinate (x,y,z) and the neighbour voxel
    /// coordinate (nx,ny,nz) on the air side of the face.
    /// </summary>
    private void GetSliceVoxelCoords(int side, int slice, int u, int v,
        out int x, out int y, out int z, out int nx, out int ny, out int nz)
    {
        switch (side)
        {
            case 0: // +X — u=Y, v=Z
                x = slice; y = u; z = v; nx = slice + 1; ny = u; nz = v; break;
            case 1: // -X — u=Y, v=Z
                x = slice; y = u; z = v; nx = slice - 1; ny = u; nz = v; break;
            case 2: // +Y — u=Z, v=X
                x = v; y = slice; z = u; nx = v; ny = slice + 1; nz = u; break;
            case 3: // -Y — u=Z, v=X
                x = v; y = slice; z = u; nx = v; ny = slice - 1; nz = u; break;
            case 4: // +Z — u=Y, v=X
                x = v; y = u; z = slice; nx = v; ny = u; nz = slice + 1; break;
            default: // 5: -Z — u=Y, v=X
                x = v; y = u; z = slice; nx = v; ny = u; nz = slice - 1; break;
        }
    }

    /// <summary>
    /// Fills `mask` so that mask[u*res+v] = source-voxel blockId where a
    /// face must be emitted on this side at this slice, or 0 otherwise.
    ///
    /// Applies all the existing face-culling logic (neighbour solidity,
    /// coarse-LOD underground shell, enclosed-underground flood-fill)
    /// at mask-build time so the greedy merge sees a faithful per-voxel
    /// view of which faces should appear.
    /// </summary>
    private void BuildSliceMask(int side, int slice,
        NativeArray<byte> mask, NativeArray<byte> blockIds,
        NativeArray<byte> reachableAir, NativeArray<int> topSurfaceYByXZ,
        bool applyCoarseUndergroundFaceCull, int minSurfaceVoxelsToKeep,
        ref FastNoiseLite surfaceNoise, ref FastNoiseLite caveNoise,
        NativeArray<FastNoiseLite> borderNoises, bool applySurfaceShell)
    {
        int res = chunkResolution;
        int res2 = res * res;

        for (int v = 0; v < res; v++)
        {
            for (int u = 0; u < res; u++)
            {
                int x, y, z, nx, ny, nz;
                GetSliceVoxelCoords(side, slice, u, v, out x, out y, out z, out nx, out ny, out nz);

                byte bid = blockIds[x * res2 + y * res + z];
                if (bid == 0) { mask[u * res + v] = 0; continue; }

                byte nbId;
                bool nbInside = nx >= 0 && nx < res && ny >= 0 && ny < res && nz >= 0 && nz < res;
                if (nbInside)
                {
                    nbId = blockIds[nx * res2 + ny * res + nz];
                }
                else
                {
                    float3 wp = new float3(nx, ny, nz) * normalizedVoxelScale
                                + worldNodePosition - centerOffset;
                    nbId = SampleDensity(wp, ref surfaceNoise, ref caveNoise, borderNoises, applySurfaceShell);
                }
                if (nbId != 0) { mask[u * res + v] = 0; continue; }

                // Coarse-LOD underground face cull.
                if (applyCoarseUndergroundFaceCull)
                {
                    int topY = topSurfaceYByXZ[x * res + z];
                    int keepFromY = topY >= 0
                        ? math.max(0, topY - (minSurfaceVoxelsToKeep - 1))
                        : int.MaxValue;
                    bool keepMinimumSurfaceBand = topY >= 0 && y >= keepFromY;

                    if (!keepMinimumSurfaceBand)
                    {
                        float3 voxelWp = new float3(x, y, z) * normalizedVoxelScale
                                       + worldNodePosition - centerOffset;
                        float3 nwp = new float3(nx, ny, nz) * normalizedVoxelScale
                                   + worldNodePosition - centerOffset;
                        float3 faceWp = (voxelWp + nwp) * 0.5f;
                        float surfaceY = SampleSurfaceHeight(faceWp.x, faceWp.z, ref surfaceNoise);
                        if (faceWp.y < surfaceY - math.max(0f, coarseLodFaceCullMargin))
                        { mask[u * res + v] = 0; continue; }
                    }
                }

                // Enclosed underground face cull (flood-fill reachability).
                if (cullEnclosedUndergroundFaces && nbInside)
                {
                    int nIdx = nx * res2 + ny * res + nz;
                    if (reachableAir[nIdx] == 0)
                    {
                        float3 nwp = new float3(nx, ny, nz) * normalizedVoxelScale
                                   + worldNodePosition - centerOffset;
                        float surfaceY = SampleSurfaceHeight(nwp.x, nwp.z, ref surfaceNoise);
                        if (nwp.y < surfaceY - math.max(0f, enclosedUndergroundFaceMargin))
                        { mask[u * res + v] = 0; continue; }
                    }
                }

                mask[u * res + v] = bid;
            }
        }
    }

    /// <summary>
    /// Greedy merges contiguous same-blockId cells in the mask into
    /// rectangular quads, emitting each merged region as a single quad
    /// with per-voxel UV tiling. Cells consumed by a merged quad are
    /// zeroed out so they are not re-emitted.
    /// </summary>
    private void GreedyEmitSlice(int side, int slice, NativeArray<byte> mask,
        NativeArray<float3> vertices, NativeArray<float3> normals,
        NativeArray<float2> uvs, NativeArray<float2> blockData,
        NativeArray<uint> indices,
        ref int vi, ref int ii)
    {
        int res = chunkResolution;

        for (int v = 0; v < res; v++)
        {
            int u = 0;
            while (u < res)
            {
                byte bid = mask[u * res + v];
                if (bid == 0) { u++; continue; }

                // Width: extend along u while the same blockId continues.
                int w = 1;
                while (u + w < res && mask[(u + w) * res + v] == bid) w++;

                // Height: extend along v while every column in [u, u+w)
                // still matches the same blockId on row v+h.
                int h = 1;
                while (v + h < res)
                {
                    bool rowOk = true;
                    for (int k = 0; k < w; k++)
                    {
                        if (mask[(u + k) * res + (v + h)] != bid) { rowOk = false; break; }
                    }
                    if (!rowOk) break;
                    h++;
                }

                EmitGreedyFace(side, slice, u, v, w, h, bid,
                    vertices, normals, uvs, blockData, indices,
                    ref vi, ref ii);

                // Clear used area so the outer loop skips it.
                for (int dv = 0; dv < h; dv++)
                {
                    int rowBase = (v + dv);
                    for (int du = 0; du < w; du++)
                        mask[(u + du) * res + rowBase] = 0;
                }

                u += w;
            }
        }
    }

    /// <summary>
    /// Writes one greedy-merged quad (4 verts + 6 indices) into the output
    /// buffers. The quad spans `w` voxels along the side's u-axis and `h`
    /// voxels along the v-axis, with UVs scaled to (w,h) so each voxel-sized
    /// cell on the merged face still gets exactly one tile of the texture.
    /// </summary>
    private void EmitGreedyFace(int side, int slice, int uStart, int vStart, int w, int h,
        byte bid,
        NativeArray<float3> vertices, NativeArray<float3> normals,
        NativeArray<float2> uvs, NativeArray<float2> blockData,
        NativeArray<uint> indices,
        ref int vi, ref int ii)
    {
        float vScale = normalizedVoxelScale;
        float3 v0, v1, v2, v3;

        // For each side, compute the 4 corners in node-local mesh space.
        // The vertex order matches Tables.BuildOrder (BL, BR, TL, TR) so the
        // standard triangulation 0-1-2 / 2-1-3 winds outward correctly.
        switch (side)
        {
            case 0: // +X
            {
                float xPlane = (slice + 1) * vScale;
                float yMin = uStart * vScale, yMax = (uStart + w) * vScale;
                float zMin = vStart * vScale, zMax = (vStart + h) * vScale;
                v0 = new float3(xPlane, yMin, zMin);
                v1 = new float3(xPlane, yMax, zMin);
                v2 = new float3(xPlane, yMin, zMax);
                v3 = new float3(xPlane, yMax, zMax);
                break;
            }
            case 1: // -X
            {
                float xPlane = slice * vScale;
                float yMin = uStart * vScale, yMax = (uStart + w) * vScale;
                float zMin = vStart * vScale, zMax = (vStart + h) * vScale;
                v0 = new float3(xPlane, yMin, zMax);
                v1 = new float3(xPlane, yMax, zMax);
                v2 = new float3(xPlane, yMin, zMin);
                v3 = new float3(xPlane, yMax, zMin);
                break;
            }
            case 2: // +Y
            {
                float yPlane = (slice + 1) * vScale;
                float xMin = vStart * vScale, xMax = (vStart + h) * vScale;
                float zMin = uStart * vScale, zMax = (uStart + w) * vScale;
                v0 = new float3(xMin, yPlane, zMin);
                v1 = new float3(xMin, yPlane, zMax);
                v2 = new float3(xMax, yPlane, zMin);
                v3 = new float3(xMax, yPlane, zMax);
                break;
            }
            case 3: // -Y
            {
                float yPlane = slice * vScale;
                float xMin = vStart * vScale, xMax = (vStart + h) * vScale;
                float zMin = uStart * vScale, zMax = (uStart + w) * vScale;
                v0 = new float3(xMax, yPlane, zMin);
                v1 = new float3(xMax, yPlane, zMax);
                v2 = new float3(xMin, yPlane, zMin);
                v3 = new float3(xMin, yPlane, zMax);
                break;
            }
            case 4: // +Z
            {
                float zPlane = (slice + 1) * vScale;
                float xMin = vStart * vScale, xMax = (vStart + h) * vScale;
                float yMin = uStart * vScale, yMax = (uStart + w) * vScale;
                v0 = new float3(xMax, yMin, zPlane);
                v1 = new float3(xMax, yMax, zPlane);
                v2 = new float3(xMin, yMin, zPlane);
                v3 = new float3(xMin, yMax, zPlane);
                break;
            }
            default: // 5: -Z
            {
                float zPlane = slice * vScale;
                float xMin = vStart * vScale, xMax = (vStart + h) * vScale;
                float yMin = uStart * vScale, yMax = (uStart + w) * vScale;
                v0 = new float3(xMin, yMin, zPlane);
                v1 = new float3(xMin, yMax, zPlane);
                v2 = new float3(xMax, yMin, zPlane);
                v3 = new float3(xMax, yMax, zPlane);
                break;
            }
        }

        v0 -= centerOffset;
        v1 -= centerOffset;
        v2 -= centerOffset;
        v3 -= centerOffset;

        vertices[vi + 0] = v0;
        vertices[vi + 1] = v1;
        vertices[vi + 2] = v2;
        vertices[vi + 3] = v3;

        float3 normal = Tables.Normals[side];
        normals[vi + 0] = normal;
        normals[vi + 1] = normal;
        normals[vi + 2] = normal;
        normals[vi + 3] = normal;

        // Per-voxel UV tiling: a 1×1 face has UV span (1,1); a w×h merged
        // face has UV span (w,h) so the texture tiles exactly once per
        // voxel cell — no stretching when the player digs/places blocks.
        uvs[vi + 0] = new float2(0f, 0f);
        uvs[vi + 1] = new float2(w, 0f);
        uvs[vi + 2] = new float2(0f, h);
        uvs[vi + 3] = new float2(w, h);

        // Tex array layer: blockId * 3 + faceSlot
        // (0=top, 1=bottom, 2=side). Side 2=up, side 3=down, others side.
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