using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;

/// <summary>
/// Singleton that stores every player-made voxel modification.
///
/// ARCHITECTURE — Position in the Density Pipeline
/// ─────────────────────────────────────────────────
/// This class is Layer 1 of NodeJob's density pipeline.
///
///   Layer 0  FLOOR          — enforced here first: Set() rejects Air mods
///                             below minSubsurfaceHeight so invalid digs
///                             are never even recorded.
///   Layer 1  MODIFICATIONS  — this class (player digs / placements)
///   Layer 2  SURFACE NOISE  — procedural terrain height
///   Layer 3  CAVE NOISE     — procedural caves (enable in WorldConfig)
///
/// THREAD-SAFETY
/// ─────────────
/// Written only from the main thread (BlockInteraction).
/// NodeJob reads a NativeArray snapshot prepared before scheduling.
/// No concurrent access.
/// </summary>
public class WorldModifications : MonoBehaviour
{
    public static WorldModifications Instance { get; private set; }

    // Populated by WorldConfigLoader so this class can enforce the floor
    // before a dig attempt is recorded (gives BlockInteraction immediate feedback)
    [HideInInspector] public float worldSizeX;
    [HideInInspector] public float worldSizeZ;
    [HideInInspector] public float minSubsurfaceHeight = float.NegativeInfinity;

    public enum VoxelState : byte
    {
        Solid = 0,  // player placed a block here
        Air = 1,  // player dug this block out
    }

    private readonly Dictionary<Vector3Int, VoxelState> modifications
        = new Dictionary<Vector3Int, VoxelState>();

    // -----------------------------------------------------------------------

    private void Awake()
    {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
    }

    // -----------------------------------------------------------------------
    //  Public API  (main thread only)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Records a voxel override at the given world position.
    ///
    /// Air modifications below minSubsurfaceHeight are silently rejected so
    /// the floor is enforced at the data layer, not just at render time.
    /// Returns false if the modification was rejected (use to give the
    /// player feedback — e.g. play a "can't dig here" sound).
    /// </summary>
    public bool Set(Vector3 worldPos, VoxelState state)
    {
        if (state == VoxelState.Air && worldPos.y < minSubsurfaceHeight)
            return false;

        modifications[WrapKey(worldPos)] = state;
        return true;
    }

    /// <summary>Removes any override, reverting the voxel to procedural generation.</summary>
    public void Remove(Vector3 worldPos) => modifications.Remove(WrapKey(worldPos));

    /// <summary>Returns true if an override exists for this position.</summary>
    public bool TryGet(Vector3 worldPos, out VoxelState state)
        => modifications.TryGetValue(WrapKey(worldPos), out state);

    /// <summary>
    /// Returns true if digging is permitted at this Y coordinate.
    /// Call this in BlockInteraction before Set() to give immediate UI feedback.
    /// </summary>
    public bool IsAboveFloor(float worldY) => worldY >= minSubsurfaceHeight;

    /// <summary>Total number of stored modifications (for the debug HUD).</summary>
    public int Count => modifications.Count;

    /// <summary>
    /// Writes all modifications that overlap a node's AABB directly into
    /// freshly-allocated NativeArrays for passing to a Burst job
    /// (NodeJob.modKeys / modValues).
    ///
    /// The AABB is expanded by 1 voxel so NodeJob can correctly evaluate
    /// neighbour voxels that sit just outside this chunk's border — needed
    /// for correct face culling at seams between chunks.
    ///
    /// PERFORMANCE
    /// ───────────
    /// This is on the hot meshing path — called once per chunk every time it
    /// is (re)scheduled. The previous implementation allocated 2 List&lt;T&gt;,
    /// boxed per-element and then ToArray()'d each one, generating GC garbage
    /// in proportion to the number of meshing jobs in flight. We now do two
    /// simple O(N) passes over the dictionary (first to count, then to fill)
    /// and write straight into the caller's NativeArrays — zero managed
    /// allocations on the typical "no overlapping mods" path, which is
    /// 99%+ of chunks during normal play.
    /// </summary>
    public void GetModificationsForNode(
        Vector3 nodeWorldPos, float nodeScale, int chunkResolution,
        Allocator allocator,
        out NativeArray<int3> keys, out NativeArray<byte> values)
    {
        // Fast path: no modifications anywhere in the world. Most chunks at
        // boot fall here, so the Dictionary iteration cost is skipped entirely.
        if (modifications.Count == 0)
        {
            keys = new NativeArray<int3>(0, allocator);
            values = new NativeArray<byte>(0, allocator);
            return;
        }

        float voxelSize = nodeScale / chunkResolution;
        float half = nodeScale * 0.5f + voxelSize;
        float minX = nodeWorldPos.x - half;
        float maxX = nodeWorldPos.x + half;
        float minY = nodeWorldPos.y - half;
        float maxY = nodeWorldPos.y + half;
        float minZ = nodeWorldPos.z - half;
        float maxZ = nodeWorldPos.z + half;

        // Pass 1 — count overlaps so the NativeArrays can be sized exactly.
        int count = 0;
        foreach (var kv in modifications)
        {
            Vector3 wp = ClosestWorldPos(kv.Key, nodeWorldPos);
            if (wp.x >= minX && wp.x <= maxX &&
                wp.y >= minY && wp.y <= maxY &&
                wp.z >= minZ && wp.z <= maxZ)
                count++;
        }

        keys = new NativeArray<int3>(count, allocator, NativeArrayOptions.UninitializedMemory);
        values = new NativeArray<byte>(count, allocator, NativeArrayOptions.UninitializedMemory);

        if (count == 0) return;

        // Pass 2 — fill the arrays.
        int i = 0;
        foreach (var kv in modifications)
        {
            Vector3 wp = ClosestWorldPos(kv.Key, nodeWorldPos);
            if (wp.x >= minX && wp.x <= maxX &&
                wp.y >= minY && wp.y <= maxY &&
                wp.z >= minZ && wp.z <= maxZ)
            {
                keys[i] = new int3(kv.Key.x, kv.Key.y, kv.Key.z);
                values[i] = (byte)kv.Value;
                i++;
            }
        }
    }

    // -----------------------------------------------------------------------
    //  Key helpers
    // -----------------------------------------------------------------------

    /// <summary>
    /// Converts a world position to a wrapped integer voxel key.
    /// Must match the key computation in NodeJob.SampleDensity() exactly.
    /// </summary>
    public Vector3Int WrapKey(Vector3 worldPos)
    {
        int x = Mathf.FloorToInt(PosMod(worldPos.x, worldSizeX));
        int y = Mathf.FloorToInt(worldPos.y);   // Y is never wrapped
        int z = Mathf.FloorToInt(PosMod(worldPos.z, worldSizeZ));
        return new Vector3Int(x, y, z);
    }

    private Vector3 ClosestWorldPos(Vector3Int key, Vector3 reference)
    {
        float x = key.x;
        float z = key.z;

        if (Mathf.Abs(x + worldSizeX - reference.x) < Mathf.Abs(x - reference.x)) x += worldSizeX;
        else if (Mathf.Abs(x - worldSizeX - reference.x) < Mathf.Abs(x - reference.x)) x -= worldSizeX;

        if (Mathf.Abs(z + worldSizeZ - reference.z) < Mathf.Abs(z - reference.z)) z += worldSizeZ;
        else if (Mathf.Abs(z - worldSizeZ - reference.z) < Mathf.Abs(z - reference.z)) z -= worldSizeZ;

        return new Vector3(x, key.y, z);
    }

    private static float PosMod(float x, float m)
    {
        float r = x % m;
        return r < 0f ? r + m : r;
    }
}