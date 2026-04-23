using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;

/// <summary>
/// Singleton that stores every player-made voxel modification.
///
/// Key   : world-space voxel position wrapped into [0, worldSizeX) × (Y free) × [0, worldSizeZ).
///         Wrapping is done here so lookups from NodeJob (which also wraps) always match.
/// Value : VoxelState — Air (dug out) or Solid (placed block).
///
/// Thread-safety note:
///   The dictionary is written only from the main thread (BlockInteraction).
///   NodeJob reads a NativeHashMap copy that is prepared before the job runs.
///   So there is no concurrent access issue.
/// </summary>
public class WorldModifications : MonoBehaviour
{
    public static WorldModifications Instance { get; private set; }

    // Exposed so Octree can read worldSize when preparing NativeHashMap copies
    [HideInInspector] public float worldSizeX;
    [HideInInspector] public float worldSizeZ;

    public enum VoxelState : byte
    {
        Solid = 0,  // player placed a block here
        Air = 1,  // player dug this block out
    }

    // Main storage: wrapped voxel position → state override
    private readonly Dictionary<Vector3Int, VoxelState> modifications
        = new Dictionary<Vector3Int, VoxelState>();

    // -----------------------------------------------------------------------

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
    }

    // -----------------------------------------------------------------------
    //  Public API (called from main thread only)
    // -----------------------------------------------------------------------

    /// <summary>Sets a voxel override at the given world position.</summary>
    public void Set(Vector3 worldPos, VoxelState state)
    {
        Vector3Int key = WrapKey(worldPos);
        modifications[key] = state;
    }

    /// <summary>Removes any override at the given world position (reverts to noise).</summary>
    public void Remove(Vector3 worldPos)
    {
        modifications.Remove(WrapKey(worldPos));
    }

    /// <summary>Returns true if an override exists and writes it to <paramref name="state"/>.</summary>
    public bool TryGet(Vector3 worldPos, out VoxelState state)
    {
        return modifications.TryGetValue(WrapKey(worldPos), out state);
    }

    /// <summary>Total number of stored modifications (for debug UI).</summary>
    public int Count => modifications.Count;

    /// <summary>
    /// Copies all modifications into a flat array suitable for passing to a Burst job.
    /// Only modifications whose wrapped XZ falls inside the node's AABB are included,
    /// so the array stays small for each individual job.
    /// </summary>
    public void GetModificationsForNode(
        Vector3 nodeWorldPos, float nodeScale, int chunkResolution,
        out Vector3Int[] keys, out VoxelState[] values)
    {
        // Expand the AABB by exactly 1 voxel on every side.
        // When the NodeJob checks a neighbour voxel that sits just outside
        // this chunk's border (local index = -1 or chunkResolution), it
        // converts that to a world position and calls IsAirWorldPos, which
        // looks up modKeys. Without the expansion, modifications in the
        // adjacent chunk are not in modKeys, so the job falls back to noise
        // and renders a phantom face / missing face on the seam.
        float voxelSize = nodeScale / chunkResolution;
        float half = nodeScale * 0.5f + voxelSize;   // +1 voxel border
        var kList = new List<Vector3Int>();
        var vList = new List<VoxelState>();

        foreach (var kv in modifications)
        {
            // Un-wrap the key back to a "closest" world position near the node
            // so we can do an AABB test properly.
            Vector3 wp = ClosestWorldPos(kv.Key, nodeWorldPos);
            if (wp.x >= nodeWorldPos.x - half && wp.x <= nodeWorldPos.x + half &&
                wp.y >= nodeWorldPos.y - half && wp.y <= nodeWorldPos.y + half &&
                wp.z >= nodeWorldPos.z - half && wp.z <= nodeWorldPos.z + half)
            {
                kList.Add(kv.Key);
                vList.Add(kv.Value);
            }
        }

        keys = kList.ToArray();
        values = vList.ToArray();
    }

    // -----------------------------------------------------------------------
    //  Internal helpers
    // -----------------------------------------------------------------------

    /// <summary>
    /// Converts a world position to a wrapped integer voxel key.
    /// We use Mathf.FloorToInt so sub-voxel positions map to the same cell.
    /// </summary>
    public Vector3Int WrapKey(Vector3 worldPos)
    {
        int x = Mathf.FloorToInt(PosMod(worldPos.x, worldSizeX));
        int y = Mathf.FloorToInt(worldPos.y);                      // Y is never wrapped
        int z = Mathf.FloorToInt(PosMod(worldPos.z, worldSizeZ));
        return new Vector3Int(x, y, z);
    }

    /// <summary>
    /// Given a wrapped key and a reference world position (node centre),
    /// returns the world-space position closest to the reference — used for
    /// AABB culling across the wrap boundary.
    /// </summary>
    private Vector3 ClosestWorldPos(Vector3Int key, Vector3 reference)
    {
        float x = key.x;
        float z = key.z;

        // Shift x by ±worldSizeX if that brings it closer to the reference
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