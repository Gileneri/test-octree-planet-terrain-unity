using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Static pool for chunk GameObjects. Procedural voxel terrain churns
/// hundreds of chunks per second when the player flies fast — at that
/// rate <c>Object.Instantiate</c> + <c>Object.Destroy</c> become a
/// dominant cost (managed allocation, transform hierarchy juggling,
/// component awakening, GC pressure on Mesh/MeshFilter wrapper objects).
///
/// The pool keeps inactive instances alive between use, so creating a
/// new node is reduced to <c>SetActive(true)</c> + a few transform
/// writes, and tearing one down is just <c>SetActive(false)</c>.
///
/// Notes
/// ─────
/// • All chunks share the same prefab in this project, so we keep a
///   single stack.
/// • <see cref="Acquire"/> falls back to <c>Object.Instantiate</c> when
///   the pool is empty.
/// • <see cref="Release"/> caps the pool at <see cref="MaxPoolSize"/>;
///   excess instances are destroyed so the pool can't grow unbounded
///   if the player crosses a huge boundary then never travels back.
/// • Pooled GameObjects keep whatever components they had previously
///   (e.g. a MeshCollider attached during a fine-LOD lifetime). When a
///   new Node uses them, the constructor reads/adds the collider only
///   when needed and leaves stale components dormant otherwise — empty
///   colliders with sharedMesh=null are essentially free.
/// </summary>
public static class ChunkObjectPool
{
    private static readonly Stack<GameObject> pool = new Stack<GameObject>();

    /// <summary>
    /// Hard upper bound on cached instances. Picked to comfortably cover
    /// the typical max in-flight node count for a renderRadius of ~10
    /// cells with chunkResolution=32 plus some headroom.
    /// </summary>
    private const int MaxPoolSize = 8192;

    /// <summary>
    /// Pulls a GameObject out of the pool, or creates a fresh one from
    /// the supplied prefab if the pool is empty. The returned instance
    /// is already <c>SetActive(true)</c> and its transform is unparented
    /// (callers re-parent immediately).
    /// </summary>
    public static GameObject Acquire(GameObject prefab)
    {
        while (pool.Count > 0)
        {
            var go = pool.Pop();
            // Pool entries can become null when the scene is unloaded
            // and Unity destroys their underlying objects from under us.
            // Skip those and try the next one.
            if (go == null) continue;

            go.SetActive(true);
            return go;
        }

        return Object.Instantiate(prefab);
    }

    /// <summary>
    /// Returns a GameObject to the pool for later reuse. Callers must
    /// detach any mesh/collider data themselves before releasing —
    /// otherwise stale Mesh references would survive across a recycle
    /// and pin Mesh assets in memory. The pool only takes care of
    /// hierarchy and active-state housekeeping.
    /// </summary>
    public static void Release(GameObject go)
    {
        if (go == null) return;

        if (pool.Count >= MaxPoolSize)
        {
            Object.Destroy(go);
            return;
        }

        // Detach from current parent so the pool doesn't keep a chain
        // of orphan transforms attached to a since-destroyed octree.
        var t = go.transform;
        if (t.parent != null) t.SetParent(null, worldPositionStays: false);

        go.SetActive(false);
        pool.Push(go);
    }

    /// <summary>
    /// Drops every cached instance. Call from OctreeGrid.OnDestroy /
    /// scene teardown so the pool doesn't outlive the scene that owns
    /// the prefab.
    /// </summary>
    public static void Clear()
    {
        while (pool.Count > 0)
        {
            var go = pool.Pop();
            if (go != null) Object.Destroy(go);
        }
    }

    /// <summary>Diagnostic: current count of pooled instances.</summary>
    public static int Count => pool.Count;
}
