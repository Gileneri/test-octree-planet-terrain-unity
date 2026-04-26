using System.Collections.Generic;
using UnityEngine;

public class Octree : MonoBehaviour
{
    [HideInInspector] public float worldSizeX;
    [HideInInspector] public float worldSizeZ;
    [HideInInspector] public float surfaceBaseHeight;
    [HideInInspector] public float surfaceNoiseAmplitude;
    [HideInInspector] public int divisions;
    [HideInInspector] public int chunkResolution;
    [HideInInspector] public int maxNodeCreationsPerFrame;
    [HideInInspector] public Transform priority;
    [HideInInspector] public GameObject chunkPrefab;
    [HideInInspector] public Vector3 cellOrigin;

    // ── WorldConfig-driven fields (added) ────────────────────────────────
    /// <summary>FastNoiseLite seed forwarded to every NodeJob.</summary>
    [HideInInspector] public int noiseSeed = 2376;

    /// <summary>
    /// Integer noise type id forwarded to every NodeJob.
    /// Convert with NodeJob.NoiseTypeToId() before assigning.
    /// </summary>
    [HideInInspector] public int noiseTypeId = 0; // 0 = OpenSimplex2

    /// <summary>
    /// Hard floor Y.  Voxels below this are always solid.
    /// Forwarded to every NodeJob.
    /// </summary>
    [HideInInspector] public float minSubsurfaceHeight = -500f;
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Per-level subdivision radii baked by OctreeGrid from lodDistanceCurve.
    /// Index 0 = finest level (divisions==1).
    /// Index divisions-1 = coarsest (root).
    /// Node subdivides when player is closer than lodRadii[level] to its AABB.
    /// </summary>
    [HideInInspector] public float[] lodRadii;

    /// <summary>
    /// Frustum planes updated every frame by OctreeGrid before calling Tick.
    /// CollectJobs uses these to skip mesh generation for nodes fully outside
    /// the camera frustum. Traverse is NOT affected — the tree structure must
    /// stay correct for all directions so nodes are ready when the player turns.
    /// </summary>
    [HideInInspector] public Plane[] frustumPlanes;

    [HideInInspector] public Node root;

    private bool initialised = false;

    // -----------------------------------------------------------------------

    public void Initialize()
    {
        int nodeResolution = (int)Mathf.Pow(2, divisions - 1);
        float nodeScale = chunkResolution * nodeResolution;

        Vector3 rootOffset = cellOrigin + new Vector3(nodeScale, nodeScale, nodeScale) * 0.5f;
        rootOffset.y = surfaceBaseHeight;

        root = new Node(this, null, rootOffset, divisions);
        initialised = true;
    }

    public void Tick(List<OctreeGrid.PrioritizedJob> jobList, Vector3 playerPos, Plane[] planes)
    {
        if (!initialised) return;
        frustumPlanes = planes;
        int creations = 0;
        Traverse(root, playerPos, ref creations);
        CollectJobs(root, jobList, playerPos);
    }

    public void Shutdown()
    {
        if (root != null) DestroyTree(root);
        root = null;
        initialised = false;
        Destroy(gameObject);
    }

    /// <summary>
    /// Destroys and recreates the entire octree from the root.
    /// Called by OctreeGrid.RebakeLodRadii so LOD changes are visible immediately.
    /// </summary>
    public void ForceRebuildTree()
    {
        if (root != null) DestroyTree(root);

        int nodeResolution = (int)UnityEngine.Mathf.Pow(2, divisions - 1);
        float nodeScale = chunkResolution * nodeResolution;
        Vector3 rootOffset = cellOrigin + new Vector3(nodeScale, nodeScale, nodeScale) * 0.5f;
        rootOffset.y = surfaceBaseHeight;

        root = new Node(this, null, rootOffset, divisions);
        // initialised stays true — Tick() will traverse and mesh immediately
    }

    // -----------------------------------------------------------------------
    //  Traversal
    // -----------------------------------------------------------------------

    private void Traverse(Node node, Vector3 playerPos, ref int creations)
    {
        if (creations > maxNodeCreationsPerFrame) return;

        if (node.divisions > 1)
        {
            if (ShouldSubdivide(node, playerPos))
            {
                if (node.children == null)
                {
                    node.Clear();
                    node.children = new Node[8];
                    for (int i = 0; i < 8; i++)
                        node.children[i] = new Node(this, node, Tables.Offsets[i], node.divisions - 1);
                    creations++;
                }
            }
            else
            {
                if (node.children != null)
                {
                    node.MarkDirty();
                    DeleteChildren(node);
                }
            }

            if (node.children != null)
                for (int i = 0; i < 8; i++)
                    Traverse(node.children[i], playerPos, ref creations);
        }
    }

    // -----------------------------------------------------------------------
    //  LOD decision — AABB distance, not center distance
    // -----------------------------------------------------------------------

    /// <summary>
    /// Squared distance from the player to the nearest point on the node's AABB.
    /// Returns 0 when the player is inside the node.
    ///
    /// Why AABB and not center?
    /// A node 1024 units wide has its center up to ~866 units from a corner.
    /// If the player stands just inside that corner, center-distance would be
    /// ~866 units, likely beyond any subdivision radius — so the node never
    /// subdivides even with the player standing in it. AABB distance is 0 in
    /// that case, guaranteeing subdivision regardless of node size.
    /// </summary>
    private static float AabbSqrDist(Node node, Vector3 p)
    {
        Vector3 c = node.NodePosition();
        float h = node.NodeScale() * 0.5f;

        float dx = Mathf.Max(0f, Mathf.Abs(p.x - c.x) - h);
        float dy = Mathf.Max(0f, Mathf.Abs(p.y - c.y) - h);
        float dz = Mathf.Max(0f, Mathf.Abs(p.z - c.z) - h);

        return dx * dx + dy * dy + dz * dz;
    }

    private bool ShouldSubdivide(Node node, Vector3 playerPos)
    {
        int level = node.divisions - 1; // 0 = finest, divisions-1 = coarsest
        float radius = (lodRadii != null && level < lodRadii.Length)
            ? lodRadii[level]
            : node.NodeScale() * 2f;   // fallback

        return AabbSqrDist(node, playerPos) < radius * radius;
    }

    // -----------------------------------------------------------------------
    //  Job collection
    // -----------------------------------------------------------------------

    private void CollectJobs(Node node, List<OctreeGrid.PrioritizedJob> jobList, Vector3 playerPos)
    {
        // Cull entire subtree when its AABB is completely outside the frustum.
        // TestPlanesAABB returns false when all 6 planes reject the box.
        // We use a slightly expanded bounds so nodes right at the edge of the
        // frustum (whose faces are still partially visible) are never skipped.
        if (frustumPlanes != null && !IsAabbInFrustum(node)) return;

        if (node.TryCollectJob(out JobCompleter completer))
        {
            jobList.Add(new OctreeGrid.PrioritizedJob
            {
                completer = completer,
                sqrDistToPlayer = AabbSqrDist(node, playerPos),
            });
        }

        if (node.children == null) return;
        for (int i = 0; i < node.children.Length; i++)
            CollectJobs(node.children[i], jobList, playerPos);
    }

    /// <summary>
    /// Returns true if the node's world-space AABB intersects or is inside
    /// the camera frustum. Returns false only when the box is fully outside
    /// at least one frustum plane — meaning nothing in the subtree is visible.
    ///
    /// We expand the test radius by 10% so nodes whose geometry slightly
    /// extends past the calculated AABB (due to voxel rounding) are never
    /// incorrectly culled at the screen edge.
    /// </summary>
    private bool IsAabbInFrustum(Node node)
    {
        Vector3 center = node.NodePosition();
        float half = node.NodeScale() * 0.5f * 1.1f; // 10% expansion
        var bounds = new Bounds(center, Vector3.one * (half * 2f));
        return GeometryUtility.TestPlanesAABB(frustumPlanes, bounds);
    }

    // -----------------------------------------------------------------------
    //  FindLeafAt
    // -----------------------------------------------------------------------

    public Node FindLeafAt(Vector3 worldPos)
    {
        if (root == null) return null;
        return FindLeafAt(root, worldPos);
    }

    private Node FindLeafAt(Node node, Vector3 worldPos)
    {
        Vector3 center = node.NodePosition();
        float half = node.NodeScale() * 0.5f;

        if (worldPos.x < center.x - half || worldPos.x > center.x + half) return null;
        if (worldPos.y < center.y - half || worldPos.y > center.y + half) return null;
        if (worldPos.z < center.z - half || worldPos.z > center.z + half) return null;

        if (node.IsLeaf()) return node;

        foreach (var child in node.children)
        {
            var r = FindLeafAt(child, worldPos);
            if (r != null) return r;
        }
        return null;
    }

    // -----------------------------------------------------------------------
    //  Cleanup
    // -----------------------------------------------------------------------

    private void DestroyTree(Node node)
    {
        if (node == null) return;
        if (node.children != null)
            foreach (var c in node.children) DestroyTree(c);
        node.Destroy();
    }

    public void DeleteChildren(Node node)
    {
        if (node.children == null) return;
        foreach (var c in node.children) DeleteChildren(c);
        foreach (var c in node.children) c.Destroy();
        node.children = null;
    }
}