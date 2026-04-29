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

    // ── Caves ─────────────────────────────────────────────────────────────
    [HideInInspector] public bool cavesEnabled = false;
    [HideInInspector] public float caveNoiseFrequency = 0.008f;
    [HideInInspector] public float caveNoiseThreshold = 0.55f;
    [HideInInspector] public float caveNoiseAmplitudeY = 0.4f;
    [HideInInspector] public float caveSurfaceFadeRange = 20f;
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

    // ── Collider budget ───────────────────────────────────────────────────
    /// <summary>
    /// Only nodes at or below this division level receive a MeshCollider.
    /// Coarser nodes skip the collider — their voxels are too large for
    /// PhysX (triggers the >500u triangle warning).
    /// Computed by OctreeGrid: highest division where voxelSize <= 32 units.
    /// </summary>
    [HideInInspector] public int maxColliderDivisions = 2;

    // ── Geological layers ─────────────────────────────────────────────────
    /// <summary>Blittable layer data forwarded to every NodeJob.</summary>
    [HideInInspector] public GeologicalLayerBlob[] geoLayerBlobs = new GeologicalLayerBlob[0];

    // ── LOD vertical bias ─────────────────────────────────────────────────
    /// <summary>
    /// How much more "expensive" vertical distance is compared to horizontal
    /// distance for LOD subdivision decisions.
    ///
    /// 1.0 = isotropic (current behaviour — subdivides equally in all directions)
    /// 2.0 = vertical distance counts 2× → nodes above/below collapse faster
    /// 3.0 = recommended default — nodes 2 cells below player behave like they
    ///       are 6 cells away horizontally, avoiding unnecessary underground
    ///       subdivision when the player is on the surface.
    ///
    /// Does NOT affect CollectJobs culling — only subdivision depth.
    /// Set via OctreeGrid.verticalLodBias and propagated in CreateOctree.
    /// </summary>
    [HideInInspector] public float verticalLodBias = 3f;

    // ── Underground culling ───────────────────────────────────────────────
    /// <summary>
    /// Absolute world-space Y threshold used by CollectJobs.
    /// Nodes whose top is below this value are skipped.
    /// </summary>
    [HideInInspector] public float undergroundCullBelowY = float.NegativeInfinity;
    [HideInInspector] public bool cullEnclosedUndergroundFaces = true;
    [HideInInspector] public float enclosedUndergroundFaceMargin = 2f;
    [HideInInspector] public bool useSurfaceShellOnCoarseLod = true;
    [HideInInspector] public int surfaceShellMinDivisions = 4;
    [HideInInspector] public bool cullUndergroundFacesOnCoarseLod = true;
    [HideInInspector] public int coarseLodFaceCullMinDivisions = 4;
    [HideInInspector] public float coarseLodFaceCullMargin = 1.5f;
    [HideInInspector] public int coarseLodMinSurfaceVoxels = 2;
    // ─────────────────────────────────────────────────────────────────────

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

    public void Tick(
        List<OctreeGrid.PrioritizedJob> jobList,
        Vector3 playerPos,
        Plane[] planes,
        float cullBelowY)
    {
        if (!initialised) return;
        frustumPlanes = planes;
        undergroundCullBelowY = cullBelowY;
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
    /// Isotropic AABB squared distance — used for job priority sorting and
    /// frustum/depth culling where physical distance is what matters.
    /// Returns 0 when the player is inside the node.
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

    /// <summary>
    /// Bias-weighted AABB squared distance used exclusively for LOD subdivision.
    ///
    /// Vertical distance (dy) is multiplied by verticalLodBias before squaring,
    /// making nodes above/below the player appear "further away" in LOD terms.
    /// This collapses underground nodes faster without affecting horizontal detail.
    ///
    /// Example with bias=3: a node 300u below the player is treated as if it
    /// were 900u away horizontally — well past most subdivision radii when the
    /// player is on the surface.
    ///
    /// CollectJobs still uses AabbSqrDist (unbiased) for culling and job
    /// priority, so meshing order and frustum culling are unaffected.
    /// </summary>
    private float AabbSqrDistLod(Node node, Vector3 p)
    {
        Vector3 c = node.NodePosition();
        float h = node.NodeScale() * 0.5f;

        float dx = Mathf.Max(0f, Mathf.Abs(p.x - c.x) - h);
        float dy = Mathf.Max(0f, Mathf.Abs(p.y - c.y) - h) * verticalLodBias;
        float dz = Mathf.Max(0f, Mathf.Abs(p.z - c.z) - h);

        return dx * dx + dy * dy + dz * dz;
    }

    private bool ShouldSubdivide(Node node, Vector3 playerPos)
    {
        int level = node.divisions - 1; // 0 = finest, divisions-1 = coarsest
        float radius = (lodRadii != null && level < lodRadii.Length)
            ? lodRadii[level]
            : node.NodeScale() * 2f;   // fallback

        // Use bias-weighted distance so vertical nodes collapse sooner
        return AabbSqrDistLod(node, playerPos) < radius * radius;
    }

    // -----------------------------------------------------------------------
    //  Job collection
    // -----------------------------------------------------------------------

    private void CollectJobs(Node node, List<OctreeGrid.PrioritizedJob> jobList, Vector3 playerPos)
    {
        // ── Frustum culling ───────────────────────────────────────────────
        // Skip entire subtree when AABB is fully outside camera frustum.
        if (frustumPlanes != null && !IsAabbInFrustum(node)) return;

        // ── Underground culling below guaranteed-surface threshold ───────
        // Skip nodes completely below the configured cull altitude.
        // This avoids meshing deep underground volumes that are enclosed by
        // solid terrain and cannot be seen from outside the planet.
        {
            Vector3 nodeCenter = node.NodePosition();
            float nodeHalfY = node.NodeScale() * 0.5f;
            float nodeTopY = nodeCenter.y + nodeHalfY;
            if (nodeTopY < undergroundCullBelowY)
                return;
        }

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