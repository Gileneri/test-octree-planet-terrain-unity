using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using System.Diagnostics;
using Unity.Mathematics;

/// <summary>
/// Flat toroidal voxel world with floating-origin streaming and block modifications.
///
/// The octree root re-anchors to the nearest grid cell each frame so terrain
/// always exists around the player.  Noise is sampled with wrapped coordinates
/// so terrain repeats seamlessly every (worldSizeX, worldSizeZ) units.
/// </summary>
public class Octree : MonoBehaviour
{
    [Header("World Size  (terrain repeats every N units)")]
    public float worldSizeX = 4096f;
    public float worldSizeZ = 4096f;

    [Header("Terrain Shape")]
    public float surfaceBaseHeight = 0f;
    public float surfaceNoiseAmplitude = 64f;

    [Header("Octree")]
    public int maxNodeCreationsPerFrame = 50;
    public int chunkResolution = 16;
    public float innerRadiusPadding = 2f;
    public int divisions = 11;

    [Header("References")]
    public Transform priority;
    public GameObject chunkPrefab;

    // -----------------------------------------------------------------------
    [HideInInspector] public Node root;

    private int nodeResolution;
    private float nodeScale;
    private double worldVoxelsCount;
    private readonly List<JobCompleter> toComplete = new List<JobCompleter>();
    private Vector3 currentTreeOrigin = new Vector3(float.MaxValue, 0, float.MaxValue);

    // -----------------------------------------------------------------------
    //  Lifecycle
    // -----------------------------------------------------------------------

    private void Start()
    {
        nodeResolution = (int)Mathf.Pow(2, divisions - 1);
        nodeScale = chunkResolution * nodeResolution;

        worldVoxelsCount = System.Math.Pow(nodeResolution, 3)
                         * System.Math.Pow(chunkResolution, 3);

        // Give WorldModifications the world dimensions so it can wrap keys correctly
        if (WorldModifications.Instance != null)
        {
            WorldModifications.Instance.worldSizeX = worldSizeX;
            WorldModifications.Instance.worldSizeZ = worldSizeZ;
        }

        UnityEngine.Debug.Log(
            $"[Octree] nodeResolution={nodeResolution}  nodeScale={nodeScale}\n" +
            $"[Octree] Terrain repeats every {worldSizeX} x {worldSizeZ} units");

        RebuildTree();
    }

    private void Update()
    {
        Vector3 desired = SnapToGrid(priority.position);
        if (Vector3.Distance(desired, currentTreeOrigin) > nodeScale * 0.1f)
            RebuildTree();

        Traverse(root, 0);
        Schedule(root);
    }

    private void LateUpdate()
    {
        if (toComplete.Count == 0) return;

        var sw = new Stopwatch();
        sw.Start();

        var handles = new NativeArray<JobHandle>(toComplete.Count, Allocator.Temp);
        for (int i = 0; i < toComplete.Count; i++)
            handles[i] = toComplete[i].schedule();
        JobHandle.CompleteAll(handles);
        handles.Dispose();

        for (int i = 0; i < toComplete.Count; i++)
            toComplete[i].onComplete();

        toComplete.Clear();
        sw.Stop();
        UnityEngine.Debug.Log($"[Octree] meshing {sw.ElapsedMilliseconds} ms");
    }

    private void OnDisable()
    {
        if (root != null) DestroyTree(root);
    }

    private void OnGUI()
    {
        GUILayout.Label($"World repeats every: {worldSizeX} x {worldSizeZ}  (toroidal)");
        if (priority != null)
            GUILayout.Label($"Player pos: {priority.position}");
        if (WorldModifications.Instance != null)
            GUILayout.Label($"Modifications: {WorldModifications.Instance.Count}");
    }

    private void OnDrawGizmos()
    {
        if (root == null) return;
        DrawNodeGizmos(root);
        DrawLODRings(divisions);
    }

    // -----------------------------------------------------------------------
    //  Tree rebuild
    // -----------------------------------------------------------------------

    private void RebuildTree()
    {
        if (root != null) DestroyTree(root);
        currentTreeOrigin = SnapToGrid(priority.position);
        root = new Node(this, null, currentTreeOrigin, divisions);
    }

    private Vector3 SnapToGrid(Vector3 pos)
    {
        float snap = nodeScale;
        return new Vector3(
            Mathf.Round(pos.x / snap) * snap,
            surfaceBaseHeight,
            Mathf.Round(pos.z / snap) * snap
        );
    }

    // -----------------------------------------------------------------------
    //  Traversal
    // -----------------------------------------------------------------------

    public void Traverse(Node node, int creations)
    {
        if (creations > maxNodeCreationsPerFrame) return;

        if (node.divisions > 1)
        {
            if (ShouldSubdivide(node))
            {
                node.Clear();
                if (node.children == null)
                {
                    node.children = new Node[8];
                    for (int i = 0; i < 8; i++)
                        node.children[i] = new Node(this, node, Tables.Offsets[i], node.divisions - 1);
                    creations++;
                }
            }
            else
            {
                DeleteChildren(node);
            }

            if (node.children != null)
                for (int i = 0; i < 8; i++)
                    Traverse(node.children[i], creations);
        }
    }

    private bool ShouldSubdivide(Node node)
    {
        Vector3 center = node.NodePosition();
        float halfSize = node.NodeScale() * innerRadiusPadding + 1f;
        Vector3 min = center - Vector3.one * halfSize;
        Vector3 max = center + Vector3.one * halfSize;
        Vector3 pp = priority.position;

        return pp.x >= min.x && pp.x <= max.x
            && pp.y >= min.y && pp.y <= max.y
            && pp.z >= min.z && pp.z <= max.z;
    }

    // -----------------------------------------------------------------------
    //  FindLeafAt  (used by BlockInteraction)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Returns the leaf node whose AABB contains <paramref name="worldPos"/>,
    /// or null if no such node exists in the current tree.
    /// </summary>
    public Node FindLeafAt(Vector3 worldPos)
    {
        if (root == null) return null;
        return FindLeafAt(root, worldPos);
    }

    private Node FindLeafAt(Node node, Vector3 worldPos)
    {
        // Check if worldPos is inside this node's AABB
        Vector3 center = node.NodePosition();
        float half = node.NodeScale() * 0.5f;

        if (worldPos.x < center.x - half || worldPos.x > center.x + half) return null;
        if (worldPos.y < center.y - half || worldPos.y > center.y + half) return null;
        if (worldPos.z < center.z - half || worldPos.z > center.z + half) return null;

        if (node.IsLeaf()) return node;

        foreach (var child in node.children)
        {
            Node result = FindLeafAt(child, worldPos);
            if (result != null) return result;
        }

        return null;
    }

    // -----------------------------------------------------------------------
    //  Scheduling
    // -----------------------------------------------------------------------

    public void Schedule(Node node)
    {
        if (node.TrySchedule(out JobCompleter c)) toComplete.Add(c);
        if (node.children == null) return;
        for (int i = 0; i < node.children.Length; i++)
            Schedule(node.children[i]);
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

    // -----------------------------------------------------------------------
    //  Gizmos
    // -----------------------------------------------------------------------

    private void DrawNodeGizmos(Node node)
    {
        Gizmos.color = Color.red;
        Gizmos.DrawWireCube(node.NodePosition(), Vector3.one * node.NodeScale());
        if (node.children != null)
            foreach (var c in node.children) DrawNodeGizmos(c);
    }

    private void DrawLODRings(int div)
    {
        if (priority == null) return;
        float nr = Mathf.Pow(2, div - 1);
        float ns = chunkResolution * nr;
        float range = ns * innerRadiusPadding * 2f;
        Gizmos.color = div < 2 ? Color.grey : Color.green;
        Gizmos.DrawWireCube(priority.position, Vector3.one * range);
        if (div >= 2) DrawLODRings(div - 1);
    }
}