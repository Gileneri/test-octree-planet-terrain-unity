using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// A single cell in the OctreeGrid.
///
/// Differences from the standalone version:
/// • No Update/LateUpdate — ticked by OctreeGrid.
/// • Tick() collects pending JobCompleters into a shared list owned by
///   OctreeGrid, which sorts them by distance and applies the global budget.
/// • Each JobCompleter now carries a cancel callback so the grid can safely
///   discard jobs that exceed the budget without leaving nodes locked.
/// </summary>
public class Octree : MonoBehaviour
{
    [HideInInspector] public float worldSizeX;
    [HideInInspector] public float worldSizeZ;
    [HideInInspector] public float surfaceBaseHeight;
    [HideInInspector] public float surfaceNoiseAmplitude;
    [HideInInspector] public int divisions;
    [HideInInspector] public int chunkResolution;
    [HideInInspector] public float innerRadiusPadding;
    [HideInInspector] public int maxNodeCreationsPerFrame;
    [HideInInspector] public Transform priority;
    [HideInInspector] public GameObject chunkPrefab;
    [HideInInspector] public Vector3 cellOrigin;

    [HideInInspector] public Node root;

    private bool initialised = false;

    // -----------------------------------------------------------------------
    //  Lifecycle
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

    /// <summary>
    /// Called every Update by OctreeGrid.
    /// Traverses the tree and collects pending jobs into <paramref name="jobList"/>.
    /// Each job is tagged with its squared distance to the player so the grid
    /// can sort and apply the global budget in LateUpdate.
    /// </summary>
    public void Tick(List<OctreeGrid.PrioritizedJob> jobList, Vector3 playerPos)
    {
        if (!initialised) return;
        Traverse(root, 0);
        CollectJobs(root, jobList, playerPos);
    }

    public void Shutdown()
    {
        if (root != null) DestroyTree(root);
        root = null;
        initialised = false;
        Destroy(gameObject);
    }

    // -----------------------------------------------------------------------
    //  Traversal
    // -----------------------------------------------------------------------

    private void Traverse(Node node, int creations)
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
        Vector3 pp = priority.position;
        Vector3 min = center - Vector3.one * halfSize;
        Vector3 max = center + Vector3.one * halfSize;

        return pp.x >= min.x && pp.x <= max.x
            && pp.y >= min.y && pp.y <= max.y
            && pp.z >= min.z && pp.z <= max.z;
    }

    // -----------------------------------------------------------------------
    //  Job collection  (replaces the old Schedule + LateUpdate pattern)
    // -----------------------------------------------------------------------

    private void CollectJobs(Node node, List<OctreeGrid.PrioritizedJob> jobList, Vector3 playerPos)
    {
        if (node.TryCollectJob(out JobCompleter completer))
        {
            float sqrDist = (node.NodePosition() - playerPos).sqrMagnitude;
            jobList.Add(new OctreeGrid.PrioritizedJob
            {
                completer = completer,
                sqrDistToPlayer = sqrDist,
            });
        }

        if (node.children == null) return;
        for (int i = 0; i < node.children.Length; i++)
            CollectJobs(node.children[i], jobList, playerPos);
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