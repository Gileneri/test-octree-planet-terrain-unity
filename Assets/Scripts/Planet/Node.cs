using UnityEngine;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;

public class Node
{
    public Octree octree;
    public Node parent;
    public Node[] children;
    public int divisions;
    public Vector3 offset;
    public GameObject gameObject;

    private MeshFilter meshFilter;
    private MeshRenderer meshRenderer;
    private MeshCollider meshCollider;

    public bool IsScheduled => isScheduled;
    private bool isScheduled = false;
    private bool needsDrawn = true;

    private NativeArray<int3> modKeys;
    private NativeArray<byte> modValues;

    // -----------------------------------------------------------------------

    public Node(Octree octree, Node parent, Vector3 offset, int divisions)
    {
        this.octree = octree;
        this.parent = parent;
        this.offset = offset;
        this.divisions = divisions;

        gameObject = GameObject.Instantiate(octree.chunkPrefab);
        if (parent != null)
            gameObject.transform.parent = parent.gameObject.transform;

        meshFilter = gameObject.GetComponent<MeshFilter>();
        meshRenderer = gameObject.GetComponent<MeshRenderer>();
        meshCollider = gameObject.GetComponent<MeshCollider>();
        if (meshCollider == null)
            meshCollider = gameObject.AddComponent<MeshCollider>();

        var pos = NodePosition();
        gameObject.transform.position = pos;
        gameObject.name = pos.ToString();
    }

    // -----------------------------------------------------------------------

    public void Clear()
    {
        meshFilter.mesh = null;
        meshCollider.sharedMesh = null;
        needsDrawn = true;
    }

    public void MarkDirty()
    {
        meshFilter.mesh = null;
        meshCollider.sharedMesh = null;
        needsDrawn = true;
        isScheduled = false;
    }

    public void Destroy()
    {
        if (modKeys.IsCreated) modKeys.Dispose();
        if (modValues.IsCreated) modValues.Dispose();
        GameObject.Destroy(gameObject);
    }

    // -----------------------------------------------------------------------
    //  TryCollectJob — replaces TrySchedule
    //
    //  Prepares the NodeJob and returns a JobCompleter with three callbacks:
    //    schedule   → called by OctreeGrid to start the Burst job
    //    onComplete → called after JobHandle.CompleteAll to apply the mesh
    //    cancel     → called when the budget drops this job; resets the node
    //                 so it will be re-collected next frame
    // -----------------------------------------------------------------------

    public bool TryCollectJob(out JobCompleter jobCompleter)
    {
        if (!IsLeaf() || !needsDrawn || isScheduled)
        {
            jobCompleter = null;
            return false;
        }

        var meshDataArray = Mesh.AllocateWritableMeshData(1);
        var bounds = new Bounds(Vector3.zero, Vector3.one * NodeScale());
        var nodePos = NodePosition();

        BuildModificationArrays(nodePos, NodeScale(), out modKeys, out modValues);

        var job = new NodeJob
        {
            meshDataArray = meshDataArray,
            bounds = bounds,
            worldSizeX = octree.worldSizeX,
            worldSizeZ = octree.worldSizeZ,
            surfaceBaseHeight = octree.surfaceBaseHeight,
            surfaceNoiseAmplitude = octree.surfaceNoiseAmplitude,
            chunkResolution = octree.chunkResolution,
            nodeScale = NodeScale(),
            worldNodePosition = nodePos,
            modKeys = modKeys,
            modValues = modValues,
        };

        // Mark as in-flight so we don't double-collect this frame
        isScheduled = true;
        needsDrawn = false;

        var capturedFilter = meshFilter;
        var capturedCollider = meshCollider;

        jobCompleter = new JobCompleter(
            // schedule — start the Burst job
            schedule: () => job.Schedule(),

            // onComplete — apply finished mesh (main thread, after CompleteAll)
            onComplete: () =>
            {
                var mesh = new Mesh { name = "node_mesh", bounds = bounds };
                Mesh.ApplyAndDisposeWritableMeshData(meshDataArray, mesh);
                mesh.RecalculateNormals();
                capturedFilter.mesh = mesh;
                capturedCollider.sharedMesh = mesh;
                isScheduled = false;
                if (modKeys.IsCreated) modKeys.Dispose();
                if (modValues.IsCreated) modValues.Dispose();
            },

            // cancel — budget dropped this job; free resources and reset so
            //          the node is re-collected on the next frame
            cancel: () =>
            {
                meshDataArray.Dispose();
                if (modKeys.IsCreated) modKeys.Dispose();
                if (modValues.IsCreated) modValues.Dispose();
                isScheduled = false;
                needsDrawn = true;   // will be re-collected next frame
            }
        );

        return true;
    }

    // -----------------------------------------------------------------------

    private void BuildModificationArrays(Vector3 nodePos, float scale,
        out NativeArray<int3> keys, out NativeArray<byte> values)
    {
        var wm = WorldModifications.Instance;
        if (wm == null)
        {
            keys = new NativeArray<int3>(0, Allocator.Persistent);
            values = new NativeArray<byte>(0, Allocator.Persistent);
            return;
        }

        wm.GetModificationsForNode(nodePos, scale, octree.chunkResolution,
            out Vector3Int[] kArr, out WorldModifications.VoxelState[] vArr);

        keys = new NativeArray<int3>(kArr.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        values = new NativeArray<byte>(vArr.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        for (int i = 0; i < kArr.Length; i++)
        {
            keys[i] = new int3(kArr[i].x, kArr[i].y, kArr[i].z);
            values[i] = (byte)vArr[i];
        }
    }

    // -----------------------------------------------------------------------

    public bool IsLeaf() => children == null;

    public int NodeResolution() => (int)Mathf.Pow(2, divisions - 1);
    public float NodeScale() => octree.chunkResolution * NodeResolution();

    public Vector3 NodePosition()
    {
        if (parent == null) return offset;
        float ns = NodeScale();
        return (offset * ns) - (Vector3.one * (ns * 0.5f)) + parent.NodePosition();
    }
}