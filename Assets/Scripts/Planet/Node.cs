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
    private int meshRevision = 0;

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

        // Only attach a MeshCollider on fine-grained nodes (small voxels).
        // Coarse nodes have voxels too large for PhysX (>500u triangle warning).
        // maxColliderDivisions is computed by OctreeGrid so voxelSize <= 32u.
        if (divisions <= octree.maxColliderDivisions)
        {
            meshCollider = gameObject.GetComponent<MeshCollider>();
            if (meshCollider == null)
                meshCollider = gameObject.AddComponent<MeshCollider>();
        }

        var pos = NodePosition();
        gameObject.transform.position = pos;
        gameObject.name = pos.ToString();
    }

    // -----------------------------------------------------------------------

    public void Clear()
    {
        meshFilter.mesh = null;
        if (meshCollider != null) meshCollider.sharedMesh = null;
        needsDrawn = true;
        // Invalidate any in-flight job result for this node (e.g. when a parent
        // node is cleared due to subdivision) so stale coarse meshes cannot be
        // applied later on top of detailed children.
        meshRevision++;
    }

    public void MarkDirty()
    {
        // Keep the current mesh visible until the refreshed mesh is ready.
        // This avoids one-frame unload/load flicker around interactions.
        needsDrawn = true;
        meshRevision++;
    }

    public void Destroy()
    {
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

        BuildModificationArrays(nodePos, NodeScale(), out NativeArray<int3> jobModKeys, out NativeArray<byte> jobModValues);
        BuildGeoLayersArray(out NativeArray<GeologicalLayerBlob> jobGeoLayers);

        var job = new NodeJob
        {
            meshDataArray = meshDataArray,
            bounds = bounds,
            worldSizeX = octree.worldSizeX,
            worldSizeZ = octree.worldSizeZ,
            surfaceBaseHeight = octree.surfaceBaseHeight,
            surfaceNoiseAmplitude = octree.surfaceNoiseAmplitude,
            // ── WorldConfig-driven fields (added) ──────────────────────────
            noiseSeed = octree.noiseSeed,
            noiseTypeId = octree.noiseTypeId,
            minSubsurfaceHeight = octree.minSubsurfaceHeight,
            cavesEnabled = octree.cavesEnabled,
            caveNoiseFrequency = octree.caveNoiseFrequency,
            caveNoiseThreshold = octree.caveNoiseThreshold,
            caveNoiseAmplitudeY = octree.caveNoiseAmplitudeY,
            caveSurfaceFadeRange = octree.caveSurfaceFadeRange,
            cullEnclosedUndergroundFaces = octree.cullEnclosedUndergroundFaces,
            enclosedUndergroundFaceMargin = octree.enclosedUndergroundFaceMargin,
            useSurfaceShellOnCoarseLod = octree.useSurfaceShellOnCoarseLod,
            surfaceShellMinDivisions = octree.surfaceShellMinDivisions,
            cullUndergroundFacesOnCoarseLod = octree.cullUndergroundFacesOnCoarseLod,
            coarseLodFaceCullMinDivisions = octree.coarseLodFaceCullMinDivisions,
            coarseLodFaceCullMargin = octree.coarseLodFaceCullMargin,
            coarseLodMinSurfaceVoxels = octree.coarseLodMinSurfaceVoxels,
            // ───────────────────────────────────────────────────────────────
            nodeDivisions = divisions,
            chunkResolution = octree.chunkResolution,
            nodeScale = NodeScale(),
            worldNodePosition = nodePos,
            modKeys = jobModKeys,
            modValues = jobModValues,
            geoLayers = jobGeoLayers,
        };
        int scheduledRevision = meshRevision;

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
                if (scheduledRevision != meshRevision)
                {
                    meshDataArray.Dispose();
                    isScheduled = false;
                    needsDrawn = true;
                    if (jobModKeys.IsCreated) jobModKeys.Dispose();
                    if (jobModValues.IsCreated) jobModValues.Dispose();
                    if (jobGeoLayers.IsCreated) jobGeoLayers.Dispose();
                    return;
                }

                if (capturedFilter == null)
                {
                    // Node/object was destroyed before completion.
                    meshDataArray.Dispose();
                    isScheduled = false;
                    needsDrawn = false;
                    if (jobModKeys.IsCreated) jobModKeys.Dispose();
                    if (jobModValues.IsCreated) jobModValues.Dispose();
                    if (jobGeoLayers.IsCreated) jobGeoLayers.Dispose();
                    return;
                }

                var mesh = new Mesh { name = "node_mesh", bounds = bounds };
                Mesh.ApplyAndDisposeWritableMeshData(meshDataArray, mesh);
                // Normals are written per-face inside NodeJob (Tables.Normals[side])
                // and arrive ready in stream 1 — RecalculateNormals() is not needed
                // and would block the main thread re-computing what we already have.
                capturedFilter.mesh = mesh;
                if (capturedCollider != null) capturedCollider.sharedMesh = mesh;
                isScheduled = false;
                if (jobModKeys.IsCreated) jobModKeys.Dispose();
                if (jobModValues.IsCreated) jobModValues.Dispose();
                if (jobGeoLayers.IsCreated) jobGeoLayers.Dispose();
            },

            // cancel — budget dropped this job; free resources and reset so
            //          the node is re-collected on the next frame
            cancel: () =>
            {
                meshDataArray.Dispose();
                if (jobModKeys.IsCreated) jobModKeys.Dispose();
                if (jobModValues.IsCreated) jobModValues.Dispose();
                if (jobGeoLayers.IsCreated) jobGeoLayers.Dispose();
                isScheduled = false;
                // If the node changed while this job was pending/running, keep
                // it dirty so a newer job is collected next frame.
                needsDrawn = true;
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

    private void BuildGeoLayersArray(out NativeArray<GeologicalLayerBlob> layers)
    {
        var blobs = octree.geoLayerBlobs;
        if (blobs == null || blobs.Length == 0)
        {
            layers = new NativeArray<GeologicalLayerBlob>(0, Allocator.Persistent);
            return;
        }

        layers = new NativeArray<GeologicalLayerBlob>(
            blobs.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        for (int i = 0; i < blobs.Length; i++)
            layers[i] = blobs[i];
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