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

    /// <summary>
    /// Reference to the mesh currently applied to this node.
    /// Used by the hierarchical visibility logic (UpdateVisibility) so we can
    /// re-attach / detach the mesh to the collider atomically when the node
    /// becomes (in)visible without re-allocating.
    /// </summary>
    private Mesh appliedMesh;

    public bool IsScheduled => isScheduled;
    private bool isScheduled = false;
    private bool needsDrawn = true;

    /// <summary>
    /// True once a NodeJob has finished and the mesh has been applied to this
    /// node. Stays true even when the mesh is "stale" relative to a new
    /// MarkDirty refresh — we want to keep showing the OLD mesh until the
    /// fresh one is ready, otherwise the player would see one-frame flicker.
    /// Reset to false only by Clear() (full invalidation, e.g. node destroyed).
    /// </summary>
    private bool meshReady = false;

    /// <summary>
    /// Cached "this entire subtree has nothing to do this frame" flag.
    /// True when this node and every descendant are meshReady, not scheduled,
    /// not needing redraw, and (recursively) their children are stable too.
    /// Octree.CollectJobs and Node.UpdateVisibility use this to skip whole
    /// subtrees instantly — without it, every Tick walks the entire tree
    /// (potentially tens of thousands of nodes) just to do nothing.
    /// </summary>
    private bool subtreeStable = false;

    /// <summary>
    /// Records the forceHidden value the last time UpdateVisibilityImpl ran on
    /// this node. We early-out only when the parent's "force me hidden" hint
    /// is the same as last frame; otherwise the ancestor flipped from
    /// rendering to delegating (or vice versa) and we have to re-evaluate
    /// our own renderer state, even if our subtree is otherwise stable.
    /// </summary>
    private bool lastForceHidden = false;

    /// <summary>
    /// Async PhysX cooking — see BakeMeshJob. We schedule the bake right after
    /// applying the mesh so the heavy main-thread cook isn't done at all on
    /// the main thread; we only assign the cooked mesh to the collider once
    /// the bake handle reports completion. Until then the collider has no
    /// physics data attached — that's fine: visibility shows the mesh
    /// instantly, only the collision lags by a few ms.
    /// </summary>
    private bool hasPendingBake = false;
    private JobHandle pendingBakeHandle;
    private Mesh pendingBakeMesh;

    /// <summary>
    /// Set true once this Node has been Destroy()'d. Because the underlying
    /// GameObject goes back to the chunk pool (instead of being destroyed),
    /// any captured MeshFilter/MeshRenderer/MeshCollider references in a
    /// pending job's onComplete lambda are STILL VALID — they would happily
    /// stomp the mesh of whichever new Node has since recycled the GO.
    /// onComplete checks this flag and returns early if the Node is gone.
    /// </summary>
    private bool destroyed = false;

    private int meshRevision = 0;

    // -----------------------------------------------------------------------

    public Node(Octree octree, Node parent, Vector3 offset, int divisions)
    {
        this.octree = octree;
        this.parent = parent;
        this.offset = offset;
        this.divisions = divisions;

        // Pull a GameObject out of the chunk pool — Instantiate/Destroy at
        // the rate this terrain churns chunks (hundreds per second while
        // flying fast) is by far the biggest non-meshing CPU cost. The
        // pool returns an already-Active instance; we just re-parent it.
        gameObject = ChunkObjectPool.Acquire(octree.chunkPrefab);
        // Mirror the original parenting policy: only attach to the
        // parent node when one exists; the Octree root node stays at
        // scene-root level the way it always has.
        if (parent != null)
            gameObject.transform.SetParent(parent.gameObject.transform, worldPositionStays: false);
        else
            gameObject.transform.SetParent(null, worldPositionStays: false);

        meshFilter = gameObject.GetComponent<MeshFilter>();
        meshRenderer = gameObject.GetComponent<MeshRenderer>();

        // ── Renderer perf hardening ───────────────────────────────────────
        // Procedural voxel terrain has thousands of chunks. Each one paying
        // for shadow casting, light/reflection probe sampling, motion vectors
        // and dynamic-occludee bookkeeping is a *huge* per-frame cost on the
        // GPU and the main thread. We force the cheapest settings here so
        // they cannot be re-introduced by accidentally re-saving the prefab.
        if (meshRenderer != null)
        {
            meshRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            meshRenderer.receiveShadows = true; // keep — gives the terrain shading depth
            meshRenderer.lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off;
            meshRenderer.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;
            meshRenderer.allowOcclusionWhenDynamic = false;
            meshRenderer.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;
        }

        // Only attach a MeshCollider on fine-grained nodes (small voxels).
        // Coarse nodes have voxels too large for PhysX (>500u triangle warning).
        // maxColliderDivisions is computed by OctreeGrid so voxelSize <= 32u.
        if (divisions <= octree.maxColliderDivisions)
        {
            meshCollider = gameObject.GetComponent<MeshCollider>();
            if (meshCollider == null)
                meshCollider = gameObject.AddComponent<MeshCollider>();

            // Skip every cooking step PhysX runs by default. The terrain is
            // procedural and rebuilt very frequently — we want fast bakes,
            // not maximally optimised query speed. None makes cooking 5-10×
            // faster (a 5–30 ms main-thread spike per chunk → sub-millisecond).
            meshCollider.cookingOptions = MeshColliderCookingOptions.None;
        }

        var pos = NodePosition();
        gameObject.transform.position = pos;
        gameObject.name = pos.ToString();

        // A brand-new node always has work to do (mesh isn't built yet).
        // Walk the parent chain to mark them as needing collection too.
        if (parent != null)
            parent.InvalidateStability();
    }

    // -----------------------------------------------------------------------

    public void Clear()
    {
        // Wait for any pending PhysX cook before tearing the mesh down —
        // BakeMesh is reading the Mesh asset by instance id on a worker
        // thread and would crash if we destroyed it from under it.
        if (hasPendingBake)
        {
            pendingBakeHandle.Complete();
            hasPendingBake = false;
            pendingBakeMesh = null;
        }

        if (meshCollider != null) meshCollider.sharedMesh = null;
        meshFilter.mesh = null;

        // Explicitly destroy the mesh asset to avoid leaking a Mesh per
        // node that lives forever until the next full GC pass.
        if (appliedMesh != null)
        {
            GameObject.Destroy(appliedMesh);
            appliedMesh = null;
        }

        needsDrawn = true;
        meshReady = false;
        // Invalidate any in-flight job result for this node so stale meshes
        // cannot be applied later on top of newer state.
        meshRevision++;
        InvalidateStability();
    }

    public void MarkDirty()
    {
        // Keep the current mesh visible until the refreshed mesh is ready.
        // This avoids one-frame unload/load flicker around interactions.
        // meshReady stays true on purpose — UpdateVisibility uses it to keep
        // showing the old mesh while the new job is in flight.
        needsDrawn = true;
        meshRevision++;
        InvalidateStability();
    }

    /// <summary>
    /// Marks this node and all its ancestors as "potentially needs work" so
    /// CollectJobs and UpdateVisibility re-visit them. Cheap walk to the root.
    /// Call any time a node's state transitions in a way that could affect
    /// scheduling or visibility (creation, MarkDirty, Clear, child change,
    /// mesh applied, etc.).
    /// </summary>
    public void InvalidateStability()
    {
        var n = this;
        while (n != null && n.subtreeStable)
        {
            n.subtreeStable = false;
            n = n.parent;
        }
    }

    /// <summary>True if Octree.CollectJobs / Node.UpdateVisibility can skip this entire subtree.</summary>
    public bool SubtreeStable => subtreeStable;

    public void Destroy()
    {
        // Mark this Node as gone before doing any teardown so any in-flight
        // job's onComplete lambda will short-circuit cleanly when it lands
        // (necessary because the GameObject lives on inside the pool —
        // otherwise the captured component refs would still look valid).
        destroyed = true;
        // Bump the mesh revision so any Octree-side equality checks that
        // hold a stale revision number recognize this Node as superseded.
        meshRevision++;

        // If a collider bake is still in flight we MUST wait for it before
        // destroying the mesh — Physics.BakeMesh references the mesh by
        // instanceID and would crash if Unity GC'd the mesh underneath it.
        if (hasPendingBake)
        {
            pendingBakeHandle.Complete();
            hasPendingBake = false;
            pendingBakeMesh = null;
        }

        // Detach the mesh from the (potentially still-attached) MeshCollider
        // BEFORE destroying it — leaving a destroyed Mesh referenced from a
        // pooled MeshCollider would create a dangling reference that bites
        // the next Node that recycles this GameObject.
        if (meshCollider != null)
            meshCollider.sharedMesh = null;
        if (meshFilter != null)
            meshFilter.mesh = null;

        // Mesh assets aren't auto-destroyed when the host GameObject dies —
        // they linger until a full GC, and after a few minutes of travel
        // that pile becomes a multi-second hitch. Free it explicitly.
        if (appliedMesh != null)
        {
            GameObject.Destroy(appliedMesh);
            appliedMesh = null;
        }

        // Reset renderer to its default enabled state so the next Node
        // that recycles this GameObject doesn't inherit a stale
        // disabled renderer from the previous owner's UpdateVisibility.
        if (meshRenderer != null)
            meshRenderer.enabled = true;

        // Hand the GameObject back to the pool instead of destroying it.
        // This skips Object.Destroy + GC + the next Object.Instantiate,
        // which together typically cost more than meshing a chunk does.
        ChunkObjectPool.Release(gameObject);
        gameObject = null;
    }

    /// <summary>
    /// Polls the pending async collider bake. If finished, completes the
    /// handle and assigns the cooked mesh to the MeshCollider. Cheap when
    /// no bake is pending. Called once per Tick from UpdateVisibilityImpl.
    /// </summary>
    private void ResolvePendingBake()
    {
        if (!hasPendingBake) return;
        if (!pendingBakeHandle.IsCompleted) return;

        pendingBakeHandle.Complete();
        // Only push the cooked mesh into the collider if the node still
        // has the same mesh applied (i.e., a newer mesh hasn't superseded it).
        if (meshCollider != null && pendingBakeMesh != null
            && pendingBakeMesh == appliedMesh
            && pendingBakeMesh.vertexCount > 0
            && meshRenderer != null && meshRenderer.enabled)
        {
            meshCollider.sharedMesh = pendingBakeMesh;
        }

        hasPendingBake = false;
        pendingBakeMesh = null;
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
        // We deliberately do NOT require IsLeaf here. Intermediate (non-leaf)
        // nodes also schedule their mesh so they can act as a coarse fallback
        // when their subtree isn't fully ready yet (e.g. during deep init,
        // during subdivision while children are loading, or during collapse).
        //
        // Without this, a player rotating the camera or moving away would see
        // holes wherever the deepest leaves haven't finished meshing.
        if (!needsDrawn || isScheduled)
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
            // Only chunks fine enough to host a MeshCollider need the
            // greedy-merge cap. Render-only coarse chunks merge freely.
            capGreedyForCollider = meshCollider != null,
        };
        int scheduledRevision = meshRevision;

        // Mark as in-flight so we don't double-collect this frame
        isScheduled = true;
        needsDrawn = false;

        var capturedFilter = meshFilter;
        var capturedCollider = meshCollider;
        var capturedRenderer = meshRenderer;

        jobCompleter = new JobCompleter(
            // schedule — start the Burst job
            schedule: () => job.Schedule(),

            // onComplete — apply finished mesh (main thread, after CompleteAll)
            onComplete: () =>
            {
                // Two ways the node can be gone by the time the job lands:
                //  1. The owning GameObject was destroyed outright
                //     (capturedFilter == null — Unity null check).
                //  2. The Node was Destroy()'d but its GameObject was
                //     returned to the chunk pool — the MeshFilter ref is
                //     still alive (pointed at a now-recycled GO!), so we
                //     must consult the explicit `destroyed` flag instead.
                if (destroyed || capturedFilter == null)
                {
                    meshDataArray.Dispose();
                    if (jobModKeys.IsCreated) jobModKeys.Dispose();
                    if (jobModValues.IsCreated) jobModValues.Dispose();
                    if (jobGeoLayers.IsCreated) jobGeoLayers.Dispose();
                    return;
                }

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

                // Wait for any older bake to finish — its mesh is about to be
                // replaced/destroyed and PhysX must not still be reading it.
                if (hasPendingBake)
                {
                    pendingBakeHandle.Complete();
                    hasPendingBake = false;
                    pendingBakeMesh = null;
                }

                // Destroy the previous mesh explicitly. MeshFilter.mesh ownership
                // does NOT auto-destroy old meshes in Play mode; without this
                // every re-mesh leaks a Mesh until the next GC, which causes
                // periodic stutters once a few thousand have piled up.
                var oldMesh = appliedMesh;

                var mesh = new Mesh { name = "node_mesh", bounds = bounds };
                Mesh.ApplyAndDisposeWritableMeshData(meshDataArray, mesh);
                // Normals are written per-face inside NodeJob (Tables.Normals[side])
                // and arrive ready in stream 1 — RecalculateNormals() is not needed
                // and would block the main thread re-computing what we already have.
                capturedFilter.mesh = mesh;
                appliedMesh = mesh;
                meshReady = true;

                if (oldMesh != null && oldMesh != mesh)
                {
                    // Detach from collider first — destroying a Mesh that is
                    // assigned to a MeshCollider's sharedMesh would leave a
                    // dangling reference inside PhysX.
                    if (capturedCollider != null && capturedCollider.sharedMesh == oldMesh)
                        capturedCollider.sharedMesh = null;
                    GameObject.Destroy(oldMesh);
                }

                // Async collider cooking. Assigning sharedMesh on the main
                // thread would trigger PhysX to cook the mesh inline (5–30 ms
                // per chunk). Instead we kick a BakeMeshJob to a worker
                // thread; once it finishes, ResolvePendingBake() (called from
                // UpdateVisibilityImpl) does a near-free sharedMesh = mesh.
                if (capturedCollider != null && mesh.vertexCount > 0)
                {
                    pendingBakeMesh = mesh;
                    pendingBakeHandle = new BakeMeshJob
                    {
                        meshId = mesh.GetInstanceID(),
                        convex = false,
                    }.Schedule();
                    hasPendingBake = true;
                }
                else if (capturedCollider != null)
                {
                    // Empty mesh — clear any stale collider data.
                    if (capturedCollider.sharedMesh != null)
                        capturedCollider.sharedMesh = null;
                }
                isScheduled = false;
                // Mesh just became ready — visibility may need re-evaluating
                // up the chain (CanCover() of ancestors might now flip true).
                InvalidateStability();
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
                if (destroyed) return; // Node is gone — nothing to mark dirty.
                isScheduled = false;
                // If the node changed while this job was pending/running, keep
                // it dirty so a newer job is collected next frame.
                needsDrawn = true;
                InvalidateStability();
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

        // WorldModifications now writes directly into our NativeArrays — no
        // intermediate List<T> / managed-array allocation per call.
        wm.GetModificationsForNode(nodePos, scale, octree.chunkResolution,
            Allocator.Persistent, out keys, out values);
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

    // -----------------------------------------------------------------------
    //  Hierarchical visibility
    //
    //  These methods replace the old "clear-parent-on-subdivide" approach.
    //  The tree always renders at the deepest level where every descendant
    //  has a mesh ready; otherwise it falls back to the coarsest ancestor
    //  that does. This gives perfectly atomic LOD swaps with no holes —
    //  the coarse parent's mesh stays visible until ALL of its children's
    //  meshes are ready, then they swap in one frame.
    // -----------------------------------------------------------------------

    /// <summary>
    /// Returns true if this subtree can be rendered with full coverage.
    /// Either this node has its own mesh ready, or every child can cover
    /// its own volume recursively.
    /// </summary>
    public bool CanCover()
    {
        if (meshReady) return true;
        if (children == null) return false;
        for (int i = 0; i < children.Length; i++)
            if (!children[i].CanCover()) return false;
        return true;
    }

    /// <summary>
    /// Recursively chooses which nodes in the subtree are rendered:
    ///   - prefer rendering at the deepest level where ALL children can cover
    ///   - otherwise render at this node if it has a mesh
    ///   - otherwise recurse and show whatever partial coverage exists
    ///
    /// Called once per Octree.Tick from the root. Top-down, O(active nodes)
    /// in the worst case — but stable subtrees early-out via subtreeStable
    /// so most frames touch only a tiny slice of the tree near the player.
    /// No allocations.
    ///
    /// Also computes and caches the subtreeStable flag as a side-effect, so
    /// CollectJobs and the next UpdateVisibility can skip stable subtrees.
    /// </summary>
    public void UpdateVisibility()
    {
        UpdateVisibilityImpl(forceHidden: false);
    }

    private void UpdateVisibilityImpl(bool forceHidden)
    {
        // Pick up any async PhysX collider bake that just finished.
        // Cheap when none is pending, otherwise assigns sharedMesh.
        ResolvePendingBake();

        // Stable subtrees have nothing to do — no pending jobs, no visibility
        // changes since last frame. Skip immediately. We also require the
        // parent's forceHidden hint to match what we saw last time; if the
        // ancestor flipped between rendering and delegating, our renderer
        // state needs to flip too even if all our jobs are done.
        if (subtreeStable && lastForceHidden == forceHidden) return;

        bool wantVisible;
        bool forceChildrenHidden;
        bool recurseChildren;

        if (forceHidden)
        {
            // An ancestor is rendering and we must stay hidden, but we still
            // recurse so child stability flags stay accurate.
            wantVisible = false;
            forceChildrenHidden = true;
            recurseChildren = children != null;
        }
        else if (children == null)
        {
            wantVisible = meshReady;
            forceChildrenHidden = false;
            recurseChildren = false;
        }
        else
        {
            bool allChildrenCanCover = true;
            for (int i = 0; i < children.Length; i++)
            {
                if (!children[i].CanCover())
                {
                    allChildrenCanCover = false;
                    break;
                }
            }

            if (allChildrenCanCover)
            {
                // Deepest-coverage path: hide this node, recurse into children.
                wantVisible = false;
                forceChildrenHidden = false;
                recurseChildren = true;
            }
            else if (meshReady)
            {
                // Children not fully ready — keep this node's coarse mesh up so
                // there is no visible hole. Force the entire subtree hidden.
                wantVisible = true;
                forceChildrenHidden = true;
                recurseChildren = true;
            }
            else
            {
                // No mesh on this node either. Render whatever children we have
                // (partial coverage). This only happens during the very first
                // frames after spawning before any mesh has been applied.
                wantVisible = false;
                forceChildrenHidden = false;
                recurseChildren = true;
            }
        }

        SetVisible(wantVisible);

        bool allChildrenStable = true;
        if (recurseChildren)
        {
            for (int i = 0; i < children.Length; i++)
            {
                children[i].UpdateVisibilityImpl(forceChildrenHidden);
                if (!children[i].subtreeStable) allChildrenStable = false;
            }
        }

        // Cache: this subtree is stable when this node has no pending work
        // AND every child's subtree is also stable. Cleared by InvalidateStability
        // whenever something changes (job apply, MarkDirty, new node, etc.).
        // A pending collider bake also counts as "work" — we need to revisit
        // next frame to resolve it.
        bool selfNoWork = meshReady && !needsDrawn && !isScheduled && !hasPendingBake;
        subtreeStable = selfNoWork && (children == null || allChildrenStable);
        lastForceHidden = forceHidden;
    }

    /// <summary>
    /// Atomically sets the renderer enabled/disabled and attaches/detaches
    /// the mesh to the collider. Keeping these two in sync prevents the
    /// player from colliding with hidden coarse geometry while looking at
    /// the fine version (and vice-versa) during LOD transitions.
    ///
    /// If a collider bake is still in flight we leave whatever sharedMesh
    /// the collider has alone — assigning the new (uncooked) mesh here
    /// would trigger a synchronous PhysX cook on the main thread, which is
    /// exactly the cost the async bake exists to avoid. ResolvePendingBake
    /// will install the cooked mesh next frame.
    /// </summary>
    private void SetVisible(bool visible)
    {
        if (meshRenderer != null && meshRenderer.enabled != visible)
            meshRenderer.enabled = visible;

        if (meshCollider == null) return;

        if (!visible)
        {
            if (meshCollider.sharedMesh != null)
                meshCollider.sharedMesh = null;
            return;
        }

        if (hasPendingBake) return; // wait for ResolvePendingBake

        // Treat empty meshes as null for the collider — assigning a mesh
        // with zero vertices to a MeshCollider triggers a noisy Unity
        // warning ("...but the mesh doesn't have any vertices"), and
        // there is nothing to collide with anyway. Empty meshes happen
        // for chunks that are entirely air or entirely solid.
        Mesh desired = (appliedMesh != null && appliedMesh.vertexCount > 0)
            ? appliedMesh
            : null;
        if (meshCollider.sharedMesh != desired)
            meshCollider.sharedMesh = desired;
    }
}