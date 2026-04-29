using Unity.Burst;
using Unity.Jobs;
using UnityEngine;

/// <summary>
/// Pre-bakes the PhysX cooked-mesh data for a Mesh on a worker thread.
/// Once this job completes, assigning the mesh to a MeshCollider's
/// sharedMesh on the main thread is essentially free — the heavy cooking
/// step has already happened off-thread.
///
/// Pair with MeshColliderCookingOptions.None on the collider to keep the
/// cook itself as cheap as possible. The combination eliminates the
/// 5–30 ms main-thread spike whenever a node finishes meshing and would
/// otherwise have its collider cooked synchronously.
/// </summary>
[BurstCompile]
public struct BakeMeshJob : IJob
{
    public int meshId;
    public bool convex;

    public void Execute()
    {
        Physics.BakeMesh(meshId, convex);
    }
}
