using UnityEngine;

/// <summary>
/// Keeps the tracked transform inside one fundamental domain [0, worldSizeX) × [0, worldSizeZ)
/// by subtracting/adding whole periods — same convention as <see cref="WorldModifications.WrapKey"/>.
/// Reduces float drift when exploring far in world space and avoids OctreeGrid velocity spikes when paired with
/// <see cref="OctreeGrid.ApplyToroidalTrackingCorrection"/>.
///
/// Place on the player (or whatever object is assigned as OctreeGrid.priority).
/// Execution order runs before <see cref="OctreeGrid"/> so wrapping happens before streaming distance logic.
///
/// Physics: after teleporting the transform, <see cref="Physics.SyncTransforms"/> runs here so the controller
/// sees an up-to-date pose. <see cref="OctreeGrid"/> performs the same again at the end of its Update whenever
/// terrain chunks rebase at the seam — without that second sync, MeshCollider roots could lag one step behind
/// the wrapped player pose inside PhysX.
///
/// Behaviour at seam:
/// - One real crossing = one discrete teleport event.
/// - Building/editing near the seam is safe because voxel keys are canonical (WrapKey/PosMod).
/// - If the player intentionally oscillates across the seam (left/right spam), multiple wraps can happen
///   in rapid sequence. Use minWrapInterval to damp that chatter.
/// </summary>
[DefaultExecutionOrder(-50)]
public class ToroidalBoundaryWrap : MonoBehaviour
{
    [Tooltip("World sizes — read from OctreeGrid each frame.")]
    public OctreeGrid octreeGrid;

    [Tooltip("If null, uses this transform.")]
    public Transform target;

    [Header("Seam stability")]
    [Tooltip("Minimum time between wrap teleports. Prevents wrap chatter when oscillating exactly on the seam.")]
    [Min(0f)] public float minWrapInterval = 0.08f;

    [Header("Optional debug")]
    [Tooltip("Logs seam wrap events (rate-limited) to confirm burst behaviour while testing.")]
    public bool logWrapEvents = false;
    [Tooltip("Minimum interval between seam wrap logs.")]
    [Min(0f)] public float wrapLogInterval = 0.5f;

    private float _lastWrapTime = float.NegativeInfinity;
    private float _lastWrapLogTime = float.NegativeInfinity;

    static float PosMod(float x, float m)
    {
        float r = x % m;
        return r < 0f ? r + m : r;
    }

    void Awake()
    {
        if (target == null)
            target = transform;
    }

    void Update()
    {
        if (octreeGrid == null || target == null)
            return;

        float sx = octreeGrid.worldSizeX;
        float sz = octreeGrid.worldSizeZ;
        if (sx <= 0f || sz <= 0f)
            return;

        Vector3 p = target.position;
        float nx = PosMod(p.x, sx);
        float nz = PosMod(p.z, sz);

        const float eps = 1e-3f;
        if (Mathf.Abs(nx - p.x) <= eps && Mathf.Abs(nz - p.z) <= eps)
            return;

        if (minWrapInterval > 0f && Time.time - _lastWrapTime < minWrapInterval)
            return;

        Vector3 newPos = new Vector3(nx, p.y, nz);
        Vector3 delta = newPos - p;

        octreeGrid.ApplyToroidalTrackingCorrection(delta);

        var cc = target.GetComponent<CharacterController>();
        if (cc != null)
        {
            cc.enabled = false;
            target.position = newPos;
            cc.enabled = true;
        }
        else
            target.position = newPos;

        _lastWrapTime = Time.time;

        if (logWrapEvents && Time.time - _lastWrapLogTime >= wrapLogInterval)
        {
            _lastWrapLogTime = Time.time;
            UnityEngine.Debug.Log(
                $"[ToroidalBoundaryWrap] wrapped target={target.name} old=({p.x:F2},{p.z:F2}) new=({newPos.x:F2},{newPos.z:F2}) delta=({delta.x:F2},{delta.z:F2})");
        }

        Physics.SyncTransforms();
    }
}
