using UnityEngine;

/// <summary>
/// Toroidal wrap for the tracked transform. Two modes:
/// <list type="bullet">
/// <item><b>Deferred</b> — world XZ may grow across many periods; when max(|x|,|z|) exceeds a threshold,
/// one teleport maps into the fundamental domain [0, worldSizeX) × [0, worldSizeZ) (same as PosMod).
/// Between teleports there is no wrap at seams, so you must traverse the threshold distance again in any
/// direction before the next recenter.</item>
/// <item><b>Each boundary cross</b> — legacy: PosMod whenever the pose leaves one period (continuous wrap).</item>
/// </list>
/// Same convention as <see cref="WorldModifications.WrapKey"/>. Pairs with <see cref="OctreeGrid.ApplyToroidalTrackingCorrection"/>.
///
/// Place on the player (or whatever object is assigned as OctreeGrid.priority).
/// Execution order runs before <see cref="OctreeGrid"/> so wrapping happens before streaming distance logic.
///
/// Physics: after teleporting the transform, <see cref="Physics.SyncTransforms"/> runs here so the controller
/// sees an up-to-date pose. <see cref="OctreeGrid"/> performs the same again at the end of its Update whenever
/// terrain chunks rebase at the seam — without that second sync, MeshCollider roots could lag one step behind
/// the wrapped player pose inside PhysX.
/// </summary>
[DefaultExecutionOrder(-50)]
public class ToroidalBoundaryWrap : MonoBehaviour
{
    public enum ToroidalWrapMode
    {
        /// <summary>Recenter only when max(|x|,|z|) exceeds the configured threshold.</summary>
        DeferredMaxCoordinateMagnitude,
        /// <summary>Original behaviour: wrap every time the pose leaves [0, worldSize) along an axis.</summary>
        EachBoundaryCross,
    }

    [Tooltip("World sizes — read from OctreeGrid each frame.")]
    public OctreeGrid octreeGrid;

    [Tooltip("If null, uses this transform.")]
    public Transform target;

    [Header("Wrap mode")]
    [Tooltip("Deferred = accumulate many laps, then one PosMod teleport. EachBoundaryCross = wrap at every seam.")]
    public ToroidalWrapMode wrapMode = ToroidalWrapMode.DeferredMaxCoordinateMagnitude;

    [Header("Deferred recenter")]
    [Tooltip("If true, threshold = recenterPeriodCount × max(worldSizeX, worldSizeZ). If false, use recenterThresholdWorld.")]
    public bool deferredThresholdFromWorldPeriods = true;

    [Tooltip("Used when deferredThresholdFromWorldPeriods is true. Lower values recentre sooner (100 ≈ 5M units with 50k world — often too late for camera stability).")]
    [Min(0.01f)] public float recenterPeriodCount = 25f;

    [Tooltip("Used when deferredThresholdFromWorldPeriods is false. World units; max(|x|,|z|) must reach this before recenter.")]
    [Min(1f)] public float recenterThresholdWorld = 1_500_000f;

    [Tooltip("Hard cap (world units): deferred recentre never waits beyond this max(|x|,|z|). Reduces jitter from huge coordinates even if period count is high.")]
    [Min(1f)] public float maxAbsCoordinateBeforeRecenter = 1_500_000f;

    [Header("Seam stability")]
    [Tooltip("Minimum time between wrap teleports. Prevents wrap chatter when oscillating exactly on the seam (mainly legacy mode).")]
    [Min(0f)] public float minWrapInterval = 0.08f;

    [Header("Optional debug")]
    [Tooltip("Logs wrap events (rate-limited) while testing.")]
    public bool logWrapEvents = false;
    [Tooltip("Minimum interval between wrap logs.")]
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

    void OnValidate()
    {
        if (recenterThresholdWorld < 1f) recenterThresholdWorld = 1f;
        if (recenterPeriodCount < 0.01f) recenterPeriodCount = 0.01f;
        if (maxAbsCoordinateBeforeRecenter < 1f) maxAbsCoordinateBeforeRecenter = 1f;
    }

    float ComputeDeferredThreshold(float sx, float sz)
    {
        float w = Mathf.Max(sx, sz);
        if (deferredThresholdFromWorldPeriods)
            return Mathf.Max(recenterPeriodCount * w, w);
        return recenterThresholdWorld;
    }

    float EffectiveDeferredThreshold(float sx, float sz)
    {
        return Mathf.Min(ComputeDeferredThreshold(sx, sz), maxAbsCoordinateBeforeRecenter);
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

        if (wrapMode == ToroidalWrapMode.EachBoundaryCross)
        {
            TryWrapEachBoundaryCross(p, sx, sz);
            return;
        }

        float threshold = EffectiveDeferredThreshold(sx, sz);
        float ax = Mathf.Abs(p.x);
        float az = Mathf.Abs(p.z);
        if (Mathf.Max(ax, az) < threshold)
            return;

        float nx = PosMod(p.x, sx);
        float nz = PosMod(p.z, sz);
        TryApplyWrapTeleport(p, new Vector3(nx, p.y, nz));
    }

    void TryWrapEachBoundaryCross(Vector3 p, float sx, float sz)
    {
        float nx = PosMod(p.x, sx);
        float nz = PosMod(p.z, sz);

        const float eps = 1e-3f;
        if (Mathf.Abs(nx - p.x) <= eps && Mathf.Abs(nz - p.z) <= eps)
            return;

        TryApplyWrapTeleport(p, new Vector3(nx, p.y, nz));
    }

    void TryApplyWrapTeleport(Vector3 oldPos, Vector3 newPos)
    {
        if (minWrapInterval > 0f && Time.time - _lastWrapTime < minWrapInterval)
            return;

        Vector3 delta = newPos - oldPos;

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
                $"[ToroidalBoundaryWrap] mode={wrapMode} target={target.name} old=({oldPos.x:F2},{oldPos.z:F2}) new=({newPos.x:F2},{newPos.z:F2}) delta=({delta.x:F2},{delta.z:F2})");
        }

        Physics.SyncTransforms();
    }
}
