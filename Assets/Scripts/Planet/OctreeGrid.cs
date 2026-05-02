using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;

/// <summary>
/// Manages a grid of Octrees around the player with seamless toroidal wrap via the FUNDAMENTAL DOMAIN
/// streaming scheme (only one Octree instance per canonical chunk, repositioned across seams instead
/// of duplicated).
///
/// STREAMING — fundamental domain + per-chunk origin shift
/// ───────────────────────────────────────────────────────
/// activeCells is keyed by canonical (i % N, j % N). Each Octree carries its currentEuclideanCell —
/// the *physical* representative whose <c>cellOrigin = currentEuclideanCell * cellSize</c> sits closest
/// to the player. When the player crosses a toroidal seam, <see cref="RebaseDriftedCells"/> picks the
/// nearest representative for every chunk and calls <see cref="Octree.RebaseCellOrigin"/>, which walks
/// the tree and reapplies <c>transform.position</c> on each Node (no rebuild, no remesh).
///
/// Density in NodeJob is already periodic via PosMod(worldSize). Because every shift is an exact
/// multiple of worldSize, <c>PosMod(oldPos) == PosMod(newPos)</c> — in-flight jobs stay correct.
///
/// IMPORTANT: pick worldSize so that <c>renderRadius * 2 * cellSize &lt; worldSize / 2</c>. Otherwise
/// the render disk overlaps itself across the seam and the player will see the same chunks twice.
///
/// LOD DISTANCE — lodDistanceCurve
/// ─────────────────────────────────
/// Controls how far away each octree level stays subdivided (detailed).
///
///   X axis : normalised level — 0.0 = finest (smallest nodes, closest detail)
///                                1.0 = coarsest (largest nodes, furthest detail)
///   Y axis : subdivision radius in MULTIPLES of that level's node size
///
/// Examples:
///   Flat at Y = 2.0  →  every level subdivides within 2× its own size (default feel)
///   Rising curve     →  coarse levels stay detailed much further away
///
/// Suggested starting point for "see more detail on the horizon":
///   (0.0, 1.5)   finest nodes  — only subdivide right next to the player
///   (0.5, 3.0)   mid nodes     — stay detailed a bit further
///   (1.0, 8.0)   coarsest      — keep subdivided far away
///
/// The radius is compared against the distance from the player to the node's
/// AABB (not its center), so a node that the player is standing inside always
/// subdivides regardless of its size.
///
/// PERFORMANCE NOTE
/// ─────────────────
/// Higher Y values = more nodes subdivided at once = more draw calls and more
/// mesh jobs per frame. Start conservative and increase gradually while watching
/// the "meshing X ms" log and your frame rate.
/// </summary>
public class OctreeGrid : MonoBehaviour
{
    [Header("World")]
    [Tooltip("Toroidal periods in world units. Prefer multiples of cellSize (see console at Start) for chunk-aligned seams. Optional: add ToroidalBoundaryWrap on the priority object.")]
    public float worldSizeX = 50000f;
    public float worldSizeZ = 50000f;

    [Header("Terrain")]
    public float surfaceBaseHeight = 0f;
    [Tooltip("Max terrain height. Scale up for larger worlds (300+ for 50 km).")]
    public float surfaceNoiseAmplitude = 300f;

    // ── WorldConfig-driven fields (added) ────────────────────────────────
    [Header("Noise")]
    [Tooltip("FastNoiseLite seed. Populated by WorldConfigLoader from the world JSON.")]
    public int noiseSeed = 2376;

    [Tooltip("FastNoiseLite noise type as an integer id.\n" +
             "0=OpenSimplex2  1=OpenSimplex2S  2=Perlin\n" +
             "Populated by WorldConfigLoader — do not edit by hand.")]
    public int noiseTypeId = 0;

    [Tooltip("Hard floor Y. Voxels below this are always solid. " +
             "Populated by WorldConfigLoader from the world JSON.")]
    public float minSubsurfaceHeight = -500f;

    [Header("Caves")]
    [Tooltip("Enable procedural cave generation.")]
    public bool cavesEnabled = false;
    public float caveNoiseFrequency = 0.008f;
    [Range(-1f, 1f)]
    public float caveNoiseThreshold = 0.55f;
    public float caveNoiseAmplitudeY = 0.4f;
    public float caveSurfaceFadeRange = 20f;

    [Tooltip("Max depth (units below surface) where caves carve. Below this, the per-voxel cave noise is skipped. " +
             "0 = always sample (deepest caves possible). Default 256u suits typical voxel terrain.")]
    public float caveMaxDepth = 256f;

    [Tooltip("Octree division level above which caves are skipped (LODs that high don't bother sampling cave noise). " +
             "Cave noise is the dominant per-voxel cost; capping it here is the biggest single perf win for cave worlds. " +
             "Set to 1–3 for best perf; higher values render caves at greater distances.")]
    [Range(1, 12)]
    public int caveMaxDivisions = 3;
    // ─────────────────────────────────────────────────────────────────────

    [Header("Octree per cell")]
    [Tooltip("Octree depth. 7 with chunkResolution=8 gives ~1024 unit cells.")]
    public int divisions = 7;
    [Tooltip("Voxels per chunk side.")]
    public int chunkResolution = 8;

    [Header("LOD Distance")]
    [Tooltip(
        "How far each LOD level stays subdivided (detailed).\n\n" +
        "X = normalised level: 0 = finest (small nodes near player), 1 = coarsest (large nodes far away)\n" +
        "Y = subdivision radius as a multiple of that level's node size\n\n" +
        "To see more detail on the horizon, raise the Y value at X=1.\n" +
        "Example: (0, 1.5) (0.5, 3) (1, 8)\n\n" +
        "Higher Y = more draw calls and mesh jobs. Increase gradually.")]
    public AnimationCurve lodDistanceCurve = AnimationCurve.Linear(0f, 2f, 1f, 2f);

    [Header("Geological Layers")]
    [Tooltip("Populated by WorldConfigLoader from WorldConfig.geologicalLayers. Do not edit by hand.")]
    public GeologicalLayerBlob[] geoLayerBlobs = new GeologicalLayerBlob[0];

    [Header("LOD Vertical Bias")]
    /* [Tooltip(
         "How much more expensive vertical distance is for LOD subdivision.

 " +
         "1.0 = isotropic (subdivides equally up/down and sideways)
 " +
         "3.0 = recommended — nodes below/above the player collapse 3× faster

 " +
         "Higher values reduce underground mesh jobs when on the surface.
 " +
         "Too high (>5) can cause pop-in when the player descends quickly.")] */
    public float verticalLodBias = 3f;

    [Header("Render Distance & Performance")]
    [Tooltip("Cells loaded in each direction. 6 = 13×13 grid.")]
    public int renderRadius = 6;

    [Tooltip("Max chunk mesh jobs completed per frame across ALL cells.")]
    public int maxJobsPerFrame = 120;

    [Tooltip("Target main-thread budget for mesh completion (ms).")]
    public float targetMeshingBudgetMs = 4.0f;

    [Tooltip("Lower bound of jobs completed per frame.")]
    public int minJobsPerFrame = 2;

    [Tooltip("Max jobs scheduled into Burst per frame.")]
    public int maxSchedulesPerFrame = 24;

    [Tooltip("Max jobs kept in-flight at once.")]
    public int maxInFlightJobs = 96;

    [Tooltip("Main-thread budget (ms) to complete/apply finished jobs each frame.")]
    public float completeBudgetMs = 2.5f;

    [Tooltip("Hard cap of completed jobs per frame.")]
    public int maxCompletionsPerFrame = 24;

    [Tooltip("Guaranteed cap reserved for interaction jobs per frame.")]
    public int maxInteractiveCompletionsPerFrame = 8;

    [Tooltip("New octree cells activated per frame when player moves.")]
    public int newCellsPerFrame = 3;

    [Tooltip("Max octree node subdivisions per cell per frame.")]
    public int maxNodeCreationsPerFrame = 50;

    [Header("Underground Culling")]
    [Tooltip("Enable hard node-level culling below a fixed underground altitude.")]
    public bool enableUndergroundNodeCull = false;

    [Tooltip("Safety margin below the planet's lowest possible surface before underground chunks are culled.")]
    public float undergroundSolidCullMargin = 10f;

    [Tooltip("Hide underground faces that border enclosed air pockets.")]
    public bool cullEnclosedUndergroundFaces = true;

    [Tooltip("Only hide enclosed faces this far below local surface (safety margin).")]
    public float enclosedUndergroundFaceMargin = 2f;

    [Header("Coarse LOD Underground Face Culling")]
    [Tooltip("Use true surface-shell approximation on coarse LODs (ignores underground caves on distant meshes).")]
    public bool useSurfaceShellOnCoarseLod = true;

    [Tooltip("Apply surface-shell approximation only on nodes with divisions >= this value.")]
    public int surfaceShellMinDivisions = 4;

    [Tooltip("On coarse LOD nodes, hide underground faces below local surface (surface shell approximation).")]
    public bool cullUndergroundFacesOnCoarseLod = false;

    [Tooltip("Apply coarse underground face culling only on nodes with divisions >= this value.")]
    public int coarseLodFaceCullMinDivisions = 4;

    [Tooltip("Safety margin below local surface for coarse LOD face culling.")]
    public float coarseLodFaceCullMargin = 1.5f;

    [Tooltip("Minimum top surface voxel layers preserved per XZ column on coarse LODs.")]
    public int coarseLodMinSurfaceVoxels = 2;

    [Header("Predictive Loading")]
    [Tooltip("Extra cells to pre-load ahead in the movement direction. 0 = off.")]
    public int predictionDistance = 3;

    [Tooltip("Player speed (units/s) above which predictive loading activates.")]
    public float predictionSpeedThreshold = 20f;

    [Tooltip("How strongly the load centre shifts ahead.")]
    public float predictionStrength = 1f;

    [Header("Seam transition (aggressive)")]
    [Tooltip("World-space band width around toroidal seams where aggressive pre-detailing activates.")]
    [Min(0f)] public float seamTransitionBandWorld = 512f;

    [Tooltip("Max temporary extra cells added to coverage radius while inside seam transition band.")]
    [Range(0, 4)] public int seamPrefetchExtraCells = 2;

    [Tooltip("Maximum LOD radius multiplier contribution near seam. 1.0 means up to +100% radius.")]
    [Min(0f)] public float seamLodBoostMax = 1.0f;

    [Tooltip("Falloff exponent for seam LOD boost (1=linear, >1 steeper near seam edge).")]
    [Min(0.1f)] public float seamBoostFalloff = 1.75f;

    [Tooltip("Extra cells that can be created per frame while near seam (helps opposite-side strips appear before crossing).")]
    [Range(0, 32)] public int seamPreloadBurstCreates = 8;

    [Tooltip("Log seam proximity / streaming counts while near a toroidal seam (rate-limited). Leave off to avoid console spam.")]
    public bool logSeamTransition = false;

    [Header("Seam stability (optional)")]
    [Tooltip("Keeps chunks just outside renderRadius alive for a short grace window near seams to reduce add/remove churn during seam oscillation.")]
    public bool useSeamChunkGrace = false;

    [Tooltip("Extra Chebyshev cells beyond renderRadius that qualify for grace retention.")]
    [Min(0)] public int seamGraceSlackCells = 1;

    [Tooltip("Seconds to keep an out-of-range chunk alive while in grace zone.")]
    [Min(0f)] public float seamGraceSeconds = 0.5f;

    [Header("References")]
    public Transform priority;
    public Camera playerCamera;   // used for frustum culling — assign the player camera
    public GameObject chunkPrefab;

    /// <summary>Chunk cell size in world units (XZ); assigned in <see cref="Start"/>.</summary>
    public float CellWorldSize { get; private set; }

    // -----------------------------------------------------------------------

    private readonly Dictionary<Vector2Int, Octree> activeCells
        = new Dictionary<Vector2Int, Octree>();
    private readonly Queue<Vector2Int> pendingCells = new Queue<Vector2Int>();
    private readonly HashSet<Vector2Int> _pendingCellsDedupe = new HashSet<Vector2Int>();
    private readonly List<PrioritizedJob> frameJobs = new List<PrioritizedJob>();
    private readonly List<JobCompleter> interactivePendingJobs = new List<JobCompleter>();
    private readonly List<ScheduledJob> inFlightJobs = new List<ScheduledJob>();
    private readonly Dictionary<Vector2Int, float> seamGraceFirstOutOfRangeAt = new Dictionary<Vector2Int, float>();

    private float cellSize;
    private Vector2Int lastPlayerCell = new Vector2Int(int.MaxValue, int.MaxValue);
    private Vector2Int lastLoadCentre = new Vector2Int(int.MaxValue, int.MaxValue);

    private Vector3 prevPlayerPos;
    private Vector3 playerVelocity;
    private const float VelocitySmooth = 0.15f;

    // Baked LOD radii shared across all octrees
    private float[] lodRadii;

    // Frustum planes extracted once per frame and shared with all octrees
    private Plane[] frustumPlanes;
    private readonly Stopwatch _meshingStopwatch = new Stopwatch();
    private readonly Stopwatch _completeStopwatch = new Stopwatch();
    private int _dynamicJobsPerFrame;
    private float _emaJobCostMs = 0f;
    private const float SpikeCutoffMultiplier = 1.75f;
    private const float SpikeBudgetDropFactor = 0.6f;
    private float _undergroundCullBelowY;
    private float _seamProximity01;
    private int _seamExtraCells;
    private float _lastSeamLogTime = float.NegativeInfinity;
    private readonly List<Vector2Int> _seamPrefetchCentres = new List<Vector2Int>(4);

    /// <summary>
    /// When the priority transform is moved by a toroidal wrap (delta = newPos - oldPos),
    /// call this with the same delta so velocity tracking and predictive loading do not see
    /// a spurious world-spanning displacement. Usually invoked from <see cref="ToroidalBoundaryWrap"/>.
    /// </summary>
    public void ApplyToroidalTrackingCorrection(Vector3 positionDelta)
    {
        prevPlayerPos += positionDelta;
    }

    // -----------------------------------------------------------------------

    public struct PrioritizedJob
    {
        public JobCompleter completer;
        public float sqrDistToPlayer;
    }

    private struct ScheduledJob
    {
        public JobHandle handle;
        public JobCompleter completer;
        public bool interactive;
    }

    // -----------------------------------------------------------------------
    //  Lifecycle
    // -----------------------------------------------------------------------

    private void Start()
    {
        int nodeResolution = (int)Mathf.Pow(2, divisions - 1);
        cellSize = chunkResolution * nodeResolution;
        CellWorldSize = cellSize;

        if (WorldModifications.Instance != null)
        {
            WorldModifications.Instance.worldSizeX = worldSizeX;
            WorldModifications.Instance.worldSizeZ = worldSizeZ;
        }

        BakeLodRadii();
        ComputeMaxColliderDivisions();

        UnityEngine.Debug.Log($"[OctreeGrid] cellSize={cellSize}  renderRadius={renderRadius}  " +
                  $"grid={(renderRadius * 2 + 1)}×{(renderRadius * 2 + 1)}  " +
                  $"world={worldSizeX}×{worldSizeZ}  " +
                  $"seed={noiseSeed}  noiseTypeId={noiseTypeId}  " +
                  $"minSubY={minSubsurfaceHeight}");

        // Fundamental-domain streaming dedups canonical chunks: when the render disk (2*renderRadius+1
        // cells) is wider than the world period in cells, multiple euclidean candidates map to the
        // same canonical key — only ONE is created, and the missing slots show up as holes/visible
        // world borders. Recommended margin is 2x for comfortable headroom.
        int diameterCells = renderRadius * 2 + 1;
        float minWorldHard = diameterCells * cellSize;
        float minWorldSoft = diameterCells * cellSize * 2f;
        if (worldSizeX > 0f && worldSizeX < minWorldHard)
            UnityEngine.Debug.LogError(
                $"[OctreeGrid] worldSizeX={worldSizeX} < required {minWorldHard:F0} " +
                $"({diameterCells} cells × {cellSize}). Toroidal wrap WILL show holes/borders. " +
                $"Pick worldSizeX >= {minWorldSoft:F0} for a comfortable seamless wrap.");
        else if (worldSizeX > 0f && worldSizeX < minWorldSoft)
            UnityEngine.Debug.LogWarning(
                $"[OctreeGrid] worldSizeX={worldSizeX} barely fits the render disk " +
                $"(min={minWorldHard:F0}). Recommended >= {minWorldSoft:F0}.");
        if (worldSizeZ > 0f && worldSizeZ < minWorldHard)
            UnityEngine.Debug.LogError(
                $"[OctreeGrid] worldSizeZ={worldSizeZ} < required {minWorldHard:F0} " +
                $"({diameterCells} cells × {cellSize}). Toroidal wrap WILL show holes/borders. " +
                $"Pick worldSizeZ >= {minWorldSoft:F0} for a comfortable seamless wrap.");
        else if (worldSizeZ > 0f && worldSizeZ < minWorldSoft)
            UnityEngine.Debug.LogWarning(
                $"[OctreeGrid] worldSizeZ={worldSizeZ} barely fits the render disk " +
                $"(min={minWorldHard:F0}). Recommended >= {minWorldSoft:F0}.");

        prevPlayerPos = priority.position;
        _dynamicJobsPerFrame = Mathf.Max(1, maxJobsPerFrame);
        RecomputeUndergroundCullBelowY();

        EnqueueDesiredCells(WorldToCell(priority.position));
        ProcessPendingCells(int.MaxValue);
    }

    private void Update()
    {
        Vector3 rawVelocity = (priority.position - prevPlayerPos) / Time.deltaTime;
        playerVelocity = Vector3.Lerp(playerVelocity, rawVelocity, VelocitySmooth);
        prevPlayerPos = priority.position;
        UpdateSeamTransitionContext(priority.position);

        Vector2Int playerCell = WorldToCell(priority.position);
        Vector2Int loadCentre = ComputeLoadCentre(playerCell);

        bool playerCellChanged = playerCell != lastPlayerCell;
        bool loadCentreChanged = loadCentre != lastLoadCentre;

        bool terrainRebased = false;
        // Rebase BEFORE Enqueue/Destroy so chunks already adopted the correct euclidean
        // representative when we measure distance and dedup canonical keys.
        if (playerCellChanged || loadCentreChanged)
            terrainRebased = RebaseDriftedCells(playerCell);

        if (loadCentreChanged)
        {
            lastLoadCentre = loadCentre;
            EnqueueDesiredCells(loadCentre);
        }

        // Preload opposite-side strip BEFORE crossing: run every frame while near a seam.
        // Bucket-gating caused strips to appear only after cell/bucket changes (too late visually).
        if (_seamProximity01 > 0f && seamTransitionBandWorld > 0f)
        {
            CollectSeamPrefetchCentres(priority.position, playerCell, _seamPrefetchCentres);
            for (int i = 0; i < _seamPrefetchCentres.Count; i++)
                EnqueueDesiredCells(_seamPrefetchCentres[i]);
        }

        if (playerCellChanged)
        {
            lastPlayerCell = playerCell;
            DestroyOutOfRangeCells(playerCell);
        }

        int seamBurst = Mathf.RoundToInt(_seamProximity01 * seamPreloadBurstCreates);
        ProcessPendingCells(newCellsPerFrame + seamBurst);

        // Extract frustum planes once per frame for all octrees to share.
        // If no camera is assigned, frustum culling is skipped (planes = null).
        frustumPlanes = playerCamera != null
            ? GeometryUtility.CalculateFrustumPlanes(playerCamera)
            : null;

        RecomputeUndergroundCullBelowY();

        foreach (var octree in activeCells.Values)
        {
            octree.maxNodeCreationsPerFrame = maxNodeCreationsPerFrame;
            octree.seamTransitionBandWorld = seamTransitionBandWorld;
            octree.seamProximity01 = _seamProximity01;
            octree.seamLodBoostMax = seamLodBoostMax;
            octree.seamBoostFalloff = seamBoostFalloff;
            octree.Tick(frameJobs, priority.position, frustumPlanes, _undergroundCullBelowY);
        }

        // ToroidalBoundaryWrap runs earlier (DefaultExecutionOrder -50) and calls SyncTransforms after
        // moving the player. RebaseDriftedCells then shifts every chunk MeshCollider transform by ±worldSize.
        // Without a second sync, PhysX can query stale poses for up to a physics step — worst case the
        // CharacterController sweeps against empty space for one frame at the seam.
        if ((worldSizeX > 0f || worldSizeZ > 0f) && (terrainRebased || playerCellChanged))
            Physics.SyncTransforms();

        if (logSeamTransition && _seamProximity01 > 0f && Time.time - _lastSeamLogTime > 0.5f)
        {
            _lastSeamLogTime = Time.time;
            UnityEngine.Debug.Log($"[OctreeGrid][Seam] proximity={_seamProximity01:F2} extraCells={_seamExtraCells} active={activeCells.Count} pending={pendingCells.Count}");
        }
    }

    private void LateUpdate()
    {
        _meshingStopwatch.Restart();
        int clampedMinJobs = Mathf.Clamp(minJobsPerFrame, 1, Mathf.Max(1, maxJobsPerFrame));
        int clampedMaxJobs = Mathf.Max(clampedMinJobs, maxJobsPerFrame);
        int dynamicScheduleBudget = Mathf.Clamp(_dynamicJobsPerFrame, clampedMinJobs, clampedMaxJobs);
        int scheduledCount = ScheduleJobs(dynamicScheduleBudget);
        int completedCount = CompleteFinishedJobs();

        _meshingStopwatch.Stop();
        float frameMeshingMs = (float)_meshingStopwatch.Elapsed.TotalMilliseconds;
        if (completedCount > 0)
        {
            float completeMs = (float)_completeStopwatch.Elapsed.TotalMilliseconds;
            float jobCostMs = completeMs / completedCount;
            _emaJobCostMs = _emaJobCostMs <= 0f ? jobCostMs : Mathf.Lerp(_emaJobCostMs, jobCostMs, 0.2f);

            if (_emaJobCostMs > 0f)
            {
                int nextBudget = Mathf.FloorToInt(targetMeshingBudgetMs / _emaJobCostMs);
                _dynamicJobsPerFrame = Mathf.Clamp(nextBudget, clampedMinJobs, clampedMaxJobs);
            }
        }

        // React immediately to spikes (EMA alone reacts too slowly).
        float spikeCutoffMs = Mathf.Max(1f, targetMeshingBudgetMs) * SpikeCutoffMultiplier;
        if (frameMeshingMs > spikeCutoffMs && dynamicScheduleBudget > clampedMinJobs)
        {
            int droppedBudget = Mathf.FloorToInt(dynamicScheduleBudget * SpikeBudgetDropFactor);
            _dynamicJobsPerFrame = Mathf.Clamp(droppedBudget, clampedMinJobs, clampedMaxJobs);
        }

        if (_meshingStopwatch.ElapsedMilliseconds > 5)
            UnityEngine.Debug.Log($"[OctreeGrid] meshing {_meshingStopwatch.ElapsedMilliseconds} ms  scheduled={scheduledCount} completed={completedCount} inFlight={inFlightJobs.Count} interactivePending={interactivePendingJobs.Count} next={_dynamicJobsPerFrame}");
    }

    private void OnDestroy()
    {
        for (int i = interactivePendingJobs.Count - 1; i >= 0; i--)
            interactivePendingJobs[i].cancel?.Invoke();
        interactivePendingJobs.Clear();

        for (int i = frameJobs.Count - 1; i >= 0; i--)
            frameJobs[i].completer.cancel?.Invoke();
        frameJobs.Clear();

        for (int i = inFlightJobs.Count - 1; i >= 0; i--)
        {
            var job = inFlightJobs[i];
            job.handle.Complete();
            job.completer.cancel?.Invoke();
        }
        inFlightJobs.Clear();

        foreach (var octree in activeCells.Values)
            octree.Shutdown();
        activeCells.Clear();
        seamGraceFirstOutOfRangeAt.Clear();
        pendingCells.Clear();

        // Drop pooled chunk GameObjects — keeping them across scene loads
        // would resurrect references to objects whose underlying prefab
        // and renderers belong to a now-destroyed scene.
        ChunkObjectPool.Clear();
    }

    private void OnGUI()
    {
        GUILayout.Label($"World: {worldSizeX} × {worldSizeZ} units  (toroidal)");
        GUILayout.Label($"Active cells: {activeCells.Count}  Pending: {pendingCells.Count}  Jobs: collect={frameJobs.Count} interactive={interactivePendingJobs.Count} inFlight={inFlightJobs.Count}");
        GUILayout.Label($"Meshing budget: current={_dynamicJobsPerFrame} targetMs={targetMeshingBudgetMs:F1} avgJobMs={_emaJobCostMs:F2}");
        GUILayout.Label($"Seam mode: proximity={_seamProximity01:F2} extraCells={_seamExtraCells} lodBoostMax={seamLodBoostMax:F2}");
        GUILayout.Label($"Underground node cull: {(enableUndergroundNodeCull ? $"ON belowY={_undergroundCullBelowY:F1}" : "OFF")}");
        GUILayout.Label($"Enclosed underground faces: {(cullEnclosedUndergroundFaces ? $"ON margin={enclosedUndergroundFaceMargin:F1}" : "OFF")}");
        GUILayout.Label($"Surface shell coarse: {(useSurfaceShellOnCoarseLod ? $"ON minDiv={surfaceShellMinDivisions}" : "OFF")}");
        GUILayout.Label($"Coarse underground faces: {(cullUndergroundFacesOnCoarseLod ? $"ON minDiv={coarseLodFaceCullMinDivisions} margin={coarseLodFaceCullMargin:F1} minSurfaceVox={coarseLodMinSurfaceVoxels}" : "OFF")}");
        GUILayout.Label($"Player: {priority.position}");
        GUILayout.Label($"Speed: {playerVelocity.magnitude:F1} u/s  " +
                        $"Prediction: {(playerVelocity.magnitude > predictionSpeedThreshold ? "ON" : "off")}");
        if (WorldModifications.Instance != null)
            GUILayout.Label($"Modifications: {WorldModifications.Instance.Count}");
    }

    public void EnqueueInteractiveJob(JobCompleter completer)
    {
        if (completer == null) return;
        interactivePendingJobs.Add(completer);
    }

    private int ScheduleJobs(int dynamicScheduleBudget)
    {
        int scheduleBudget = Mathf.Max(0, Mathf.Min(maxSchedulesPerFrame, dynamicScheduleBudget));
        int availableInFlight = Mathf.Max(0, maxInFlightJobs - inFlightJobs.Count);
        int toSchedule = Mathf.Min(scheduleBudget, availableInFlight);
        int scheduled = 0;

        while (scheduled < toSchedule && interactivePendingJobs.Count > 0)
        {
            var completer = interactivePendingJobs[0];
            interactivePendingJobs.RemoveAt(0);
            inFlightJobs.Add(new ScheduledJob
            {
                handle = completer.schedule(),
                completer = completer,
                interactive = true,
            });
            scheduled++;
        }

        if (frameJobs.Count > 1)
            frameJobs.Sort((a, b) => a.sqrDistToPlayer.CompareTo(b.sqrDistToPlayer));

        int remaining = toSchedule - scheduled;
        int normalCount = Mathf.Min(frameJobs.Count, remaining);
        for (int i = 0; i < normalCount; i++)
        {
            var job = frameJobs[i];
            inFlightJobs.Add(new ScheduledJob
            {
                handle = job.completer.schedule(),
                completer = job.completer,
                interactive = false,
            });
            scheduled++;
        }

        for (int i = normalCount; i < frameJobs.Count; i++)
            frameJobs[i].completer.cancel?.Invoke();
        frameJobs.Clear();

        return scheduled;
    }

    private int CompleteFinishedJobs()
    {
        _completeStopwatch.Restart();
        int completedTotal = 0;
        int completedInteractive = 0;
        int maxCompleted = Mathf.Max(1, maxCompletionsPerFrame);
        int maxInteractive = Mathf.Max(0, maxInteractiveCompletionsPerFrame);
        float budgetMs = Mathf.Max(0.1f, completeBudgetMs);

        bool budgetExceeded = false;
        for (int pass = 0; pass < 2 && !budgetExceeded; pass++)
        {
            for (int i = inFlightJobs.Count - 1; i >= 0; i--)
            {
                if (_completeStopwatch.Elapsed.TotalMilliseconds >= budgetMs ||
                    completedTotal >= maxCompleted)
                {
                    budgetExceeded = true;
                    break;
                }

                var scheduled = inFlightJobs[i];
                if (pass == 0)
                {
                    if (!scheduled.interactive || completedInteractive >= maxInteractive)
                        continue;
                }
                else if (scheduled.interactive)
                {
                    continue;
                }

                if (!scheduled.handle.IsCompleted) continue;

                scheduled.handle.Complete();
                scheduled.completer.onComplete();
                SwapBackRemove(inFlightJobs, i);
                completedTotal++;
                if (scheduled.interactive) completedInteractive++;
            }
        }

        _completeStopwatch.Stop();
        return completedTotal;
    }

    private static void SwapBackRemove<T>(List<T> list, int index)
    {
        int last = list.Count - 1;
        list[index] = list[last];
        list.RemoveAt(last);
    }

    private void OnDrawGizmos()
    {
        if (!Application.isPlaying) return;
        Gizmos.color = new Color(0f, 1f, 0.5f, 0.15f);
        foreach (var kv in activeCells)
        {
            // Draw at the physical representative — the canonical key may sit a full world away.
            Vector3 centre = CellToWorld(kv.Value.currentEuclideanCell)
                             + new Vector3(cellSize, 0, cellSize) * 0.5f;
            Gizmos.DrawWireCube(centre, new Vector3(cellSize, cellSize * 0.1f, cellSize));
        }
    }

    // -----------------------------------------------------------------------
    //  LOD baking
    // -----------------------------------------------------------------------

    /// <summary>
    /// Precomputes one subdivision radius per octree level from lodDistanceCurve
    /// and distributes it to every active (and future) Octree instance.
    ///
    /// Called once at Start. Call again at runtime if you tweak the curve in
    /// the Inspector (e.g. wrap in an Update check or expose a button).
    ///
    /// Level 0 = finest (divisions == 1, node size = chunkResolution voxels)
    /// Level divisions-1 = coarsest (root, node size = cellSize)
    /// </summary>
    /// <summary>
    /// Called by OctreeGridEditor Rebake and Apply button at runtime.
    /// Recalculates LOD radii from the curve, propagates to all active octrees,
    /// and forces each one to rebuild its tree so the change is visible immediately.
    /// </summary>
    public void RebakeLodRadii()
    {
        BakeLodRadii();
        ComputeMaxColliderDivisions();
        foreach (var oct in activeCells.Values)
        {
            oct.lodRadii = lodRadii;
            oct.maxColliderDivisions = _maxColliderDivisions;
            oct.undergroundCullBelowY = _undergroundCullBelowY;
            oct.cullEnclosedUndergroundFaces = cullEnclosedUndergroundFaces;
            oct.enclosedUndergroundFaceMargin = enclosedUndergroundFaceMargin;
            oct.useSurfaceShellOnCoarseLod = useSurfaceShellOnCoarseLod;
            oct.surfaceShellMinDivisions = surfaceShellMinDivisions;
            oct.cullUndergroundFacesOnCoarseLod = cullUndergroundFacesOnCoarseLod;
            oct.coarseLodFaceCullMinDivisions = coarseLodFaceCullMinDivisions;
            oct.coarseLodFaceCullMargin = coarseLodFaceCullMargin;
            oct.coarseLodMinSurfaceVoxels = coarseLodMinSurfaceVoxels;
            oct.seamTransitionBandWorld = seamTransitionBandWorld;
            oct.seamLodBoostMax = seamLodBoostMax;
            oct.seamBoostFalloff = seamBoostFalloff;
            oct.seamProximity01 = _seamProximity01;
            oct.caveMaxDepth = caveMaxDepth;
            oct.caveMaxDivisions = caveMaxDivisions;
            oct.verticalLodBias = verticalLodBias;
            oct.geoLayerBlobs = geoLayerBlobs;
            oct.ForceRebuildTree();
        }
    }

    // Cached result of ComputeMaxColliderDivisions
    private int _maxColliderDivisions = 2;

    /// <summary>
    /// Computes the highest division level where voxelSize stays at or below
    /// 32 units — the safe limit for PhysX triangle meshes.
    /// voxelSize = chunkResolution * 2^(level-1) / chunkResolution
    ///           = 2^(level-1)   ... wait, voxelSize = nodeScale / chunkResolution
    ///           = (chunkRes * 2^(div-1)) / chunkRes = 2^(div-1)
    /// So voxelSize <= 32  =>  2^(div-1) <= 32  =>  div-1 <= 5  =>  div <= 6.
    /// We cap at divisions so we never exceed the tree depth.
    /// </summary>
    private void ComputeMaxColliderDivisions()
    {
        // voxelSize at division d = 2^(d-1)
        // find highest d where 2^(d-1) <= 32
        int maxDiv = 1;
        for (int d = 1; d <= divisions; d++)
        {
            float voxelSize = Mathf.Pow(2f, d - 1);
            if (voxelSize <= 32f) maxDiv = d;
            else break;
        }
        _maxColliderDivisions = maxDiv;
        UnityEngine.Debug.Log($"[OctreeGrid] maxColliderDivisions={_maxColliderDivisions} " +
                  $"(voxelSize at that level = {Mathf.Pow(2f, _maxColliderDivisions - 1):F0}u)");
    }

    private void BakeLodRadii()
    {
        lodRadii = new float[divisions];

        for (int level = 0; level < divisions; level++)
        {
            // t = 0 at finest level, t = 1 at coarsest
            float t = divisions > 1 ? (float)level / (divisions - 1) : 0f;

            // Physical size of a node at this level
            float nodeSize = chunkResolution * Mathf.Pow(2f, level);

            float multiplier = lodDistanceCurve.Evaluate(t);
            lodRadii[level] = nodeSize * multiplier;
        }

        // Log so you can see the actual radii in the console
        var sb = new System.Text.StringBuilder("[OctreeGrid] LOD radii:\n");
        for (int i = 0; i < lodRadii.Length; i++)
            sb.AppendLine($"  level {i} (nodeSize={chunkResolution * Mathf.Pow(2f, i):F0}): radius={lodRadii[i]:F0} units");
        UnityEngine.Debug.Log(sb.ToString());
    }

    // -----------------------------------------------------------------------
    //  Predictive load centre
    // -----------------------------------------------------------------------

    private Vector2Int ComputeLoadCentre(Vector2Int playerCell)
    {
        if (predictionDistance <= 0) return playerCell;

        float speed = playerVelocity.magnitude;
        if (speed < predictionSpeedThreshold) return playerCell;

        float excess = (speed - predictionSpeedThreshold) * predictionStrength;
        float shiftCells = Mathf.Clamp(excess / cellSize, 0f, predictionDistance);
        Vector3 dir = playerVelocity.normalized;

        return new Vector2Int(
            playerCell.x + Mathf.RoundToInt(dir.x * shiftCells),
            playerCell.y + Mathf.RoundToInt(dir.z * shiftCells)
        );
    }

    private static float PosMod(float x, float m)
    {
        float r = x % m;
        return r < 0f ? r + m : r;
    }

    private static float AxisSeamDistance(float axisWorld, float worldSize)
    {
        if (worldSize <= 0f) return float.PositiveInfinity;
        float w = PosMod(axisWorld, worldSize);
        return Mathf.Min(w, worldSize - w);
    }

    private void UpdateSeamTransitionContext(Vector3 playerPos)
    {
        if (seamTransitionBandWorld <= 0f || (worldSizeX <= 0f && worldSizeZ <= 0f))
        {
            _seamProximity01 = 0f;
            _seamExtraCells = 0;
            return;
        }

        float band = seamTransitionBandWorld;
        float dx = AxisSeamDistance(playerPos.x, worldSizeX);
        float dz = AxisSeamDistance(playerPos.z, worldSizeZ);
        float pX = worldSizeX > 0f ? Mathf.Clamp01(1f - dx / band) : 0f;
        float pZ = worldSizeZ > 0f ? Mathf.Clamp01(1f - dz / band) : 0f;
        _seamProximity01 = Mathf.Max(pX, pZ);
        _seamExtraCells = Mathf.RoundToInt(Mathf.Clamp01(_seamProximity01) * seamPrefetchExtraCells);
    }

    private void CollectSeamPrefetchCentres(Vector3 playerPos, Vector2Int playerCell, List<Vector2Int> into)
    {
        into.Clear();
        if (_seamProximity01 <= 0f || seamTransitionBandWorld <= 0f) return;

        int shiftX = 0;
        int shiftZ = 0;

        if (worldSizeX > 0f && CellsPerAxisX > 0)
        {
            float w = PosMod(playerPos.x, worldSizeX);
            if (w < seamTransitionBandWorld) shiftX = CellsPerAxisX;
            else if (worldSizeX - w < seamTransitionBandWorld) shiftX = -CellsPerAxisX;
        }
        if (worldSizeZ > 0f && CellsPerAxisZ > 0)
        {
            float w = PosMod(playerPos.z, worldSizeZ);
            if (w < seamTransitionBandWorld) shiftZ = CellsPerAxisZ;
            else if (worldSizeZ - w < seamTransitionBandWorld) shiftZ = -CellsPerAxisZ;
        }

        if (shiftX == 0 && shiftZ == 0) return;

        // Anchor prefetch disks on the player's cell so wrapped representatives stay aligned with
        // streaming distance checks (loadCentre can lag behind when prediction is off).
        void TryAdd(Vector2Int c)
        {
            for (int i = 0; i < into.Count; i++)
                if (into[i] == c) return;
            into.Add(c);
        }

        if (shiftX != 0) TryAdd(new Vector2Int(playerCell.x + shiftX, playerCell.y));
        if (shiftZ != 0) TryAdd(new Vector2Int(playerCell.x, playerCell.y + shiftZ));
        if (shiftX != 0 && shiftZ != 0) TryAdd(new Vector2Int(playerCell.x + shiftX, playerCell.y + shiftZ));
    }

    // -----------------------------------------------------------------------
    //  Grid management
    // -----------------------------------------------------------------------

    private void EnqueueDesiredCells(Vector2Int centre)
    {
        Vector2Int playerCell = WorldToCell(priority.position);
        // While in the seam band, ensure at least seamPrefetchExtraCells is applied so toroidal
        // neighbours (e.g. ideal rep one period away) enter the disk before proximity ramps to 1.
        int seamCellMargin = 0;
        if (_seamProximity01 > 0f && seamTransitionBandWorld > 0f && seamPrefetchExtraCells > 0)
            seamCellMargin = Mathf.Max(_seamExtraCells, seamPrefetchExtraCells);
        else
            seamCellMargin = _seamExtraCells;
        int effectiveRadius = renderRadius + seamCellMargin;

        // Dedup by canonical key — when a small toroidal world wraps inside the render disk
        // multiple euclidean cells can map to the same canonical chunk. Always enqueue the
        // representative nearest to the player (not the toroidal alias seen from `centre`), so
        // distance sorting matches DestroyOutOfRangeCells and opposite-side strips spawn in range.
        var missingByCanonical = new Dictionary<Vector2Int, (Vector2Int ideal, float sqrDist)>();

        for (int dx = -effectiveRadius; dx <= effectiveRadius; dx++)
            for (int dz = -effectiveRadius; dz <= effectiveRadius; dz++)
            {
                var eucCell = new Vector2Int(centre.x + dx, centre.y + dz);
                var canonical = CanonicalCell(eucCell);
                if (activeCells.ContainsKey(canonical)) continue;

                var ideal = NearestEuclideanCell(canonical, playerCell);
                int pdx = ideal.x - playerCell.x;
                int pdz = ideal.y - playerCell.y;
                float sqrDist = pdx * pdx + pdz * pdz;

                if (!missingByCanonical.TryGetValue(canonical, out var existing) || sqrDist < existing.sqrDist)
                    missingByCanonical[canonical] = (ideal, sqrDist);
            }

        var sorted = new List<(Vector2Int ideal, float sqrDist)>(missingByCanonical.Values);
        sorted.Sort((a, b) => a.sqrDist.CompareTo(b.sqrDist));
        foreach (var entry in sorted)
        {
            if (_pendingCellsDedupe.Contains(entry.ideal)) continue;
            pendingCells.Enqueue(entry.ideal);
            _pendingCellsDedupe.Add(entry.ideal);
        }
    }

    private void DestroyOutOfRangeCells(Vector2Int playerCell)
    {
        var toRemove = new List<Vector2Int>();
        bool canUseGrace = useSeamChunkGrace && (worldSizeX > 0f || worldSizeZ > 0f) && seamGraceSeconds > 0f;
        int seamCellMargin = 0;
        if (_seamProximity01 > 0f && seamTransitionBandWorld > 0f && seamPrefetchExtraCells > 0)
            seamCellMargin = Mathf.Max(_seamExtraCells, seamPrefetchExtraCells);
        else
            seamCellMargin = _seamExtraCells;
        int effectiveRadius = renderRadius + seamCellMargin;
        int graceMaxChebyshev = effectiveRadius + Mathf.Max(0, seamGraceSlackCells);
        float now = Time.time;

        foreach (var kv in activeCells)
        {
            // Measure range using the representative nearest the player for this canonical chunk.
            // Using currentEuclideanCell alone can destroy toroidal neighbours that are still
            // visible (wrong-period alias) or keep the wrong instance — ideal matches EnqueueDesiredCells.
            var canonical = kv.Key;
            var ideal = NearestEuclideanCell(canonical, playerCell);
            int dx = Mathf.Abs(ideal.x - playerCell.x);
            int dz = Mathf.Abs(ideal.y - playerCell.y);
            int chebyshev = Mathf.Max(dx, dz);

            if (chebyshev <= effectiveRadius)
            {
                seamGraceFirstOutOfRangeAt.Remove(kv.Key);
                continue;
            }

            if (canUseGrace && chebyshev <= graceMaxChebyshev)
            {
                if (!seamGraceFirstOutOfRangeAt.TryGetValue(kv.Key, out float firstOutAt))
                {
                    seamGraceFirstOutOfRangeAt[kv.Key] = now;
                    continue;
                }

                if (now - firstOutAt < seamGraceSeconds)
                    continue;
            }

            kv.Value.Shutdown();
            toRemove.Add(kv.Key);
            seamGraceFirstOutOfRangeAt.Remove(kv.Key);
        }

        foreach (var k in toRemove)
            activeCells.Remove(k);
    }
    /// <summary>
    /// For every active chunk, picks the euclidean representative closest to the player
    /// (canonical + k * cellsPerAxis on each toroidal axis) and shifts the Octree's cellOrigin
    /// to match. After a player wrap (e.g. via <see cref="ToroidalBoundaryWrap"/>) this puts
    /// the chunks behind the player on the near side of the seam without any rebuild.
    /// </summary>
    /// <returns>True if at least one chunk shifted its physical origin this frame.</returns>
    private bool RebaseDriftedCells(Vector2Int playerEucCell)
    {
        if (CellsPerAxisX <= 0 && CellsPerAxisZ <= 0) return false;

        bool any = false;
        foreach (var kv in activeCells)
        {
            var canonical = kv.Key;
            var oct = kv.Value;
            var nearest = NearestEuclideanCell(canonical, playerEucCell);
            if (nearest != oct.currentEuclideanCell)
            {
                oct.RebaseCellOrigin(CellToWorld(nearest), nearest);
                any = true;
            }
        }
        return any;
    }

    private void ProcessPendingCells(int maxThisFrame)
    {
        int created = 0;
        while (pendingCells.Count > 0 && created < maxThisFrame)
        {
            Vector2Int eucCell = pendingCells.Dequeue();
            _pendingCellsDedupe.Remove(eucCell);
            Vector2Int canonical = CanonicalCell(eucCell);
            if (!activeCells.ContainsKey(canonical))
            {
                activeCells[canonical] = CreateOctree(eucCell);
                created++;
            }
        }
    }

    // -----------------------------------------------------------------------
    //  Octree factory
    // -----------------------------------------------------------------------

    private Octree CreateOctree(Vector2Int eucCell)
    {
        // eucCell is the *physical* (euclidean) representative used for cellOrigin; its
        // canonical wrap is the dictionary key in activeCells (computed by the caller).
        var go = new GameObject($"Cell_{eucCell.x}_{eucCell.y}");
        go.transform.parent = transform;

        var octree = go.AddComponent<Octree>();
        octree.worldSizeX = worldSizeX;
        octree.worldSizeZ = worldSizeZ;
        octree.surfaceBaseHeight = surfaceBaseHeight;
        octree.surfaceNoiseAmplitude = surfaceNoiseAmplitude;
        // ── WorldConfig-driven fields (added) ──────────────────────────────
        octree.noiseSeed = noiseSeed;
        octree.noiseTypeId = noiseTypeId;
        octree.minSubsurfaceHeight = minSubsurfaceHeight;
        octree.cavesEnabled = cavesEnabled;
        octree.caveNoiseFrequency = caveNoiseFrequency;
        octree.caveNoiseThreshold = caveNoiseThreshold;
        octree.caveNoiseAmplitudeY = caveNoiseAmplitudeY;
        octree.caveSurfaceFadeRange = caveSurfaceFadeRange;
        octree.caveMaxDepth = caveMaxDepth;
        octree.caveMaxDivisions = caveMaxDivisions;
        octree.verticalLodBias = verticalLodBias;
        octree.geoLayerBlobs = geoLayerBlobs;
        // ───────────────────────────────────────────────────────────────────
        octree.divisions = divisions;
        octree.chunkResolution = chunkResolution;
        octree.maxNodeCreationsPerFrame = maxNodeCreationsPerFrame;
        octree.priority = priority;
        octree.chunkPrefab = chunkPrefab;
        octree.cellOrigin = CellToWorld(eucCell);
        octree.currentEuclideanCell = eucCell;
        octree.maxColliderDivisions = _maxColliderDivisions;
        octree.undergroundCullBelowY = _undergroundCullBelowY;
        octree.cullEnclosedUndergroundFaces = cullEnclosedUndergroundFaces;
        octree.enclosedUndergroundFaceMargin = enclosedUndergroundFaceMargin;
        octree.useSurfaceShellOnCoarseLod = useSurfaceShellOnCoarseLod;
        octree.surfaceShellMinDivisions = surfaceShellMinDivisions;
        octree.cullUndergroundFacesOnCoarseLod = cullUndergroundFacesOnCoarseLod;
        octree.coarseLodFaceCullMinDivisions = coarseLodFaceCullMinDivisions;
        octree.coarseLodFaceCullMargin = coarseLodFaceCullMargin;
        octree.coarseLodMinSurfaceVoxels = coarseLodMinSurfaceVoxels;
        octree.seamTransitionBandWorld = seamTransitionBandWorld;
        octree.seamProximity01 = _seamProximity01;
        octree.seamLodBoostMax = seamLodBoostMax;
        octree.seamBoostFalloff = seamBoostFalloff;
        octree.lodRadii = lodRadii;   // share the baked array — read-only at runtime

        octree.Initialize();
        return octree;
    }

    private void RecomputeUndergroundCullBelowY()
    {
        if (!enableUndergroundNodeCull)
        {
            _undergroundCullBelowY = float.NegativeInfinity;
            return;
        }

        float lowestPossibleSurfaceY = surfaceBaseHeight - Mathf.Abs(surfaceNoiseAmplitude);
        _undergroundCullBelowY = lowestPossibleSurfaceY - Mathf.Max(0f, undergroundSolidCullMargin);
    }

    // -----------------------------------------------------------------------
    //  Coordinate helpers
    // -----------------------------------------------------------------------

    public Vector2Int WorldToCell(Vector3 worldPos) => new Vector2Int(
        Mathf.FloorToInt(worldPos.x / cellSize),
        Mathf.FloorToInt(worldPos.z / cellSize));

    public Vector3 CellToWorld(Vector2Int cell) =>
        new Vector3(cell.x * cellSize, 0f, cell.y * cellSize);

    /// <summary>
    /// Number of chunk cells per toroidal period on X. Zero when worldSizeX &lt;= 0
    /// (open world: canonical == euclidean and no rebase happens).
    /// </summary>
    private int CellsPerAxisX => worldSizeX > 0f && cellSize > 0f
        ? Mathf.Max(1, Mathf.RoundToInt(worldSizeX / cellSize))
        : 0;

    private int CellsPerAxisZ => worldSizeZ > 0f && cellSize > 0f
        ? Mathf.Max(1, Mathf.RoundToInt(worldSizeZ / cellSize))
        : 0;

    /// <summary>
    /// Wraps a euclidean cell index into [0, CellsPerAxis) on each toroidal axis.
    /// Acts as identity on axes where worldSize == 0.
    /// </summary>
    public Vector2Int CanonicalCell(Vector2Int eucCell)
    {
        int nx = CellsPerAxisX;
        int nz = CellsPerAxisZ;
        int cx = nx > 0 ? ((eucCell.x % nx) + nx) % nx : eucCell.x;
        int cz = nz > 0 ? ((eucCell.y % nz) + nz) % nz : eucCell.y;
        return new Vector2Int(cx, cz);
    }

    /// <summary>
    /// Returns the euclidean representative (canonical + k*CellsPerAxis on each axis) closest
    /// to <paramref name="playerEucCell"/>. Used to keep each chunk's physical origin next to
    /// the player after a toroidal seam crossing.
    /// </summary>
    public Vector2Int NearestEuclideanCell(Vector2Int canonical, Vector2Int playerEucCell)
    {
        int nx = CellsPerAxisX;
        int nz = CellsPerAxisZ;
        int rx = canonical.x;
        int rz = canonical.y;

        if (nx > 0)
        {
            int delta = playerEucCell.x - canonical.x;
            int k = Mathf.RoundToInt((float)delta / nx);
            rx = canonical.x + k * nx;
        }
        if (nz > 0)
        {
            int delta = playerEucCell.y - canonical.y;
            int k = Mathf.RoundToInt((float)delta / nz);
            rz = canonical.y + k * nz;
        }
        return new Vector2Int(rx, rz);
    }

    // -----------------------------------------------------------------------
    //  Block interaction support
    // -----------------------------------------------------------------------

    public Node FindLeafAt(Vector3 worldPos)
    {
        Vector2Int eucCell = WorldToCell(worldPos);
        Vector2Int canonical = CanonicalCell(eucCell);
        if (!activeCells.TryGetValue(canonical, out Octree octree)) return null;

        // The chunk's tree is positioned at currentEuclideanCell, which may differ from the
        // worldPos's natural eucCell by a multiple of cellsPerAxis. Translate the query into
        // the chunk's frame so AABB containment in Octree.FindLeafAt still matches.
        Vector2Int chunkEuc = octree.currentEuclideanCell;
        if (chunkEuc != eucCell)
        {
            Vector3 delta = new Vector3(
                (chunkEuc.x - eucCell.x) * cellSize,
                0f,
                (chunkEuc.y - eucCell.y) * cellSize);
            worldPos += delta;
        }
        return octree.FindLeafAt(worldPos);
    }
}