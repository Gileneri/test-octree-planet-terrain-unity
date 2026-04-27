using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;

/// <summary>
/// Manages a grid of Octrees centred on the player with seamless toroidal wrap.
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

    [Header("Render Distance & Performance")]
    [Tooltip("Cells loaded in each direction. 6 = 13×13 grid.")]
    public int renderRadius = 6;

    [Tooltip("Max chunk mesh jobs completed per frame across ALL cells.")]
    public int maxJobsPerFrame = 120;

    [Tooltip("New octree cells activated per frame when player moves.")]
    public int newCellsPerFrame = 3;

    [Tooltip("Max octree node subdivisions per cell per frame.")]
    public int maxNodeCreationsPerFrame = 50;

    [Header("Predictive Loading")]
    [Tooltip("Extra cells to pre-load ahead in the movement direction. 0 = off.")]
    public int predictionDistance = 3;

    [Tooltip("Player speed (units/s) above which predictive loading activates.")]
    public float predictionSpeedThreshold = 20f;

    [Tooltip("How strongly the load centre shifts ahead.")]
    public float predictionStrength = 1f;

    [Header("References")]
    public Transform priority;
    public Camera playerCamera;   // used for frustum culling — assign the player camera
    public GameObject chunkPrefab;

    // -----------------------------------------------------------------------

    private readonly Dictionary<Vector2Int, Octree> activeCells
        = new Dictionary<Vector2Int, Octree>();
    private readonly Queue<Vector2Int> pendingCells = new Queue<Vector2Int>();
    private readonly List<PrioritizedJob> frameJobs = new List<PrioritizedJob>();

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

    // -----------------------------------------------------------------------

    public struct PrioritizedJob
    {
        public JobCompleter completer;
        public float sqrDistToPlayer;
    }

    // -----------------------------------------------------------------------
    //  Lifecycle
    // -----------------------------------------------------------------------

    private void Start()
    {
        int nodeResolution = (int)Mathf.Pow(2, divisions - 1);
        cellSize = chunkResolution * nodeResolution;

        if (WorldModifications.Instance != null)
        {
            WorldModifications.Instance.worldSizeX = worldSizeX;
            WorldModifications.Instance.worldSizeZ = worldSizeZ;
        }

        BakeLodRadii();
        ComputeMaxColliderDivisions();

        Debug.Log($"[OctreeGrid] cellSize={cellSize}  renderRadius={renderRadius}  " +
                  $"grid={(renderRadius * 2 + 1)}×{(renderRadius * 2 + 1)}  " +
                  $"world={worldSizeX}×{worldSizeZ}  " +
                  $"seed={noiseSeed}  noiseTypeId={noiseTypeId}  " +
                  $"minSubY={minSubsurfaceHeight}");

        prevPlayerPos = priority.position;

        EnqueueDesiredCells(WorldToCell(priority.position));
        ProcessPendingCells(int.MaxValue);
    }

    private void Update()
    {
        Vector3 rawVelocity = (priority.position - prevPlayerPos) / Time.deltaTime;
        playerVelocity = Vector3.Lerp(playerVelocity, rawVelocity, VelocitySmooth);
        prevPlayerPos = priority.position;

        Vector2Int playerCell = WorldToCell(priority.position);
        Vector2Int loadCentre = ComputeLoadCentre(playerCell);

        if (loadCentre != lastLoadCentre)
        {
            lastLoadCentre = loadCentre;
            EnqueueDesiredCells(loadCentre);
        }

        if (playerCell != lastPlayerCell)
        {
            lastPlayerCell = playerCell;
            DestroyOutOfRangeCells(playerCell);
        }

        ProcessPendingCells(newCellsPerFrame);

        // Extract frustum planes once per frame for all octrees to share.
        // If no camera is assigned, frustum culling is skipped (planes = null).
        frustumPlanes = playerCamera != null
            ? GeometryUtility.CalculateFrustumPlanes(playerCamera)
            : null;

        foreach (var octree in activeCells.Values)
            octree.Tick(frameJobs, priority.position, frustumPlanes);
    }

    private void LateUpdate()
    {
        if (frameJobs.Count == 0) return;

        var sw = new System.Diagnostics.Stopwatch();
        sw.Start();

        frameJobs.Sort((a, b) => a.sqrDistToPlayer.CompareTo(b.sqrDistToPlayer));

        int count = Mathf.Min(frameJobs.Count, maxJobsPerFrame);

        var handles = new NativeArray<JobHandle>(count, Allocator.Temp);
        for (int i = 0; i < count; i++)
            handles[i] = frameJobs[i].completer.schedule();
        JobHandle.CompleteAll(handles);
        handles.Dispose();

        for (int i = 0; i < count; i++)
            frameJobs[i].completer.onComplete();

        for (int i = count; i < frameJobs.Count; i++)
            frameJobs[i].completer.cancel?.Invoke();

        frameJobs.Clear();

        sw.Stop();
        if (sw.ElapsedMilliseconds > 5)
            Debug.Log($"[OctreeGrid] meshing {sw.ElapsedMilliseconds} ms  jobs={count}");
    }

    private void OnDestroy()
    {
        foreach (var octree in activeCells.Values)
            octree.Shutdown();
        activeCells.Clear();
        pendingCells.Clear();
    }

    private void OnGUI()
    {
        GUILayout.Label($"World: {worldSizeX} × {worldSizeZ} units  (toroidal)");
        GUILayout.Label($"Active cells: {activeCells.Count}  Pending: {pendingCells.Count}  Jobs: {frameJobs.Count}");
        GUILayout.Label($"Player: {priority.position}");
        GUILayout.Label($"Speed: {playerVelocity.magnitude:F1} u/s  " +
                        $"Prediction: {(playerVelocity.magnitude > predictionSpeedThreshold ? "ON" : "off")}");
        if (WorldModifications.Instance != null)
            GUILayout.Label($"Modifications: {WorldModifications.Instance.Count}");
    }

    private void OnDrawGizmos()
    {
        if (!Application.isPlaying) return;
        Gizmos.color = new Color(0f, 1f, 0.5f, 0.15f);
        foreach (var kv in activeCells)
        {
            Vector3 centre = CellToWorld(kv.Key) + new Vector3(cellSize, 0, cellSize) * 0.5f;
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
            oct.undergroundCullDepth = cellSize * 1.5f;
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
        Debug.Log($"[OctreeGrid] maxColliderDivisions={_maxColliderDivisions} " +
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
        Debug.Log(sb.ToString());
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

    // -----------------------------------------------------------------------
    //  Grid management
    // -----------------------------------------------------------------------

    private void EnqueueDesiredCells(Vector2Int centre)
    {
        Vector2Int playerCell = WorldToCell(priority.position);
        var missing = new List<(Vector2Int cell, float sqrDist)>();

        for (int dx = -renderRadius; dx <= renderRadius; dx++)
            for (int dz = -renderRadius; dz <= renderRadius; dz++)
            {
                var cell = new Vector2Int(centre.x + dx, centre.y + dz);
                if (!activeCells.ContainsKey(cell))
                {
                    int pdx = cell.x - playerCell.x;
                    int pdz = cell.y - playerCell.y;
                    missing.Add((cell, pdx * pdx + pdz * pdz));
                }
            }

        missing.Sort((a, b) => a.sqrDist.CompareTo(b.sqrDist));
        foreach (var (cell, _) in missing)
            pendingCells.Enqueue(cell);
    }

    private void DestroyOutOfRangeCells(Vector2Int playerCell)
    {
        var toRemove = new List<Vector2Int>();
        foreach (var kv in activeCells)
        {
            int dx = Mathf.Abs(kv.Key.x - playerCell.x);
            int dz = Mathf.Abs(kv.Key.y - playerCell.y);
            if (dx > renderRadius || dz > renderRadius)
            {
                kv.Value.Shutdown();
                toRemove.Add(kv.Key);
            }
        }
        foreach (var k in toRemove) activeCells.Remove(k);
    }

    private void ProcessPendingCells(int maxThisFrame)
    {
        int created = 0;
        while (pendingCells.Count > 0 && created < maxThisFrame)
        {
            Vector2Int cell = pendingCells.Dequeue();
            if (!activeCells.ContainsKey(cell))
            {
                activeCells[cell] = CreateOctree(cell);
                created++;
            }
        }
    }

    // -----------------------------------------------------------------------
    //  Octree factory
    // -----------------------------------------------------------------------

    private Octree CreateOctree(Vector2Int cell)
    {
        var go = new GameObject($"Cell_{cell.x}_{cell.y}");
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
        // ───────────────────────────────────────────────────────────────────
        octree.divisions = divisions;
        octree.chunkResolution = chunkResolution;
        octree.maxNodeCreationsPerFrame = maxNodeCreationsPerFrame;
        octree.priority = priority;
        octree.chunkPrefab = chunkPrefab;
        octree.cellOrigin = CellToWorld(cell);
        octree.maxColliderDivisions = _maxColliderDivisions;
        octree.undergroundCullDepth = cellSize * 1.5f;
        octree.lodRadii = lodRadii;   // share the baked array — read-only at runtime

        octree.Initialize();
        return octree;
    }

    // -----------------------------------------------------------------------
    //  Coordinate helpers
    // -----------------------------------------------------------------------

    public Vector2Int WorldToCell(Vector3 worldPos) => new Vector2Int(
        Mathf.FloorToInt(worldPos.x / cellSize),
        Mathf.FloorToInt(worldPos.z / cellSize));

    public Vector3 CellToWorld(Vector2Int cell) =>
        new Vector3(cell.x * cellSize, 0f, cell.y * cellSize);

    // -----------------------------------------------------------------------
    //  Block interaction support
    // -----------------------------------------------------------------------

    public Node FindLeafAt(Vector3 worldPos)
    {
        Vector2Int cell = WorldToCell(worldPos);
        return activeCells.TryGetValue(cell, out Octree octree)
            ? octree.FindLeafAt(worldPos)
            : null;
    }
}