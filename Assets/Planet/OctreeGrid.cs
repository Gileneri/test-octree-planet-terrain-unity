using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;

/// <summary>
/// Manages a grid of Octrees centred on the player with seamless toroidal wrap.
///
/// KEY FEATURES
/// ─────────────
/// • Configurable render distance (renderRadius cells in each direction).
/// • Global per-frame job budget shared across ALL active octrees.
///   Jobs are sorted by distance to the player so closer chunks mesh first.
/// • Predictive generation: the load centre is shifted ahead of the player
///   proportional to speed, so chunks are ready before the player arrives.
/// • New cells are introduced gradually to avoid frame spikes.
/// • Cells leaving the radius are destroyed immediately.
///
/// RECOMMENDED SETTINGS FOR A 50 KM WORLD
/// ────────────────────────────────────────
///   worldSizeX / worldSizeZ : 50000
///   divisions               : 7        (cell size ≈ 1024 units at chunkRes=8)
///   chunkResolution         : 8
///   renderRadius            : 6        (13×13 = 169 cells, ~13 km view distance)
///   maxJobsPerFrame         : 120
///   newCellsPerFrame        : 3
///   predictionDistance      : 3        (cells ahead to pre-load)
///   predictionSpeedThreshold: 20       (units/s before prediction kicks in)
///   surfaceNoiseAmplitude   : 300      (taller mountains for a bigger world)
///
/// SETUP
/// ─────
/// 1. Create a GameObject → "OctreeGrid" → Add Component → OctreeGrid.
/// 2. Fill in the Inspector fields above.
/// 3. Add a WorldModifications component to any scene GameObject.
/// 4. Point BlockInteraction.octreeGrid at this component.
/// </summary>
public class OctreeGrid : MonoBehaviour
{
    [Header("World  (50 km planet example: worldSizeX = worldSizeZ = 50000)")]
    public float worldSizeX = 50000f;
    public float worldSizeZ = 50000f;

    [Header("Terrain")]
    public float surfaceBaseHeight = 0f;
    [Tooltip("Max terrain height. Scale up for larger worlds (300+ for 50 km).")]
    public float surfaceNoiseAmplitude = 300f;

    [Header("Octree per cell")]
    [Tooltip("Octree depth. 7 with chunkResolution=8 gives ~1024 unit cells.")]
    public int divisions = 7;
    [Tooltip("Voxels per chunk side. Lower = smaller cells = smoother streaming.")]
    public int chunkResolution = 8;
    public float innerRadiusPadding = 2f;

    [Header("Render Distance & Performance")]
    [Tooltip("Cells loaded in each direction. 6 = 13×13 grid ≈ 13 km at 50km scale.")]
    public int renderRadius = 6;

    [Tooltip("Max chunk mesh jobs completed per frame across ALL cells.")]
    public int maxJobsPerFrame = 120;

    [Tooltip("New octree cells activated per frame when player moves. 3 = fast load, smooth.")]
    public int newCellsPerFrame = 3;

    [Tooltip("Max octree node subdivisions per cell per frame.")]
    public int maxNodeCreationsPerFrame = 50;

    [Header("Predictive Loading")]
    [Tooltip("How many extra cells ahead of the player to pre-load in the movement direction.\n" +
             "Higher = less pop-in at speed, more memory used.\n" +
             "0 = disabled. Recommended: 2–4.")]
    public int predictionDistance = 3;

    [Tooltip("Player speed (units/s) above which predictive loading activates.")]
    public float predictionSpeedThreshold = 20f;

    [Tooltip("How strongly the load centre shifts ahead. 1 = one full cell per " +
             "predictionDistance unit of speed above threshold.")]
    public float predictionStrength = 1f;

    [Header("References")]
    public Transform priority;
    public GameObject chunkPrefab;

    // -----------------------------------------------------------------------

    // Active cells: integer grid coordinate → Octree instance
    private readonly Dictionary<Vector2Int, Octree> activeCells
        = new Dictionary<Vector2Int, Octree>();

    // Cells queued to be created (populated when player changes grid cell)
    private readonly Queue<Vector2Int> pendingCells = new Queue<Vector2Int>();

    // All pending jobs this frame, sorted by distance before completing
    private readonly List<PrioritizedJob> frameJobs = new List<PrioritizedJob>();

    private float cellSize;
    private Vector2Int lastPlayerCell = new Vector2Int(int.MaxValue, int.MaxValue);
    private Vector2Int lastLoadCentre = new Vector2Int(int.MaxValue, int.MaxValue);

    // Velocity tracking for predictive loading
    private Vector3 prevPlayerPos;
    private Vector3 playerVelocity;      // smoothed world-space velocity
    private const float VelocitySmooth = 0.15f; // lerp factor per frame

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

        Debug.Log($"[OctreeGrid] cellSize={cellSize}  renderRadius={renderRadius}  " +
                  $"grid={(renderRadius * 2 + 1)}×{(renderRadius * 2 + 1)}  " +
                  $"maxJobsPerFrame={maxJobsPerFrame}  " +
                  $"worldSize={worldSizeX}×{worldSizeZ}");

        prevPlayerPos = priority.position;

        // Seed the pending queue immediately so the first cells load at start
        EnqueueDesiredCells(WorldToCell(priority.position));
        ProcessPendingCells(int.MaxValue); // load all cells synchronously on first frame
    }

    private void Update()
    {
        // --- Velocity tracking (smoothed) ---
        Vector3 rawVelocity = (priority.position - prevPlayerPos) / Time.deltaTime;
        playerVelocity = Vector3.Lerp(playerVelocity, rawVelocity, VelocitySmooth);
        prevPlayerPos = priority.position;

        // --- Predictive load centre ---
        // Shift the grid centre ahead of the player in the movement direction,
        // proportional to how fast they are moving above the threshold.
        Vector2Int playerCell = WorldToCell(priority.position);
        Vector2Int loadCentre = ComputeLoadCentre(playerCell);

        // Refresh the grid whenever the LOAD CENTRE changes (not just playerCell),
        // so predictive cells are created even before the player crosses a boundary.
        if (loadCentre != lastLoadCentre)
        {
            lastLoadCentre = loadCentre;
            EnqueueDesiredCells(loadCentre);
        }

        // Destroy cells that are out of range relative to PLAYER (not load centre)
        // so we don't keep geometry that the player can no longer see.
        if (playerCell != lastPlayerCell)
        {
            lastPlayerCell = playerCell;
            DestroyOutOfRangeCells(playerCell);
        }

        // Activate a few pending cells per frame
        ProcessPendingCells(newCellsPerFrame);

        // Tick every active octree (traverse + collect pending jobs)
        foreach (var octree in activeCells.Values)
            octree.Tick(frameJobs, priority.position);
    }

    private void LateUpdate()
    {
        if (frameJobs.Count == 0) return;

        var sw = new System.Diagnostics.Stopwatch();
        sw.Start();

        // Sort closest-first so nearby chunks appear before distant ones
        frameJobs.Sort((a, b) => a.sqrDistToPlayer.CompareTo(b.sqrDistToPlayer));

        // Clamp to budget
        int count = Mathf.Min(frameJobs.Count, maxJobsPerFrame);

        // Schedule all jobs in the budget in one batch (maximises parallelism)
        var handles = new NativeArray<JobHandle>(count, Allocator.Temp);
        for (int i = 0; i < count; i++)
            handles[i] = frameJobs[i].completer.schedule();
        JobHandle.CompleteAll(handles);
        handles.Dispose();

        // Apply meshes
        for (int i = 0; i < count; i++)
            frameJobs[i].completer.onComplete();

        // Jobs beyond the budget are discarded this frame — they will be
        // re-queued next frame because their node still has needsDrawn = true.
        // We must cancel them cleanly so the node doesn't stay locked.
        for (int i = count; i < frameJobs.Count; i++)
            frameJobs[i].completer.cancel?.Invoke();

        frameJobs.Clear();

        sw.Stop();
        if (sw.ElapsedMilliseconds > 5)
            Debug.Log($"[OctreeGrid] meshing {sw.ElapsedMilliseconds} ms  ({count} jobs)");
    }

    private void OnDisable()
    {
        foreach (var octree in activeCells.Values)
            octree.Shutdown();
        activeCells.Clear();
        pendingCells.Clear();
    }

    private void OnGUI()
    {
        GUILayout.Label($"World: {worldSizeX} × {worldSizeZ} units  (toroidal)");
        GUILayout.Label($"Active cells: {activeCells.Count}  Pending: {pendingCells.Count}");
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
    //  Predictive load centre
    // -----------------------------------------------------------------------

    /// <summary>
    /// Returns the grid cell that should be used as the centre of the loaded
    /// region.  When the player is moving fast enough, this is shifted ahead
    /// in the movement direction so cells are ready before the player arrives.
    ///
    /// Example: player moving at 80 u/s with threshold=20, strength=1,
    /// predictionDistance=3 → load centre shifts up to 3 cells ahead.
    /// </summary>
    private Vector2Int ComputeLoadCentre(Vector2Int playerCell)
    {
        if (predictionDistance <= 0) return playerCell;

        float speed = playerVelocity.magnitude;
        if (speed < predictionSpeedThreshold) return playerCell;

        // How far ahead to shift, clamped to predictionDistance
        float excess = (speed - predictionSpeedThreshold) * predictionStrength;
        float shiftCells = Mathf.Clamp(excess / cellSize, 0f, predictionDistance);

        // Direction of movement projected onto the XZ grid
        Vector3 dir = playerVelocity.normalized;
        float offX = dir.x * shiftCells;
        float offZ = dir.z * shiftCells;

        return new Vector2Int(
            playerCell.x + Mathf.RoundToInt(offX),
            playerCell.y + Mathf.RoundToInt(offZ)
        );
    }

    // -----------------------------------------------------------------------
    //  Grid management
    // -----------------------------------------------------------------------

    /// <summary>
    /// Fills pendingCells with every cell in the render radius around
    /// <paramref name="centre"/> that is not already active, sorted by
    /// distance to the real player cell so nearby chunks always load first.
    /// </summary>
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
                    // Order by distance to REAL player, not to the predicted centre,
                    // so cells directly under the player always appear first.
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
        octree.divisions = divisions;
        octree.chunkResolution = chunkResolution;
        octree.innerRadiusPadding = innerRadiusPadding;
        octree.maxNodeCreationsPerFrame = maxNodeCreationsPerFrame;
        octree.priority = priority;
        octree.chunkPrefab = chunkPrefab;
        octree.cellOrigin = CellToWorld(cell);

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