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
/// • New cells are introduced one per frame to spread the load when the
///   player crosses a cell boundary.
/// • Cells leaving the radius are destroyed immediately (their geometry is gone
///   from the player's view so the cost is zero).
///
/// SETUP
/// ─────
/// 1. Create a GameObject → name it "OctreeGrid".
/// 2. Add Component → OctreeGrid.
/// 3. Fill in the Inspector:
///      priority              → player Transform
///      chunkPrefab           → prefab with MeshFilter + MeshRenderer
///      renderRadius          → cells in each direction (3=7×7, 8=17×17, …)
///      maxJobsPerFrame       → how many chunk jobs to COMPLETE per frame
///                              (50–200 is a good range; lower = smoother,
///                               higher = faster initial load)
///      newCellsPerFrame      → how many new octree cells to activate per frame
///                              when the player moves (1–3 recommended)
///      worldSizeX/Z          → wrap period in world units
///      divisions             → octree depth per cell (keep 6–8 for good LOD)
///      chunkResolution       → voxels per chunk side
///      surfaceBaseHeight     → Y of sea level
///      surfaceNoiseAmplitude → max terrain height
/// 4. Add a WorldModifications component to any scene GameObject.
/// 5. Point BlockInteraction.octreeGrid at this component.
/// </summary>
public class OctreeGrid : MonoBehaviour
{
    [Header("World")]
    public float worldSizeX = 4096f;
    public float worldSizeZ = 4096f;

    [Header("Terrain")]
    public float surfaceBaseHeight = 0f;
    public float surfaceNoiseAmplitude = 64f;

    [Header("Octree per cell")]
    public int divisions = 7;
    public int chunkResolution = 16;
    public float innerRadiusPadding = 2f;

    [Header("Render Distance & Performance")]
    [Tooltip("Cells loaded in every direction. 3=7×7, 5=11×11, 10=21×21.")]
    public int renderRadius = 5;

    [Tooltip("Max chunk mesh jobs completed per frame across ALL cells. " +
             "Lower = smoother framerate, higher = faster terrain appearance.")]
    public int maxJobsPerFrame = 80;

    [Tooltip("How many new octree cells to activate per frame when the player " +
             "moves into a new grid cell. 1 = smoothest, higher = faster load.")]
    public int newCellsPerFrame = 2;

    [Tooltip("Max octree node subdivisions created per cell per frame.")]
    public int maxNodeCreationsPerFrame = 50;

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
                  $"maxJobsPerFrame={maxJobsPerFrame}");

        // Seed the pending queue immediately so the first cells load at start
        EnqueueDesiredCells(WorldToCell(priority.position));
        ProcessPendingCells(int.MaxValue); // load all cells synchronously on first frame
    }

    private void Update()
    {
        Vector2Int playerCell = WorldToCell(priority.position);

        if (playerCell != lastPlayerCell)
        {
            lastPlayerCell = playerCell;
            EnqueueDesiredCells(playerCell);
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
        GUILayout.Label($"World repeats every {worldSizeX} × {worldSizeZ}  (toroidal)");
        GUILayout.Label($"Active cells: {activeCells.Count}  Pending: {pendingCells.Count}");
        GUILayout.Label($"Player: {priority.position}");
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
    //  Grid management
    // -----------------------------------------------------------------------

    /// <summary>
    /// Fills pendingCells with every cell in the render radius that is not
    /// already active, sorted nearest-first so they load in the right order.
    /// </summary>
    private void EnqueueDesiredCells(Vector2Int playerCell)
    {
        // Collect missing cells with their distance
        var missing = new List<(Vector2Int cell, float sqrDist)>();

        for (int dx = -renderRadius; dx <= renderRadius; dx++)
            for (int dz = -renderRadius; dz <= renderRadius; dz++)
            {
                var cell = new Vector2Int(playerCell.x + dx, playerCell.y + dz);
                if (!activeCells.ContainsKey(cell))
                {
                    float sqrDist = dx * dx + dz * dz;
                    missing.Add((cell, sqrDist));
                }
            }

        // Sort nearest-first before enqueuing
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