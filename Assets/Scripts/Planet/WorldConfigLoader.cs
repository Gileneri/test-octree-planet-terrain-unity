using UnityEngine;

/// <summary>
/// Sits on the same GameObject as OctreeGrid and pushes WorldConfig
/// values into it during Awake — before OctreeGrid.Start() runs.
///
/// SETUP
/// -----
/// 1. Add this component to the GameObject that also has OctreeGrid.
/// 2. Drag your WorldConfig ScriptableObject asset into the "Config" slot.
/// 3. Toggle "Load From Json On Awake" if you want the JSON file on disk
///    to override the ScriptableObject's serialised values at launch
///    (useful so designers can edit Earths_Moon.json without reopening Unity).
///
/// WHAT IT TOUCHES
/// ---------------
/// Only the public fields that OctreeGrid.Start() reads:
///   worldSizeX, worldSizeZ, surfaceBaseHeight, surfaceNoiseAmplitude,
///   divisions, chunkResolution, renderRadius.
///
/// It does NOT touch: priority, chunkPrefab, playerCamera, lodDistanceCurve,
/// maxJobsPerFrame, newCellsPerFrame, maxNodeCreationsPerFrame,
/// predictionDistance, predictionSpeedThreshold, predictionStrength.
/// Those remain configured in the Inspector as before.
///
/// NOISE / SEED
/// ------------
/// noiseType and seed are stored in WorldConfig but live inside NodeJob
/// (a Burst IJob).  NodeJob is a struct created per-frame; there is no
/// persistent instance to patch at runtime.  Instead, use the companion
/// helper NodeJobNoiseSettings (see bottom of this file) — if you create
/// it, NodeJob will read from it instead of its hard-coded defaults.
/// </summary>
[RequireComponent(typeof(OctreeGrid))]
public class WorldConfigLoader : MonoBehaviour
{
    [Tooltip("The WorldConfig ScriptableObject for this world.")]
    public WorldConfig config;

    [Tooltip("If true, LoadFromJson() is called in Awake so the .json file " +
             "on disk always wins over the ScriptableObject's saved values.")]
    public bool loadFromJsonOnAwake = true;

    // ------------------------------------------------------------------

    private OctreeGrid grid;

    private void Awake()
    {
        if (config == null)
        {
            Debug.LogError("[WorldConfigLoader] No WorldConfig assigned. " +
                           "OctreeGrid will use its own Inspector values.");
            return;
        }

        if (loadFromJsonOnAwake)
            config.LoadFromJson();   // disk → ScriptableObject fields

        grid = GetComponent<OctreeGrid>();
        PushToGrid();
    }

    // ------------------------------------------------------------------
    //  Push
    // ------------------------------------------------------------------

    /// <summary>
    /// Copies WorldConfig fields into OctreeGrid.
    /// Call this at runtime if you hot-reload the config.
    /// </summary>
    public void PushToGrid()
    {
        if (grid == null) grid = GetComponent<OctreeGrid>();
        if (config == null || grid == null) return;

        grid.worldSizeX = config.worldSizeX;
        grid.worldSizeZ = config.worldSizeZ;
        grid.surfaceBaseHeight = config.surfaceBaseHeight;
        grid.surfaceNoiseAmplitude = config.surfaceNoiseAmplitude;
        grid.noiseSeed = config.seed;
        grid.noiseTypeId = NodeJob.NoiseTypeToId(config.noiseType);
        grid.minSubsurfaceHeight = config.minSubsurfaceHeight;
        grid.divisions = config.divisions;
        grid.chunkResolution = config.chunkResolution;
        grid.renderRadius = config.renderRadius;

        Debug.Log($"[WorldConfigLoader] Applied '{config.worldName}' → OctreeGrid. " +
                  $"World={config.worldSizeX}×{config.worldSizeZ}  " +
                  $"SurfaceBase={config.surfaceBaseHeight}  " +
                  $"Seed={config.seed}  NoiseType={config.noiseType}(id={grid.noiseTypeId})  " +
                  $"MinSubY={config.minSubsurfaceHeight}  " +
                  $"Divisions={config.divisions}  ChunkRes={config.chunkResolution}  " +
                  $"RenderRadius={config.renderRadius}");
    }

    // ------------------------------------------------------------------
    //  Runtime hot-reload
    // ------------------------------------------------------------------

    /// <summary>
    /// Re-reads the JSON from disk, pushes to OctreeGrid, and rebuilds
    /// all active octree cells so the change is visible immediately.
    /// Safe to call from a dev console or a settings UI button.
    /// </summary>
    public void HotReload()
    {
        if (config == null || grid == null) return;

        bool ok = config.LoadFromJson();
        if (!ok) return;

        PushToGrid();
        grid.RebakeLodRadii();   // rebuilds all active Octree cells
        Debug.Log("[WorldConfigLoader] Hot-reload complete.");
    }
}