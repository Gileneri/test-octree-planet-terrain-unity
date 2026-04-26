using System;
using System.IO;
using UnityEngine;

/// <summary>
/// ScriptableObject that mirrors every field in a world JSON file
/// (e.g. Earths_Moon.json).
///
/// USAGE
/// -----
/// 1. Assets > Create > Voxel Void > World Config  →  creates an asset.
/// 2. Set "Json File Name" to the file you want (without .json extension).
///    By default the file lives in  Application.streamingAssetsPath/Worlds/.
/// 3. Call LoadFromJson() at runtime to populate the fields from disk,
///    or SaveToJson() to write the current field values back.
/// 4. Attach WorldConfigLoader to the same GameObject as OctreeGrid;
///    it calls LoadFromJson() in Awake and pushes the values into OctreeGrid
///    before OctreeGrid.Start() runs.
///
/// EDITOR WORKFLOW
/// ---------------
/// The custom editor (WorldConfigEditor) shows a "Load", "Save" and
/// "Open in Explorer" button so you can round-trip the JSON without
/// entering Play Mode.
///
/// COMPATIBILITY
/// -------------
/// This class only *reads and writes* data.  It does not modify any logic
/// inside OctreeGrid, Octree, Node, or NodeJob.
/// </summary>
[CreateAssetMenu(menuName = "Voxel Void/World Config", fileName = "WorldConfig")]
public class WorldConfig : ScriptableObject
{
    // ------------------------------------------------------------------
    // Path
    // ------------------------------------------------------------------

    [Header("File")]
    [Tooltip("File name without extension. Resolved inside StreamingAssets/Worlds/.")]
    public string jsonFileName = "Earths_Moon";

    // ------------------------------------------------------------------
    // Identity
    // ------------------------------------------------------------------

    [Header("Identity")]
    public string worldName = "Earths_Moon";

    // ------------------------------------------------------------------
    // Noise
    // ------------------------------------------------------------------

    [Header("Noise")]
    [Tooltip("FastNoiseLite noise type. Available in this build: OpenSimplex2, OpenSimplex2S, Perlin.")]
    public string noiseType = "OpenSimplex2";

    public int seed = 2376;

    // ------------------------------------------------------------------
    // World size
    // ------------------------------------------------------------------

    [Header("World Size")]
    public float worldSizeX = 50000f;
    public float worldSizeZ = 50000f;

    // ------------------------------------------------------------------
    // Terrain
    // ------------------------------------------------------------------

    [Header("Terrain")]
    public float surfaceBaseHeight = 0f;
    public float surfaceNoiseAmplitude = 300f;

    [Tooltip("Lowest Y coordinate where solid terrain can generate. " +
             "Informational only — not yet wired into NodeJob density. " +
             "Extend IsAirWorldPos if you want a hard floor.")]
    public float minSubsurfaceHeight = -500f;

    // ------------------------------------------------------------------
    // Octree
    // ------------------------------------------------------------------

    [Header("Octree")]
    [Tooltip("Octree depth. 7 with chunkResolution=8 gives ~512 unit cells.")]
    [Range(1, 12)]
    public int divisions = 7;

    [Tooltip("Voxels per chunk side.")]
    [Range(4, 32)]
    public int chunkResolution = 8;

    // ------------------------------------------------------------------
    // Render
    // ------------------------------------------------------------------

    [Header("Render")]
    [Tooltip("Cells loaded in each direction. 6 = 13x13 grid.")]
    [Range(1, 20)]
    public int renderRadius = 6;

    // ------------------------------------------------------------------
    // Derived (read-only, shown in inspector via custom editor)
    // ------------------------------------------------------------------

    /// <summary>World-space side length of one octree cell.</summary>
    public float CellSize => chunkResolution * Mathf.Pow(2f, divisions - 1);

    /// <summary>Total cells in the active grid (one axis).</summary>
    public int GridSide => renderRadius * 2 + 1;

    // ------------------------------------------------------------------
    // JSON I/O
    // ------------------------------------------------------------------

    public string ResolvedPath =>
        Path.Combine(Application.streamingAssetsPath, "Worlds", jsonFileName + ".json");

    /// <summary>
    /// Loads all fields from the JSON file on disk.
    /// Safe to call at runtime or from the editor.
    /// </summary>
    public bool LoadFromJson()
    {
        string path = ResolvedPath;
        if (!File.Exists(path))
        {
            Debug.LogWarning($"[WorldConfig] File not found: {path}");
            return false;
        }

        try
        {
            string json = File.ReadAllText(path);
            JsonPayload p = JsonUtility.FromJson<JsonPayload>(json);
            ApplyPayload(p);
            Debug.Log($"[WorldConfig] Loaded '{worldName}' from {path}");
            return true;
        }
        catch (Exception e)
        {
            Debug.LogError($"[WorldConfig] Failed to parse {path}: {e.Message}");
            return false;
        }
    }

    /// <summary>
    /// Writes all current field values back to the JSON file.
    /// Creates the Worlds directory if it doesn't exist.
    /// </summary>
    public bool SaveToJson()
    {
        string path = ResolvedPath;
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            string json = JsonUtility.ToJson(BuildPayload(), prettyPrint: true);
            File.WriteAllText(path, json);
            Debug.Log($"[WorldConfig] Saved '{worldName}' to {path}");
            return true;
        }
        catch (Exception e)
        {
            Debug.LogError($"[WorldConfig] Failed to save {path}: {e.Message}");
            return false;
        }
    }

    // ------------------------------------------------------------------
    // Internal serialisation helpers
    // ------------------------------------------------------------------

    private void ApplyPayload(JsonPayload p)
    {
        worldName = p.name;
        noiseType = p.noiseType;
        seed = p.seed;
        worldSizeX = p.worldSizeX;
        worldSizeZ = p.worldSizeZ;
        surfaceBaseHeight = p.surfaceBaseHeight;
        surfaceNoiseAmplitude = p.surfaceNoiseAmplitude;
        minSubsurfaceHeight = p.minSubsurfaceHeight;
        divisions = p.divisions;
        chunkResolution = p.chunkResolution;
        renderRadius = p.renderRadius;
    }

    private JsonPayload BuildPayload() => new JsonPayload
    {
        name = worldName,
        noiseType = noiseType,
        seed = seed,
        worldSizeX = worldSizeX,
        worldSizeZ = worldSizeZ,
        surfaceBaseHeight = surfaceBaseHeight,
        surfaceNoiseAmplitude = surfaceNoiseAmplitude,
        minSubsurfaceHeight = minSubsurfaceHeight,
        divisions = divisions,
        chunkResolution = chunkResolution,
        renderRadius = renderRadius,
    };

    // Matches the exact key names used by the web editor
    [Serializable]
    private class JsonPayload
    {
        public string name;
        public string noiseType;
        public int seed;
        public float worldSizeX;
        public float worldSizeZ;
        public float surfaceBaseHeight;
        public float surfaceNoiseAmplitude;
        public float minSubsurfaceHeight;
        public int divisions;
        public int chunkResolution;
        public int renderRadius;
    }
}