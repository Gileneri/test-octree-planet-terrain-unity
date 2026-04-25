#if UNITY_EDITOR
using System.Diagnostics;
using System.IO;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Custom Inspector for WorldConfig.
///
/// Adds three buttons below the standard fields:
///   [Load from JSON]   — reads the file on disk into the ScriptableObject
///   [Save to JSON]     — writes the ScriptableObject fields back to disk
///   [Open JSON file]   — opens the file in the OS default editor
///
/// Also shows read-only derived values (cell size, grid dimensions).
///
/// Place this file anywhere inside an Editor/ folder in your project.
/// </summary>
[CustomEditor(typeof(WorldConfig))]
public class WorldConfigEditor : Editor
{
    private GUIStyle _derivedStyle;

    public override void OnInspectorGUI()
    {
        var cfg = (WorldConfig)target;

        DrawDefaultInspector();

        EditorGUILayout.Space(8);
        EditorGUILayout.LabelField("Derived (read-only)", EditorStyles.boldLabel);

        if (_derivedStyle == null)
        {
            _derivedStyle = new GUIStyle(EditorStyles.helpBox)
            {
                fontSize = 11,
                richText = true
            };
        }

        float cellSize = cfg.chunkResolution * Mathf.Pow(2f, cfg.divisions - 1);
        int   gridSide = cfg.renderRadius * 2 + 1;

        EditorGUILayout.LabelField(
            $"Cell size: <b>{cellSize:F0} units</b>   " +
            $"Grid: <b>{gridSide}×{gridSide} cells</b>",
            _derivedStyle);

        EditorGUILayout.Space(8);
        EditorGUILayout.LabelField("JSON", EditorStyles.boldLabel);

        string path = cfg.ResolvedPath;
        bool   exists = File.Exists(path);

        EditorGUILayout.LabelField(
            "Path: " + (exists
                ? $"<color=#4CAF50>{path}</color>"
                : $"<color=#F44336>{path} (not found)</color>"),
            new GUIStyle(EditorStyles.miniLabel) { richText = true, wordWrap = true });

        EditorGUILayout.Space(4);

        EditorGUILayout.BeginHorizontal();

        // ── Load ──
        using (new EditorGUI.DisabledScope(!exists))
        {
            if (GUILayout.Button("Load from JSON", GUILayout.Height(28)))
            {
                Undo.RecordObject(cfg, "Load WorldConfig from JSON");
                cfg.LoadFromJson();
                EditorUtility.SetDirty(cfg);
                AssetDatabase.SaveAssets();
            }
        }

        // ── Save ──
        if (GUILayout.Button("Save to JSON", GUILayout.Height(28)))
        {
            cfg.SaveToJson();
        }

        // ── Open ──
        using (new EditorGUI.DisabledScope(!exists))
        {
            if (GUILayout.Button("Open JSON file", GUILayout.Height(28)))
            {
                Process.Start(new ProcessStartInfo(path) { UseShellExecute = true });
            }
        }

        EditorGUILayout.EndHorizontal();

        // ── Hot-reload at runtime ──
        if (Application.isPlaying)
        {
            EditorGUILayout.Space(4);
            if (GUILayout.Button("Hot-Reload (apply to running scene)", GUILayout.Height(28)))
            {
                var loader = FindObjectOfType<WorldConfigLoader>();
                if (loader != null && loader.config == cfg)
                    loader.HotReload();
                else
                    UnityEngine.Debug.LogWarning(
                        "[WorldConfigEditor] No WorldConfigLoader found in scene " +
                        "referencing this config.");
            }
        }
    }
}
#endif
